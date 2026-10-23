import argparse
import csv
import itertools
import json
import math
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

from billing_provenance import SourceAudit, build_analysis, numeric_value
from adoption_tool import inventory_rows


DEFAULT_NAMING_CONVENTION = {
    "version": 1,
    "source": "microsoft-azure-caf",
    "description": "CAF: <resource_type>-<workload>-<environment>-<region>-<instance>",
    "default": {
        "delimiter": "-",
        "pattern": ["resource_type", "workload", "environment", "region", "instance"],
    },
    "providers": {
        "aws": {},
        "azure": {},
        "gcp": {},
    },
}

DEFAULT_FIELD_ALIASES = {
    "product": ["product", "workload", "app", "application", "service"],
    "platform": ["platform", "team", "org", "organization", "business_unit", "bu"],
    "environment": ["environment", "env", "stage"],
    "resource_type": ["resource_type", "resource", "rtype", "service", "svc", "type"],
    "region": ["region", "location", "geo"],
    "instance": ["instance", "inst", "ordinal", "id", "index"],
}

ENVIRONMENT_ALIASES = {
    "prod": "prod",
    "production": "prod",
    "prd": "prod",
    "dev": "dev",
    "development": "dev",
    "test": "test",
    "qa": "test",
    "uat": "test",
    "stg": "staging",
    "stage": "staging",
    "staging": "staging",
}


def normalize_token(token: str) -> str:
    return token.strip().lower().replace(" ", "_")


def normalize_environment(value: str) -> str:
    if not value:
        return ""
    key = value.strip().lower()
    return ENVIRONMENT_ALIASES.get(key, value)


def normalize_convention_config(config: dict) -> dict:
    if "default" in config or "providers" in config:
        base = config.get("default", {})
        providers = config.get("providers") or {}
    else:
        base = {k: config.get(k) for k in ("pattern", "delimiter", "field_map") if k in config}
        providers = config.get("providers") or {}

    default_cfg = {
        "delimiter": base.get("delimiter", "-"),
        "pattern": base.get("pattern", []),
        "field_map": base.get("field_map", {}),
    }
    return {
        "version": config.get("version", 1),
        "source": config.get("source", ""),
        "description": config.get("description", ""),
        "default": default_cfg,
        "providers": providers,
    }


def merge_convention(base: dict, override: Optional[dict]) -> dict:
    merged = dict(base or {})
    if override:
        for key, value in override.items():
            merged[key] = value
    return merged


def select_convention(config: dict, provider: str) -> dict:
    normalized = normalize_convention_config(config)
    base = normalized["default"]
    overrides = normalized["providers"].get(provider, {})
    return merge_convention(base, overrides)


def load_naming_convention(path: Optional[pathlib.Path]) -> dict:
    if not path:
        return DEFAULT_NAMING_CONVENTION
    if not path.exists():
        raise FileNotFoundError(f"Naming convention file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def detect_provider(resource_id: str) -> str:
    if not resource_id:
        return "unknown"
    rid = resource_id.strip().lower()
    if rid.startswith("arn:"):
        return "aws"
    if rid.startswith("/subscriptions/") or "/providers/microsoft." in rid or "azure.com" in rid:
        return "azure"
    if rid.startswith("projects/") or "/projects/" in rid or "googleapis.com" in rid or rid.startswith("//"):
        return "gcp"
    return "unknown"


def normalize_resource_name(resource_id: str, provider: str) -> str:
    if not resource_id:
        return ""
    rid = resource_id.strip()
    if provider == "aws":
        if rid.lower().startswith("arn:"):
            tail = rid.split(":", 5)[-1]
            return tail.split("/")[-1]
        return rid.split("/")[-1]
    if provider in ("azure", "gcp"):
        return rid.rstrip("/").split("/")[-1]
    return rid


def split_name_parts(name: str, delimiter: str) -> List[str]:
    if not name or not delimiter:
        return []
    return [part.strip() for part in name.split(delimiter)]


def map_parts_to_tokens(parts: List[str], tokens: List[str], delimiter: str) -> Dict[str, str]:
    if not tokens:
        return {}
    if len(parts) > len(tokens):
        head = parts[: len(tokens) - 1]
        tail = delimiter.join(parts[len(tokens) - 1 :]).strip()
        parts = head + [tail]
    token_map: Dict[str, str] = {}
    for idx, token in enumerate(tokens):
        token_map[token] = parts[idx] if idx < len(parts) else ""
    return token_map


def resolve_naming_fields(token_map: Dict[str, str], field_map: Dict[str, str]) -> Dict[str, str]:
    def resolve(field: str) -> str:
        token = field_map.get(field)
        if token:
            return token_map.get(token, "")
        for alias in DEFAULT_FIELD_ALIASES.get(field, []):
            val = token_map.get(alias, "")
            if val:
                return val
        return ""

    return {
        "product": resolve("product"),
        "platform": resolve("platform"),
        "environment": resolve("environment"),
        "resource_type": resolve("resource_type"),
        "region": resolve("region"),
        "instance": resolve("instance"),
    }


def is_opaque_resource_name(name: str) -> bool:
    """Cloud-generated IDs are identifiers, not ownership naming conventions."""
    return bool(re.fullmatch(
        r"(?:i|vol|snap|ami|eni|vpc|subnet|sg)-[0-9a-f]+|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|\d+",
        name, re.IGNORECASE,
    ))


def normalize_provider(value: str) -> str:
    key = (value or "").strip().lower()
    return {
        "amazon web services": "aws", "amazon web services, inc.": "aws", "amazon": "aws",
        "microsoft": "azure", "microsoft azure": "azure",
        "google": "gcp", "google cloud": "gcp", "google cloud platform": "gcp",
    }.get(key, key or "unknown")


def parse_resource_naming(resource_id: str, naming_config: dict, provider: str = "unknown") -> dict:
    if provider == "unknown":
        provider = detect_provider(resource_id)
    resource_name = normalize_resource_name(resource_id, provider)
    convention = select_convention(naming_config, provider)
    delimiter = convention.get("delimiter", "-")
    raw_pattern = convention.get("pattern") or []
    if isinstance(raw_pattern, str):
        raw_pattern = [p.strip() for p in raw_pattern.split(",")]
    pattern = [normalize_token(t) for t in raw_pattern if t]
    field_map_raw = convention.get("field_map", {}) or {}
    field_map = {normalize_token(k): normalize_token(v) for k, v in field_map_raw.items() if k and v}

    token_map: Dict[str, str] = {}
    if resource_name and pattern and delimiter and not is_opaque_resource_name(resource_name):
        parts = split_name_parts(resource_name, delimiter)
        token_map = map_parts_to_tokens(parts, pattern, delimiter)

    fields = resolve_naming_fields(token_map, field_map)
    if fields.get("environment"):
        fields["environment"] = normalize_environment(fields["environment"])

    return {
        "provider": provider,
        "resource_name": resource_name,
        "tokens": token_map,
        "fields": fields,
    }


def infer_schema(fieldnames: List[str]) -> dict:
    fields = set(fieldnames or [])
    lower_map = {name.lower(): name for name in fields}

    def pick(*names: str) -> Optional[str]:
        for name in names:
            if name in fields:
                return name
            lower = name.lower()
            if lower in lower_map:
                return lower_map[lower]
        return None

    if pick("BilledCost") or pick("EffectiveCost"):
        service_fields = [pick("ServiceCategory"), pick("ServiceName")]
        return {
            "schema": "focus",
            "provider": "unknown",
            "provider_column": pick("ProviderName"),
            "cost": pick("BilledCost", "EffectiveCost"),
            "service_fields": [field for field in service_fields if field],
            "resource_id": pick("ResourceId", "ResourceName"),
            "product_tag": pick("tags/Product"),
            "environment_tag": pick("tags/Environment"),
            "platform_tag": pick("tags/Platform"),
            "json_tags": pick("Tags"),
            "account": pick("SubAccountName", "SubAccountId"),
            "currency": pick("BillingCurrency"),
            "usage_start": pick("ChargePeriodStart"),
            "usage_end": pick("ChargePeriodEnd"),
            "billing_start": pick("BillingPeriodStart"),
            "billing_end": pick("BillingPeriodEnd"),
            "cost_basis": "billed" if pick("BilledCost") else "effective",
            "usage_unit": pick("ConsumedUnit", "UsageUnit"),
            "usage_amount": pick("ConsumedQuantity", "UsageQuantity"),
            "usage_type": pick("SkuId", "SkuPriceId"),
            "sku_name": pick("SkuId"),
        }
    if pick("lineItem/UnblendedCost"):
        service_fields = [pick("lineItem/ProductCode")]
        return {
            "schema": "aws",
            "provider": "aws",
            "cost": pick("lineItem/UnblendedCost"),
            "service_fields": [field for field in service_fields if field],
            "resource_id": pick("resourceId", "lineItem/ResourceId"),
            "product_tag": pick("resourceTags/user:Product", "tags/Product"),
            "environment_tag": pick("resourceTags/user:Environment", "tags/Environment"),
            "platform_tag": pick("resourceTags/user:Platform", "tags/Platform"),
            "currency": pick("lineItem/CurrencyCode"),
            "usage_start": pick("lineItem/UsageStartDate"),
            "usage_end": pick("lineItem/UsageEndDate"),
            "billing_start": pick("bill/BillingPeriodStartDate"),
            "billing_end": pick("bill/BillingPeriodEndDate"),
            "usage_unit": pick("lineItem/UsageUnit"),
            "usage_amount": pick("lineItem/UsageAmount"),
            "usage_type": pick("lineItem/UsageType"),
            "sku_name": None,
        }
    if pick("Cost") and pick("InstanceId", "ResourceId", "resourceId"):
        service_fields = [pick("ServiceName"), pick("ConsumedService"), pick("MeterCategory"), pick("MeterSubCategory")]
        return {
            "schema": "azure",
            "provider": "azure",
            "cost": pick("Cost"),
            "service_fields": [field for field in service_fields if field],
            "resource_id": pick("InstanceId", "ResourceId", "resourceId"),
            "product_tag": pick("tags/Product"),
            "environment_tag": pick("tags/Environment"),
            "platform_tag": pick("tags/Platform"),
            "json_tags": pick("Tags"),
            "currency": pick("BillingCurrencyCode", "BillingCurrency", "Currency"),
            "usage_start": pick("Date", "UsageDateTime", "UsageStartDate"),
            "usage_end": pick("UsageEndDate"),
            "billing_start": pick("BillingPeriodStartDate"),
            "billing_end": pick("BillingPeriodEndDate"),
            "usage_unit": pick("UnitOfMeasure"),
            "usage_amount": pick("Quantity"),
            "usage_type": pick("ServiceName", "MeterName"),
            "sku_name": pick("SkuName", "MeterName"),
        }
    if pick("Cost") and ((pick("LineItem") and pick("UsageStartDate")) or pick("service.description")):  # Flat GCP export
        return {
            "schema": "gcp",
            "provider": "gcp",
            "cost": pick("Cost"),
            "service_fields": [pick("ServiceDescription", "service.description"), pick("SkuDescription", "sku.description")],
            "resource_id": pick("resource.global_name", "resource.name", "LineItem"),
            "product_tag": pick("tags/Product"),
            "environment_tag": pick("tags/Environment"),
            "platform_tag": pick("tags/Platform"),
            "json_tags": pick("Labels", "project.labels"),
            "currency": pick("Currency", "BillingCurrency"),
            "usage_start": pick("UsageStartDate", "usage_start_time"),
            "usage_end": pick("UsageEndDate", "usage_end_time"),
            "usage_unit": pick("UsageUnit"),
            "usage_amount": pick("UsageAmount"),
            "usage_type": pick("SkuDescription"),
            "sku_name": pick("SkuDescription"),
        }
    raise ValueError("Unsupported CSV schema: missing required cost/resource columns.")


AWS_SERVICE_MAP = {
    "AmazonEC2": "Compute/VM",
    "AmazonEBS": "Block Storage",
    "AmazonS3": "Object Storage",
    "AmazonRDS": "Database",
    "AmazonDynamoDB": "Database",
    "AmazonElastiCache": "Cache",
    "AmazonVPC": "Networking",
    "ElasticLoadBalancing": "Networking",
    "AmazonEKS": "Container",
    "AmazonECS": "Container",
    "AmazonECR": "Container",
    "AWSLambda": "Serverless",
}

SERVICE_CATEGORY_RULES = [
    ("Block Storage", ["ebs", "managed disk", "managed disks", "disk", "persistent disk", "block storage", "microsoft.compute/disks"]),
    ("Object Storage", ["s3", "blob", "object storage", "storage account", "storage accounts", "cloud storage", "gcs", "microsoft.storage"]),
    ("Database", ["rds", "sql database", "database", "cosmos", "dynamodb", "spanner", "cloud sql", "microsoft.sql"]),
    ("Cache", ["redis", "cache", "elasticache", "memorydb", "microsoft.cache"]),
    ("Container", ["kubernetes", "container service", "container", "aks", "eks", "gke", "ecs", "fargate", "microsoft.containerservice"]),
    ("Networking", ["vpc", "elasticloadbalancing", "load balancer", "network", "cdn", "front door", "nat", "gateway", "public ip"]),
    ("Serverless", ["lambda", "functions", "cloud functions", "app service", "cloud run"]),
    ("Compute/VM", ["ec2", "virtual machine", "virtual machines", "microsoft.compute", "compute engine", "vm", "vmss"]),
]

AZURE_VM_SKU_MAP = {
    "Standard_D2s_v5": {"vcpu": 2, "ram_gib": 8},
    "Standard_D4s_v5": {"vcpu": 4, "ram_gib": 16},
    "Standard_E4s_v5": {"vcpu": 4, "ram_gib": 32},
    "Standard_F8s_v2": {"vcpu": 8, "ram_gib": 16},
    "Standard_B2ms": {"vcpu": 2, "ram_gib": 8},
}

AZURE_SERIES_RAM_PER_VCPU = {
    "B": 4,
    "D": 4,
    "E": 8,
    "F": 2,
}


def normalize_service(provider: str, service_values: List[str], resource_id: str) -> str:
    raw = next((val for val in service_values if val), "")
    
    if provider == "aws":
        if raw in AWS_SERVICE_MAP:
            return AWS_SERVICE_MAP[raw]
    
    if provider == "gcp":
        # GCP Service Descriptions can be verbose
        if "compute engine" in raw.lower():
            return "Compute/VM"
        if "cloud storage" in raw.lower():
            return "Object Storage"
        if "cloud sql" in raw.lower() or "spanner" in raw.lower():
            return "Database"
        if "kubernetes" in raw.lower():
            return "Container"
        if "cloud functions" in raw.lower() or "cloud run" in raw.lower():
            return "Serverless"
        if "load balancing" in raw.lower():
            return "Networking"
        
    combined = " ".join([val for val in service_values if val] + ([resource_id] if resource_id else []))
    combined_lower = combined.lower()
    for category, keywords in SERVICE_CATEGORY_RULES:
        if any(keyword in combined_lower for keyword in keywords):
            return category
            
    return raw or "UnknownService"


def parse_azure_vm_specs(sku_name: str) -> Optional[Dict[str, float]]:
    if not sku_name:
        return None
    sku = sku_name.strip()
    if sku in AZURE_VM_SKU_MAP:
        return AZURE_VM_SKU_MAP[sku]
    match = re.search(r"(?:Standard|Basic)_([A-Za-z]+)(\d+)", sku)
    if not match:
        return None
    series = match.group(1).upper()
    vcpu = int(match.group(2))
    ratio = AZURE_SERIES_RAM_PER_VCPU.get(series[:1])
    ram_gib = vcpu * ratio if ratio else 0
    return {"vcpu": vcpu, "ram_gib": ram_gib}


def is_size_unit(unit: str) -> bool:
    return "gb" in (unit or "").lower()


def vm_combo_label(vcpu: int, ram_gib: float) -> str:
    if vcpu and ram_gib:
        return f"{vcpu} vCPU / {ram_gib} GiB"
    return "Unknown size"

def select_dimension_value(naming_value: str, tag_value: str, mode: str) -> str:
    naming_value = clean_dimension(naming_value)
    tag_value = clean_dimension(tag_value)
    if mode == "override":
        return naming_value or tag_value
    if tag_value:
        return tag_value
    return naming_value or tag_value


def clean_dimension(value) -> str:
    text = str(value).strip() if isinstance(value, (str, int, float)) else ""
    return "" if text.lower() in {"", "untagged", "unknown", "unallocated", "none", "null"} else text


def classify_resource_name(resource_id: str) -> str:
    """Heuristic to bucket resources by naming convention/prefix."""
    if not resource_id:
        return "unknown"
    rid = resource_id.strip()
    if rid.startswith("arn:"):
        tail = rid.split(":", 5)[-1]
        tail = tail.split("/")[-1]
        return tail.split("-")[0] if "-" in tail else tail.split(":")[0]
    token = rid.split("-")[0]
    return token or "unknown"


def extract_platform(resource_id: str) -> str:
    """Heuristic to extract a platform marker from a resource name like 'disk-PlatformB-dev-0001'."""
    if not resource_id or is_opaque_resource_name(resource_id):
        return "unknown"
    parts = resource_id.split("-")
    if len(parts) >= 2:
        return parts[1] or "unknown"
    return "unknown"


def bucketize_resources(resources: Dict[str, float], include_resources: bool = False):
    """Bucket resource usage amounts (e.g., GB-Mo) into coarse ranges."""
    buckets = [
        (0, 100, "0-100"),
        (100, 500, "100-500"),
        (500, 1000, "500-1k"),
        (1000, 5000, "1k-5k"),
        (5000, 10000, "5k-10k"),
        (10000, math.inf, "10k+"),
    ]
    agg = []
    for low, high, label in buckets:
        res_ids = [rid for rid, amt in resources.items() if low <= amt < high]
        total = sum(resources[r] for r in res_ids)
        if res_ids:
            bucket = {"label": label, "resources": len(res_ids), "gbMo": round(total, 2)}
            if include_resources:
                ordered = sorted(res_ids, key=lambda r: resources[r], reverse=True)
                bucket["resourceIds"] = [{"id": rid, "amount": round(resources[rid], 2)} for rid in ordered]
            agg.append(bucket)
    return agg


def parse_json_tags(label_string: str) -> Dict[str, str]:
    """Read tag objects, key/value label arrays, and prefixed GCP label objects."""
    if not label_string:
        return {}
    try:
        text = label_string.strip()
        if not text.startswith(("{", "[")):
            text = text[text.index("{"):]
        data = json.loads(text)
        if isinstance(data, list):
            data = {item["key"]: item.get("value") for item in data if isinstance(item, dict) and "key" in item}
        if not isinstance(data, dict):
            return {}
        tags = {str(key).lower(): clean_dimension(value) for key, value in data.items()}
        return {
            "product": tags.get("product", ""),
            "environment": tags.get("environment") or tags.get("env", ""),
            "platform": tags.get("platform", ""),
        }
    except (ValueError, TypeError):
        return {}


def load_cur(path: pathlib.Path, naming_config: dict, naming_mode: str = "fallback", assumed_currency: Optional[str] = None):
    inventory_evidence = []
    prod_cost: Dict[str, float] = defaultdict(float)
    env_cost: Dict[str, float] = defaultdict(float)
    svc_cost: Dict[str, float] = defaultdict(float)
    prod_svc_cost: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    prod_svc_res: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    prod_rows: Dict[str, int] = defaultdict(int)
    prod_service_sets: Dict[str, Set[str]] = defaultdict(set)

    res_name_cost: Dict[str, float] = defaultdict(float)
    res_name_count: Dict[str, int] = defaultdict(int)
    platform_cost: Dict[str, float] = defaultdict(float)
    platform_count: Dict[str, int] = defaultdict(int)
    platform_res_name_cost: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_res_name_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    platform_ebs_gbmo: Dict[str, float] = defaultdict(float)
    platform_s3_gbmo: Dict[str, float] = defaultdict(float)

    platform_ec2_hours: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_ec2_instances: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    platform_vm_types: Dict[str, Dict[str, str]] = defaultdict(dict)

    platform_rows: Dict[str, int] = defaultdict(int)
    platform_product_cost: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    product_provider_cost = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    platform_env_cost: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_service_cost: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_product_rows: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    platform_env_rows: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    platform_service_rows: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    platform_product_service_cost: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    platform_product_env_cost: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    platform_product_res_name_cost: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    platform_product_res_name_count: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    platform_product_ebs_gbmo: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_product_s3_gbmo: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_product_ec2_hours: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    platform_product_ec2_instances: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    platform_product_vm_types: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    platform_ebs_resources: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_s3_resources: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_product_ebs_resources: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    platform_product_s3_resources: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )

    platform_vm_combo_hours: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_vm_combo_instances: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    platform_vm_combo_specs: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    platform_product_vm_combo_hours: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    platform_product_vm_combo_instances: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    platform_product_vm_combo_specs: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    platform_block_resources: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_object_resources: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    platform_product_block_resources: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    platform_product_object_resources: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )

    provider_cost: Dict[str, float] = defaultdict(float)
    provider_rows: Dict[str, int] = defaultdict(int)
    provider_res_name_cost: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    provider_res_name_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    instance_info = {
        "c5.xlarge": {"vcpu": 4, "ram_gib": 8},
        "m5.large": {"vcpu": 2, "ram_gib": 8},
        "t3.medium": {"vcpu": 2, "ram_gib": 4},
    }

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        schema = infer_schema(reader.fieldnames or [])
        audit = SourceAudit(path, schema, assumed_currency)
        inventory_headers = {name.lower(): name for name in reader.fieldnames or []}

        def scope_value(row_data, *names):
            return next(((row_data.get(inventory_headers.get(name.lower())) or "").strip() for name in names if (row_data.get(inventory_headers.get(name.lower())) or "").strip()), "")

        def read_value(row_data, key):
            return (row_data.get(key) or "").strip() if key else ""

        for row in reader:
            if None in row:
                raise ValueError(f"{path.name}: row {reader.line_num}: extra columns; check CSV quoting.")
            cost_decimal = numeric_value(read_value(row, schema["cost"]), path, reader.line_num, schema["cost"])
            cost = float(cost_decimal)
            audit.observe(row, cost_decimal, reader.line_num)
            rid = read_value(row, schema["resource_id"]) or ""
            usage_unit = read_value(row, schema["usage_unit"])
            usage_amount = float(numeric_value(read_value(row, schema["usage_amount"]), path, reader.line_num, schema["usage_amount"], optional=True))
            usage_type = read_value(row, schema["usage_type"]) or ""
            sku_name = read_value(row, schema.get("sku_name"))

            service_values = [read_value(row, field) for field in schema.get("service_fields", [])]
            raw_service = next((val for val in service_values if val), "")

            product_tag = read_value(row, schema["product_tag"])
            env_tag = read_value(row, schema["environment_tag"])
            platform_tag = read_value(row, schema["platform_tag"])

            provider = normalize_provider(read_value(row, schema.get("provider_column")) or schema.get("provider"))
            if provider == "unknown":
                provider = detect_provider(rid)
            inventory_evidence.append({"provider": provider, "resourceId": rid, "cost": cost_decimal, "service": raw_service,
                "account": scope_value(row, "lineItem/UsageAccountId", "project.id", "ProjectId", "SubAccountId", "SubscriptionId"),
                "region": scope_value(row, "product/regionCode", "location.region", "RegionId", "Region"),
                "zone": scope_value(row, "lineItem/AvailabilityZone", "location.zone", "Zone")})
            naming = parse_resource_naming(rid, naming_config, provider)
            fields = naming["fields"]

            svc = normalize_service(provider, service_values, rid)

            raw_tags = read_value(row, schema.get("json_tags"))
            tags = parse_json_tags(raw_tags)
            if raw_tags and not tags:
                audit.invalid_tag_rows += 1
            product_tag = clean_dimension(product_tag) or tags.get("product", "")
            env_tag = clean_dimension(env_tag) or tags.get("environment", "")
            platform_tag = clean_dimension(platform_tag) or tags.get("platform", "")

            prod = select_dimension_value(fields.get("product"), product_tag, naming_mode) or "UnTagged"
            env = normalize_environment(select_dimension_value(fields.get("environment"), env_tag, naming_mode)) or "UnTagged"
            platform_value = select_dimension_value(fields.get("platform"), platform_tag, naming_mode)
            if not platform_value:
                platform_value = read_value(row, schema.get("account")) or extract_platform(naming["resource_name"] or rid)
            platform = platform_value or "unknown"

            prod_cost[prod] += cost
            env_cost[env] += cost
            svc_cost[svc] += cost
            prod_svc_cost[prod][svc] += cost
            prod_svc_res[prod][svc].add(rid)
            prod_rows[prod] += 1
            prod_service_sets[prod].add(svc)

            res_key = fields.get("resource_type") or classify_resource_name(naming["resource_name"] or rid)
            res_name_cost[res_key] += cost
            res_name_count[res_key] += 1

            platform_cost[platform] += cost
            platform_count[platform] += 1
            platform_res_name_cost[platform][res_key] += cost
            platform_res_name_count[platform][res_key] += 1
            platform_rows[platform] += 1
            platform_product_cost[platform][prod] += cost
            product_provider_cost[platform][prod][provider] += cost
            platform_env_cost[platform][env] += cost
            platform_service_cost[platform][svc] += cost
            platform_product_rows[platform][prod] += 1
            platform_env_rows[platform][env] += 1
            platform_service_rows[platform][svc] += 1
            platform_product_service_cost[platform][prod][svc] += cost
            platform_product_env_cost[platform][prod][env] += cost
            platform_product_res_name_cost[platform][prod][res_key] += cost
            platform_product_res_name_count[platform][prod][res_key] += 1

            if svc == "Compute/VM":
                if provider == "aws" and raw_service == "AmazonEC2" and usage_type.startswith("BoxUsage:"):
                    sku = usage_type.split(":", 1)[1]
                    info = instance_info.get(sku, {"vcpu": 0, "ram_gib": 0})
                    vcpu = info["vcpu"]
                    ram = info["ram_gib"]
                    combo = vm_combo_label(vcpu, ram)
                    platform_vm_combo_hours[platform][combo] += usage_amount
                    platform_vm_combo_instances[platform][combo].add(rid)
                    platform_vm_combo_specs[platform][combo] = {"vcpu": vcpu, "ramGiB": ram}
                    platform_product_vm_combo_hours[platform][prod][combo] += usage_amount
                    platform_product_vm_combo_instances[platform][prod][combo].add(rid)
                    platform_product_vm_combo_specs[platform][prod][combo] = {"vcpu": vcpu, "ramGiB": ram}
                elif provider != "aws" and "hour" in (usage_unit or "").lower():
                    specs = parse_azure_vm_specs(sku_name)
                    vcpu = specs["vcpu"] if specs else 0
                    ram = specs["ram_gib"] if specs else 0
                    combo = vm_combo_label(vcpu, ram) if specs else "Unknown size"
                    platform_vm_combo_hours[platform][combo] += usage_amount
                    platform_vm_combo_instances[platform][combo].add(rid)
                    platform_vm_combo_specs[platform][combo] = {"vcpu": vcpu, "ramGiB": ram}
                    platform_product_vm_combo_hours[platform][prod][combo] += usage_amount
                    platform_product_vm_combo_instances[platform][prod][combo].add(rid)
                    platform_product_vm_combo_specs[platform][prod][combo] = {"vcpu": vcpu, "ramGiB": ram}

            elif provider == "gcp" and "core" in (usage_type or "").lower() and "hour" in (usage_unit or "").lower():
                # GCP often splits vCPU and RAM billing. We track vCPU hours if we can infer it.
                vcpu = 1  # Default placeholder since we can't easily parse "N1 Predefined Instance Core" without more logic
                ram = 3.75 # Default ratio
                combo = "GCP Instance (Estimated)"
                platform_vm_combo_hours[platform][combo] += usage_amount
                platform_vm_combo_instances[platform][combo].add(rid)
                platform_vm_combo_specs[platform][combo] = {"vcpu": vcpu, "ramGiB": ram}
                platform_product_vm_combo_hours[platform][prod][combo] += usage_amount
                platform_product_vm_combo_instances[platform][prod][combo].add(rid)
                platform_product_vm_combo_specs[platform][prod][combo] = {"vcpu": vcpu, "ramGiB": ram}

            if svc == "Block Storage" and is_size_unit(usage_unit):
                platform_block_resources[platform][rid] += usage_amount
                platform_product_block_resources[platform][prod][rid] += usage_amount
            if svc == "Object Storage" and is_size_unit(usage_unit):
                platform_object_resources[platform][rid] += usage_amount
                platform_product_object_resources[platform][prod][rid] += usage_amount

            provider_cost[provider] += cost
            provider_rows[provider] += 1
            provider_res_name_cost[provider][res_key] += cost
            provider_res_name_count[provider][res_key] += 1

            # Platform-level storage aggregation (AWS only)
            if schema["schema"] == "aws":
                if raw_service == "AmazonEBS" and usage_unit == "GB-Mo":
                    platform_ebs_gbmo[platform] += usage_amount
                    platform_product_ebs_gbmo[platform][prod] += usage_amount
                    platform_ebs_resources[platform][rid] += usage_amount
                    platform_product_ebs_resources[platform][prod][rid] += usage_amount
                if raw_service == "AmazonS3" and usage_unit == "GB-Mo":
                    platform_s3_gbmo[platform] += usage_amount
                    platform_product_s3_gbmo[platform][prod] += usage_amount
                    platform_s3_resources[platform][rid] += usage_amount
                    platform_product_s3_resources[platform][prod][rid] += usage_amount

                # Platform-level EC2 SKU aggregation
                if raw_service == "AmazonEC2" and usage_type.startswith("BoxUsage:"):
                    sku = usage_type.split(":", 1)[1]
                    platform_ec2_hours[platform][sku] += usage_amount
                    platform_ec2_instances[platform][sku].add(rid)
                    if rid not in platform_vm_types[platform]:
                        platform_vm_types[platform][rid] = sku
                    platform_product_ec2_hours[platform][prod][sku] += usage_amount
                    platform_product_ec2_instances[platform][prod][sku].add(rid)
                    if rid not in platform_product_vm_types[platform][prod]:
                        platform_product_vm_types[platform][prod][rid] = sku

    source_metadata = audit.finish()
    resource_inventory, inventory_summary = inventory_rows(inventory_evidence, source_metadata["sha256"])
    return {
        "sourceMetadata": [source_metadata],
        "resourceInventory": resource_inventory,
        "resourceInventorySummary": inventory_summary,
        "prod_cost": prod_cost,
        "env_cost": env_cost,
        "svc_cost": svc_cost,
        "prod_svc_cost": prod_svc_cost,
        "prod_svc_res": prod_svc_res,
        "prod_rows": prod_rows,
        "prod_service_sets": prod_service_sets,
        "res_name_cost": res_name_cost,
        "res_name_count": res_name_count,
        "platform_cost": platform_cost,
        "platform_count": platform_count,
        "platform_res_name_cost": platform_res_name_cost,
        "platform_res_name_count": platform_res_name_count,
        "platform_ebs_gbmo": platform_ebs_gbmo,
        "platform_s3_gbmo": platform_s3_gbmo,
        "platform_ec2_hours": platform_ec2_hours,
        "platform_ec2_instances": platform_ec2_instances,
        "platform_vm_types": platform_vm_types,
        "instance_info": instance_info,
        "platform_rows": platform_rows,
        "platform_product_cost": platform_product_cost,
        "product_provider_cost": product_provider_cost,
        "platform_env_cost": platform_env_cost,
        "platform_service_cost": platform_service_cost,
        "platform_product_rows": platform_product_rows,
        "platform_env_rows": platform_env_rows,
        "platform_service_rows": platform_service_rows,
        "platform_product_service_cost": platform_product_service_cost,
        "platform_product_env_cost": platform_product_env_cost,
        "platform_product_res_name_cost": platform_product_res_name_cost,
        "platform_product_res_name_count": platform_product_res_name_count,
        "platform_product_ebs_gbmo": platform_product_ebs_gbmo,
        "platform_product_s3_gbmo": platform_product_s3_gbmo,
        "platform_product_ec2_hours": platform_product_ec2_hours,
        "platform_product_ec2_instances": platform_product_ec2_instances,
        "platform_product_vm_types": platform_product_vm_types,
        "platform_ebs_resources": platform_ebs_resources,
        "platform_s3_resources": platform_s3_resources,
        "platform_product_ebs_resources": platform_product_ebs_resources,
        "platform_product_s3_resources": platform_product_s3_resources,
        "platform_vm_combo_hours": platform_vm_combo_hours,
        "platform_vm_combo_instances": platform_vm_combo_instances,
        "platform_vm_combo_specs": platform_vm_combo_specs,
        "platform_product_vm_combo_hours": platform_product_vm_combo_hours,
        "platform_product_vm_combo_instances": platform_product_vm_combo_instances,
        "platform_product_vm_combo_specs": platform_product_vm_combo_specs,
        "platform_block_resources": platform_block_resources,
        "platform_object_resources": platform_object_resources,
        "platform_product_block_resources": platform_product_block_resources,
        "platform_product_object_resources": platform_product_object_resources,
        "provider_cost": provider_cost,
        "provider_rows": provider_rows,
        "provider_res_name_cost": provider_res_name_cost,
        "provider_res_name_count": provider_res_name_count,
    }


def top_items(items: Dict[str, float], limit: int = 10):
    return sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def merge_nested(dest, src):
    if isinstance(src, dict):
        if dest is None or not isinstance(dest, dict):
            dest = {}
        for key, value in src.items():
            dest[key] = merge_nested(dest.get(key), value)
        return dest
    if isinstance(src, set):
        if dest is None or not isinstance(dest, set):
            dest = set()
        dest |= src
        return dest
    if isinstance(src, (int, float)):
        return (dest or 0) + src
    return dest if dest not in (None, "") else src


def merge_nested_no_sum(dest, src):
    if isinstance(src, dict):
        if dest is None or not isinstance(dest, dict):
            dest = {}
        for key, value in src.items():
            dest[key] = merge_nested_no_sum(dest.get(key), value)
        return dest
    if isinstance(src, set):
        if dest is None or not isinstance(dest, set):
            dest = set()
        dest |= src
        return dest
    return dest if dest not in (None, "") else src


def merge_cur_states(states: List[dict]) -> dict:
    merged: Dict[str, object] = {}
    for state in states:
        for key, value in state.items():
            if key in {"sourceMetadata", "resourceInventory"}:
                merged.setdefault(key, []).extend(value)
                continue
            if key == "instance_info":
                if "instance_info" not in merged:
                    merged["instance_info"] = value
                continue
            if key in {"platform_vm_combo_specs", "platform_product_vm_combo_specs"}:
                merged[key] = merge_nested_no_sum(merged.get(key), value)
            else:
                merged[key] = merge_nested(merged.get(key), value)
    if "instance_info" not in merged and states:
        merged["instance_info"] = states[0].get("instance_info", {})
    return merged


def build_summary(cur: dict, limit_services_per_product: int = 6):
    prod_cost = cur["prod_cost"]
    env_cost = cur["env_cost"]
    svc_cost = cur["svc_cost"]
    prod_svc_cost = cur["prod_svc_cost"]
    prod_svc_res = cur["prod_svc_res"]
    prod_rows = cur["prod_rows"]
    prod_service_sets = cur["prod_service_sets"]
    res_name_cost = cur["res_name_cost"]
    res_name_count = cur["res_name_count"]
    platform_cost = cur["platform_cost"]
    platform_count = cur["platform_count"]
    platform_res_name_cost = cur["platform_res_name_cost"]
    platform_res_name_count = cur["platform_res_name_count"]
    platform_ebs_gbmo = cur["platform_ebs_gbmo"]
    platform_s3_gbmo = cur["platform_s3_gbmo"]
    platform_ec2_hours = cur["platform_ec2_hours"]
    platform_ec2_instances = cur["platform_ec2_instances"]
    platform_vm_types = cur["platform_vm_types"]
    instance_info = cur["instance_info"]
    platform_rows = cur["platform_rows"]
    platform_product_cost = cur["platform_product_cost"]
    platform_env_cost = cur["platform_env_cost"]
    platform_service_cost = cur["platform_service_cost"]
    platform_product_rows = cur["platform_product_rows"]
    platform_env_rows = cur["platform_env_rows"]
    platform_service_rows = cur["platform_service_rows"]
    platform_product_service_cost = cur["platform_product_service_cost"]
    platform_product_env_cost = cur["platform_product_env_cost"]
    platform_product_res_name_cost = cur["platform_product_res_name_cost"]
    platform_product_res_name_count = cur["platform_product_res_name_count"]
    platform_product_ebs_gbmo = cur["platform_product_ebs_gbmo"]
    platform_product_s3_gbmo = cur["platform_product_s3_gbmo"]
    platform_product_ec2_hours = cur["platform_product_ec2_hours"]
    platform_product_ec2_instances = cur["platform_product_ec2_instances"]
    platform_product_vm_types = cur["platform_product_vm_types"]
    platform_ebs_resources = cur["platform_ebs_resources"]
    platform_s3_resources = cur["platform_s3_resources"]
    platform_product_ebs_resources = cur["platform_product_ebs_resources"]
    platform_product_s3_resources = cur["platform_product_s3_resources"]
    platform_vm_combo_hours = cur["platform_vm_combo_hours"]
    platform_vm_combo_instances = cur["platform_vm_combo_instances"]
    platform_vm_combo_specs = cur["platform_vm_combo_specs"]
    platform_product_vm_combo_hours = cur["platform_product_vm_combo_hours"]
    platform_product_vm_combo_instances = cur["platform_product_vm_combo_instances"]
    platform_product_vm_combo_specs = cur["platform_product_vm_combo_specs"]
    platform_block_resources = cur["platform_block_resources"]
    platform_object_resources = cur["platform_object_resources"]
    platform_product_block_resources = cur["platform_product_block_resources"]
    platform_product_object_resources = cur["platform_product_object_resources"]
    provider_cost = cur["provider_cost"]
    provider_rows = cur["provider_rows"]
    provider_res_name_cost = cur["provider_res_name_cost"]
    provider_res_name_count = cur["provider_res_name_count"]

    total_cost = sum(prod_cost.values())

    def pct(val: float) -> float:
        return 0.0 if math.isclose(total_cost, 0.0) else (val / total_cost * 100.0)

    products = []
    for prod, cost in top_items(prod_cost, len(prod_cost)):
        prod_total = prod_cost[prod]
        services = []
        for svc, svc_cost_val in top_items(prod_svc_cost[prod], limit_services_per_product):
            share = 0.0 if math.isclose(prod_total, 0.0) else (svc_cost_val / prod_total * 100.0)
            services.append(
                {
                    "ServiceCategory": svc,
                    "BilledCost": round(svc_cost_val, 2),
                    "pctOfProduct": round(share, 2),
                    "resourceCount": len(prod_svc_res[prod][svc]),
                }
            )

        shares = [(c / prod_total) for c in prod_svc_cost[prod].values()] if prod_total else []
        hhi = sum(s * s for s in shares)

        products.append(
            {
                "Product": prod,
                "BilledCost": round(cost, 2),
                "pctOfTotal": round(pct(cost), 2),
                "rows": prod_rows[prod],
                "serviceCount": len(prod_service_sets[prod]),
                "services": services,
                "hhi": round(hhi, 3),
            }
        )

    services = [
        {"ServiceCategory": svc, "BilledCost": round(cost, 2), "pctOfTotal": round(pct(cost), 2)}
        for svc, cost in top_items(svc_cost, len(svc_cost))
    ]

    environments = [
        {"Environment": env, "BilledCost": round(cost, 2), "pctOfTotal": round(pct(cost), 2)}
        for env, cost in top_items(env_cost, len(env_cost))
    ]

    resource_names = [
        {
            "ResourceName": name,
            "BilledCost": round(cost, 2),
            "pctOfTotal": round(pct(cost), 2),
            "rows": res_name_count[name],
        }
        for name, cost in top_items(res_name_cost, len(res_name_cost))
    ]

    providers = []
    for provider, cost_val in top_items(provider_cost, len(provider_cost)):
        provider_resnames = []
        for rname, rcost in sorted(provider_res_name_cost.get(provider, {}).items(), key=lambda kv: kv[1], reverse=True):
            provider_resnames.append(
                {
                    "ResourceName": rname,
                    "BilledCost": round(rcost, 2),
                    "pctOfProvider": round((rcost / cost_val * 100) if cost_val else 0.0, 2),
                    "rows": provider_res_name_count[provider][rname],
                }
            )
        providers.append(
            {
                "ProviderName": provider,
                "BilledCost": round(cost_val, 2),
                "pctOfTotal": round(pct(cost_val), 2),
                "rows": provider_rows.get(provider, 0),
                "resourceNames": provider_resnames,
            }
        )

    platforms = []
    for name, cost_val in top_items(platform_cost, len(platform_cost)):
        pct_val = pct(cost_val)

        vcpu_total = 0
        ram_total = 0.0
        nic_count = len(platform_vm_types.get(name, {}))
        for rid, sku in platform_vm_types.get(name, {}).items():
            info = instance_info.get(sku, {"vcpu": 0, "ram_gib": 0})
            vcpu_total += info["vcpu"]
            ram_total += info["ram_gib"]

        ebs_gbmo = platform_ebs_gbmo.get(name, 0.0)
        s3_gbmo = platform_s3_gbmo.get(name, 0.0)

        ec2_skus = []
        for sku, hours in platform_ec2_hours.get(name, {}).items():
            info = instance_info.get(sku, {"vcpu": 0, "ram_gib": 0})
            ec2_skus.append(
                {
                    "sku": sku,
                    "hours": round(hours, 2),
                    "instances": len(platform_ec2_instances.get(name, {}).get(sku, set())),
                    "vcpu": info["vcpu"],
                    "ramGiB": info["ram_gib"],
                }
            )
        ec2_skus = sorted(ec2_skus, key=lambda x: x["hours"], reverse=True)

        vm_skus = []
        for combo, hours in platform_vm_combo_hours.get(name, {}).items():
            spec = platform_vm_combo_specs.get(name, {}).get(combo, {})
            resource_ids = sorted(platform_vm_combo_instances.get(name, {}).get(combo, set()))
            vm_skus.append(
                {
                    "sku": combo,
                    "hours": round(hours, 2),
                    "instances": len(platform_vm_combo_instances.get(name, {}).get(combo, set())),
                    "vcpu": spec.get("vcpu", 0),
                    "ramGiB": spec.get("ramGiB", 0),
                    "resourceIds": [{"id": rid} for rid in resource_ids],
                }
            )
        vm_skus = sorted(vm_skus, key=lambda x: x["hours"], reverse=True)

        resnames = []
        for rname, rcost in sorted(platform_res_name_cost.get(name, {}).items(), key=lambda kv: kv[1], reverse=True):
            resnames.append(
                {
                    "ResourceName": rname,
                    "BilledCost": round(rcost, 2),
                    "pctOfTotal": round(pct(rcost), 2),
                    "rows": platform_res_name_count[name][rname],
                }
            )

        platforms.append(
            {
                "Platform": name,
                "BilledCost": round(cost_val, 2),
                "pctOfTotal": round(pct_val, 2),
                "rows": platform_rows.get(name, 0),
                "vcpuTotal": vcpu_total,
                "ramGiBTotal": round(ram_total, 2),
                "ebsGbMo": round(ebs_gbmo, 2),
                "s3GbMo": round(s3_gbmo, 2),
                "nicCount": nic_count,
                "ec2Skus": ec2_skus,
                "vmSkus": vm_skus,
                "resourceNames": resnames,
                "ebsBuckets": bucketize_resources(platform_ebs_resources.get(name, {}), True),
                "s3Buckets": bucketize_resources(platform_s3_resources.get(name, {}), True),
                "blockBuckets": bucketize_resources(platform_block_resources.get(name, {}), True),
                "objectBuckets": bucketize_resources(platform_object_resources.get(name, {}), True),
                "products": [],
                "services": [
                    {
                        "ServiceCategory": svc,
                        "BilledCost": round(val, 2),
                        "pctOfPlatform": round((val / cost_val * 100) if cost_val else 0.0, 2),
                        "rows": platform_service_rows[name].get(svc, 0),
                    }
                    for svc, val in sorted(platform_service_cost.get(name, {}).items(), key=lambda kv: kv[1], reverse=True)
                ],
                "environments": [
                    {
                        "Environment": env,
                        "BilledCost": round(val, 2),
                        "pctOfPlatform": round((val / cost_val * 100) if cost_val else 0.0, 2),
                        "rows": platform_env_rows[name].get(env, 0),
                    }
                    for env, val in sorted(platform_env_cost.get(name, {}).items(), key=lambda kv: kv[1], reverse=True)
                ],
            }
        )

        for prod, pcost in sorted(platform_product_cost.get(name, {}).items(), key=lambda kv: kv[1], reverse=True):
            vcpu_p = 0
            ram_p = 0.0
            nic_p = len(platform_product_vm_types.get(name, {}).get(prod, {}))
            for rid, sku in platform_product_vm_types.get(name, {}).get(prod, {}).items():
                info = instance_info.get(sku, {"vcpu": 0, "ram_gib": 0})
                vcpu_p += info["vcpu"]
                ram_p += info["ram_gib"]

            ec2_prod_skus = []
            for sku, hours in platform_product_ec2_hours.get(name, {}).get(prod, {}).items():
                info = instance_info.get(sku, {"vcpu": 0, "ram_gib": 0})
                ec2_prod_skus.append(
                    {
                        "sku": sku,
                        "hours": round(hours, 2),
                    "instances": len(platform_product_ec2_instances.get(name, {}).get(prod, {}).get(sku, set())),
                        "vcpu": info["vcpu"],
                        "ramGiB": info["ram_gib"],
                    }
                )
            ec2_prod_skus = sorted(ec2_prod_skus, key=lambda x: x["hours"], reverse=True)

            vm_prod_skus = []
            for combo, hours in platform_product_vm_combo_hours.get(name, {}).get(prod, {}).items():
                spec = platform_product_vm_combo_specs.get(name, {}).get(prod, {}).get(combo, {})
                resource_ids = sorted(platform_product_vm_combo_instances.get(name, {}).get(prod, {}).get(combo, set()))
                vm_prod_skus.append(
                    {
                        "sku": combo,
                        "hours": round(hours, 2),
                        "instances": len(
                            platform_product_vm_combo_instances.get(name, {}).get(prod, {}).get(combo, set())
                        ),
                        "vcpu": spec.get("vcpu", 0),
                        "ramGiB": spec.get("ramGiB", 0),
                        "resourceIds": [{"id": rid} for rid in resource_ids],
                    }
                )
            vm_prod_skus = sorted(vm_prod_skus, key=lambda x: x["hours"], reverse=True)

            prod_resnames = []
            for rname, rcost in sorted(
                platform_product_res_name_cost.get(name, {}).get(prod, {}).items(), key=lambda kv: kv[1], reverse=True
            ):
                prod_resnames.append(
                    {
                        "ResourceName": rname,
                        "BilledCost": round(rcost, 2),
                        "pctOfPlatform": round((rcost / cost_val * 100) if cost_val else 0.0, 2),
                        "rows": platform_product_res_name_count[name][prod][rname],
                    }
                )

            prod_services = []
            for svc, sval in sorted(
                platform_product_service_cost.get(name, {}).get(prod, {}).items(), key=lambda kv: kv[1], reverse=True
            ):
                prod_services.append(
                    {
                        "ServiceCategory": svc,
                        "BilledCost": round(sval, 2),
                        "pctOfPlatform": round((sval / cost_val * 100) if cost_val else 0.0, 2),
                        "rows": platform_service_rows[name].get(svc, 0),
                    }
                )

            prod_envs = []
            for env, eval_ in sorted(
                platform_product_env_cost.get(name, {}).get(prod, {}).items(), key=lambda kv: kv[1], reverse=True
            ):
                prod_envs.append(
                    {
                        "Environment": env,
                        "BilledCost": round(eval_, 2),
                        "pctOfPlatform": round((eval_ / cost_val * 100) if cost_val else 0.0, 2),
                        "rows": platform_env_rows[name].get(env, 0),
                    }
                )

            platforms[-1]["products"].append(
                {
                    "Product": prod,
                    "providers": [
                        {"ProviderName": provider, "BilledCost": round(amount, 2)}
                        for provider, amount in sorted(cur["product_provider_cost"][name][prod].items())
                    ],
                    "BilledCost": round(pcost, 2),
                    "pctOfPlatform": round((pcost / cost_val * 100) if cost_val else 0.0, 2),
                    "rows": platform_product_rows[name].get(prod, 0),
                    "vcpuTotal": vcpu_p,
                    "ramGiBTotal": round(ram_p, 2),
                    "ebsGbMo": round(platform_product_ebs_gbmo.get(name, {}).get(prod, 0.0), 2),
                    "s3GbMo": round(platform_product_s3_gbmo.get(name, {}).get(prod, 0.0), 2),
                    "nicCount": nic_p,
                    "ec2Skus": ec2_prod_skus,
                    "vmSkus": vm_prod_skus,
                    "resourceNames": prod_resnames,
                    "ebsBuckets": bucketize_resources(
                        platform_product_ebs_resources.get(name, {}).get(prod, {}),
                        True,
                    ),
                    "s3Buckets": bucketize_resources(
                        platform_product_s3_resources.get(name, {}).get(prod, {}),
                        True,
                    ),
                    "blockBuckets": bucketize_resources(
                        platform_product_block_resources.get(name, {}).get(prod, {}),
                        True,
                    ),
                    "objectBuckets": bucketize_resources(
                        platform_product_object_resources.get(name, {}).get(prod, {}),
                        True,
                    ),
                    "services": prod_services,
                    "environments": prod_envs,
                }
            )

    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for prod, svc_set in prod_service_sets.items():
        for a, b in itertools.combinations(sorted(svc_set), 2):
            pair_counts[(a, b)] += 1

    co_occurrence = [
        {"pair": [a, b], "products": count}
        for (a, b), count in sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]

    stack_signature: Dict[frozenset, List[str]] = defaultdict(list)
    for prod, svc_set in prod_service_sets.items():
        stack_signature[frozenset(svc_set)].append(prod)

    shared_stacks = []
    for stack, prods in stack_signature.items():
        if len(prods) > 1:
            shared_stacks.append({"products": sorted(prods), "services": sorted(stack)})

    return {
        "analysis": build_analysis(cur["sourceMetadata"], round(total_cost, 2)),
        "resourceInventory": cur.get("resourceInventory", []),
        "resourceInventorySummary": cur.get("resourceInventorySummary", {"unidentifiedCost": 0, "unidentifiedRows": 0}),
        "allocation": {
            "unallocatedProductCost": round(sum(cost for name, cost in prod_cost.items() if not clean_dimension(name)), 2),
            "unallocatedPlatformCost": round(sum(cost for name, cost in platform_cost.items() if not clean_dimension(name)), 2),
            "unallocatedEnvironmentCost": round(sum(cost for name, cost in env_cost.items() if not clean_dimension(name)), 2),
        },
        "TotalBilledCost": round(total_cost, 2),
        "products": products,
        "services": services,
        "environments": environments,
        "coOccurrence": co_occurrence,
        "sharedStacks": shared_stacks,
        "resourceNames": resource_names,
        "platforms": platforms,
        "providers": providers,
    }


def build_naming_metadata(naming_config: dict, naming_path: Optional[pathlib.Path], naming_mode: str) -> dict:
    normalized = normalize_convention_config(naming_config)
    base = normalized["default"]
    source = normalized.get("source") or ("custom" if naming_path else DEFAULT_NAMING_CONVENTION.get("source", ""))
    meta = {
        "source": source,
        "description": normalized.get("description", ""),
        "delimiter": base.get("delimiter", "-"),
        "pattern": base.get("pattern", []),
        "mode": naming_mode,
    }
    providers = {}
    for provider, override in normalized.get("providers", {}).items():
        provider_cfg = merge_convention(base, override)
        providers[provider] = {
            "delimiter": provider_cfg.get("delimiter", "-"),
            "pattern": provider_cfg.get("pattern", []),
        }
    if providers:
        meta["providers"] = providers
    return meta


def main():
    parser = argparse.ArgumentParser(description="Summarize CUR slice(s) into JSON for the report UI.")
    parser.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        action="append",
        help="Path to CUR CSV file (repeatable)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=pathlib.Path("report_data.json"),
        help="Path to write JSON summary",
    )
    parser.add_argument(
        "--naming",
        "-n",
        type=pathlib.Path,
        default=None,
        help="Path to naming convention JSON (optional)",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Optional report label for each input (repeatable)",
    )
    parser.add_argument(
        "--currency",
        help="Explicit currency assumption for rows without a currency (e.g. USD). Does not convert or overwrite source currencies.",
    )
    parser.add_argument(
        "--naming-mode",
        choices=["fallback", "override"],
        default="fallback",
        help="When to use naming convention values vs tags (fallback or override)",
    )
    args = parser.parse_args()

    naming_config = load_naming_convention(args.naming)
    input_paths = args.input or [pathlib.Path("mock_aws_cur_iaas_heavy_named20251122.csv")]
    labels = list(args.label or [])
    if len(labels) < len(input_paths):
        labels += [p.stem for p in input_paths[len(labels) :]]
    labels = labels[: len(input_paths)]

    states = []
    reports = []
    seen_sources = set()
    for idx, path in enumerate(input_paths):
        try:
            cur = load_cur(path, naming_config, args.naming_mode, args.currency)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        fingerprint = cur["sourceMetadata"][0]["sha256"]
        if fingerprint in seen_sources:
            parser.error(f"Duplicate billing input: {path.name}. The same file contents cannot be counted twice.")
        seen_sources.add(fingerprint)
        states.append(cur)
        try:
            source_summary = build_summary(cur)
        except ValueError as error:
            parser.error(str(error))
        reports.append(
            {
                "id": f"report-{idx + 1}",
                "label": labels[idx],
                "source": path.name,
                "summary": source_summary,
            }
        )

    aggregate_state = merge_cur_states(states) if len(states) > 1 else states[0]
    try:
        summary = build_summary(aggregate_state)
    except ValueError as error:
        parser.error(str(error))
    summary["namingConvention"] = build_naming_metadata(naming_config, args.naming, args.naming_mode)
    summary["reports"] = reports
    summary["reportCount"] = len(reports)
    summary["sampleData"] = False
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    currency_label = summary["analysis"]["currency"] or "currency unspecified"
    print(f"Wrote {args.output} with {summary['TotalBilledCost']:,.2f} ({currency_label}; {summary['analysis']['costBasis']} cost)")
    for warning in summary["analysis"]["warnings"]:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
