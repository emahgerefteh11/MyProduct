import argparse
import csv
import itertools
import json
import math
import pathlib
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def classify_resource_name(resource_id: str) -> str:
    """Heuristic to bucket resources by naming convention/prefix."""
    if not resource_id:
        return "unknown"
    rid = resource_id.strip()
    if rid.startswith("arn:"):
        # arn:partition:service:region:account:resourcetype/resource
        tail = rid.split(":", 5)[-1]
        tail = tail.split("/")[-1]
        return tail.split("-")[0] if "-" in tail else tail.split(":")[0]
    # common AWS ids like i-xxxx, vol-xxxx, db-xxxx, etc.
    token = rid.split("-")[0]
    return token or "unknown"


def extract_platform(resource_id: str) -> str:
    """
    Heuristic to extract a platform marker from a resource name like
    'disk-PlatformB-dev-0001'. Returns 'unknown' if not present.
    """
    if not resource_id:
        return "unknown"
    parts = resource_id.split("-")
    if len(parts) >= 2:
        return parts[1] or "unknown"
    return "unknown"


def load_cur(path: pathlib.Path):
    prod_cost: Dict[str, float] = defaultdict(float)
    env_cost: Dict[str, float] = defaultdict(float)
    svc_cost: Dict[str, float] = defaultdict(float)
    prod_svc_cost: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    prod_svc_res: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    prod_rows: Dict[str, int] = defaultdict(int)
    prod_service_sets: Dict[str, Set[str]] = defaultdict(set)

    # resource naming breakdown
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
    platform_vm_types: Dict[str, Dict[str, str]] = defaultdict(dict)  # platform -> resourceId -> instance_type

    platform_rows: Dict[str, int] = defaultdict(int)
    platform_product_cost: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
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

    instance_info = {
        "c5.xlarge": {"vcpu": 4, "ram_gib": 8},
        "m5.large": {"vcpu": 2, "ram_gib": 8},
        "t3.medium": {"vcpu": 2, "ram_gib": 4},
    }

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cost = float(row["lineItem/UnblendedCost"] or 0)
            prod = row.get("tags/Product") or "UnTagged"
            env = row.get("tags/Environment") or "UnTagged"
            svc = row["lineItem/ProductCode"]
            rid = row.get("resourceId") or ""

            prod_cost[prod] += cost
            env_cost[env] += cost
            svc_cost[svc] += cost
            prod_svc_cost[prod][svc] += cost
            prod_svc_res[prod][svc].add(rid)
            prod_rows[prod] += 1
            prod_service_sets[prod].add(svc)

            res_key = classify_resource_name(rid)
            res_name_cost[res_key] += cost
            res_name_count[res_key] += 1

            platform = extract_platform(rid)
            platform_cost[platform] += cost
            platform_count[platform] += 1
            platform_res_name_cost[platform][res_key] += cost
            platform_res_name_count[platform][res_key] += 1
            platform_rows[platform] += 1
            platform_product_cost[platform][prod] += cost
            platform_env_cost[platform][env] += cost
            platform_service_cost[platform][svc] += cost
            platform_product_rows[platform][prod] += 1
            platform_env_rows[platform][env] += 1
            platform_service_rows[platform][svc] += 1
            platform_product_service_cost[platform][prod][svc] += cost
            platform_product_env_cost[platform][prod][env] += cost
            platform_product_res_name_cost[platform][prod][res_key] += cost
            platform_product_res_name_count[platform][prod][res_key] += 1

            # Platform-level storage aggregation
            if row["lineItem/ProductCode"] == "AmazonEBS" and row["lineItem/UsageUnit"] == "GB-Mo":
                amt = float(row["lineItem/UsageAmount"] or 0)
                platform_ebs_gbmo[platform] += amt
                platform_product_ebs_gbmo[platform][prod] += amt
            if row["lineItem/ProductCode"] == "AmazonS3" and row["lineItem/UsageUnit"] == "GB-Mo":
                amt = float(row["lineItem/UsageAmount"] or 0)
                platform_s3_gbmo[platform] += amt
                platform_product_s3_gbmo[platform][prod] += amt

            # Platform-level EC2 SKU aggregation
            if row["lineItem/ProductCode"] == "AmazonEC2":
                ut = row["lineItem/UsageType"]
                if ut.startswith("BoxUsage:"):
                    sku = ut.split(":", 1)[1]
                    hours = float(row["lineItem/UsageAmount"] or 0)
                    platform_ec2_hours[platform][sku] += hours
                    platform_ec2_instances[platform][sku].add(rid)
                    if rid not in platform_vm_types[platform]:
                        platform_vm_types[platform][rid] = sku
                    platform_product_ec2_hours[platform][prod][sku] += hours
                    platform_product_ec2_instances[platform][prod][sku].add(rid)
                    if rid not in platform_product_vm_types[platform][prod]:
                        platform_product_vm_types[platform][prod][rid] = sku

    return {
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
    }


def top_items(items: Dict[str, float], limit: int = 10):
    return sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:limit]


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
                    "service": svc,
                    "cost": round(svc_cost_val, 2),
                    "pctOfProduct": round(share, 2),
                    "resourceCount": len(prod_svc_res[prod][svc]),
                }
            )

        shares = [(c / prod_total) for c in prod_svc_cost[prod].values()] if prod_total else []
        hhi = sum(s * s for s in shares)

        products.append(
            {
                "product": prod,
                "cost": round(cost, 2),
                "pctOfTotal": round(pct(cost), 2),
                "rows": prod_rows[prod],
                "serviceCount": len(prod_service_sets[prod]),
                "services": services,
                "hhi": round(hhi, 3),
            }
        )

    services = [
        {"service": svc, "cost": round(cost, 2), "pctOfTotal": round(pct(cost), 2)}
        for svc, cost in top_items(svc_cost, len(svc_cost))
    ]

    environments = [
        {"environment": env, "cost": round(cost, 2), "pctOfTotal": round(pct(cost), 2)}
        for env, cost in top_items(env_cost, len(env_cost))
    ]

    resource_names = [
        {
            "name": name,
            "cost": round(cost, 2),
            "pctOfTotal": round(pct(cost), 2),
            "rows": res_name_count[name],
        }
        for name, cost in top_items(res_name_cost, len(res_name_cost))
    ]

    platforms = []
    for name, cost_val in top_items(platform_cost, len(platform_cost)):
        pct_val = pct(cost_val)

        # vCPU / RAM totals based on unique VMs
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
                    "instances": len(platform_ec2_instances[name].get(sku, set())),
                    "vcpu": info["vcpu"],
                    "ramGiB": info["ram_gib"],
                }
            )
        ec2_skus = sorted(ec2_skus, key=lambda x: x["hours"], reverse=True)

        resnames = []
        for rname, rcost in sorted(platform_res_name_cost.get(name, {}).items(), key=lambda kv: kv[1], reverse=True):
            resnames.append(
                {
                    "name": rname,
                    "cost": round(rcost, 2),
                    "pctOfTotal": round(pct(rcost), 2),
                    "rows": platform_res_name_count[name][rname],
                }
            )

        platforms.append(
            {
                "platform": name,
                "cost": round(cost_val, 2),
                "pctOfTotal": round(pct_val, 2),
                "rows": platform_rows.get(name, 0),
                "vcpuTotal": vcpu_total,
                "ramGiBTotal": round(ram_total, 2),
                "ebsGbMo": round(ebs_gbmo, 2),
                "s3GbMo": round(s3_gbmo, 2),
                "nicCount": nic_count,
                "ec2Skus": ec2_skus,
                "resourceNames": resnames,
                "products": [],
                "services": [
                    {
                        "service": svc,
                        "cost": round(val, 2),
                        "pctOfPlatform": round((val / cost_val * 100) if cost_val else 0.0, 2),
                        "rows": platform_service_rows[name].get(svc, 0),
                    }
                    for svc, val in sorted(platform_service_cost.get(name, {}).items(), key=lambda kv: kv[1], reverse=True)
                ],
                "environments": [
                    {
                        "environment": env,
                        "cost": round(val, 2),
                        "pctOfPlatform": round((val / cost_val * 100) if cost_val else 0.0, 2),
                        "rows": platform_env_rows[name].get(env, 0),
                    }
                    for env, val in sorted(platform_env_cost.get(name, {}).items(), key=lambda kv: kv[1], reverse=True)
                ],
            }
        )

        # populate per-platform products with nested details
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
                        "instances": len(platform_product_ec2_instances[name][prod].get(sku, set())),
                        "vcpu": info["vcpu"],
                        "ramGiB": info["ram_gib"],
                    }
                )
            ec2_prod_skus = sorted(ec2_prod_skus, key=lambda x: x["hours"], reverse=True)

            prod_resnames = []
            for rname, rcost in sorted(
                platform_product_res_name_cost.get(name, {}).get(prod, {}).items(), key=lambda kv: kv[1], reverse=True
            ):
                prod_resnames.append(
                    {
                        "name": rname,
                        "cost": round(rcost, 2),
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
                        "service": svc,
                        "cost": round(sval, 2),
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
                        "environment": env,
                        "cost": round(eval_, 2),
                        "pctOfPlatform": round((eval_ / cost_val * 100) if cost_val else 0.0, 2),
                        "rows": platform_env_rows[name].get(env, 0),
                    }
                )

            platforms[-1]["products"].append(
                {
                    "product": prod,
                    "cost": round(pcost, 2),
                    "pctOfPlatform": round((pcost / cost_val * 100) if cost_val else 0.0, 2),
                    "rows": platform_product_rows[name].get(prod, 0),
                    "vcpuTotal": vcpu_p,
                    "ramGiBTotal": round(ram_p, 2),
                    "ebsGbMo": round(platform_product_ebs_gbmo.get(name, {}).get(prod, 0.0), 2),
                    "s3GbMo": round(platform_product_s3_gbmo.get(name, {}).get(prod, 0.0), 2),
                    "nicCount": nic_p,
                    "ec2Skus": ec2_prod_skus,
                    "resourceNames": prod_resnames,
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
            shared_stacks.append(
                {"products": sorted(prods), "services": sorted(stack)}
            )

    return {
        "totalCost": round(total_cost, 2),
        "products": products,
        "services": services,
        "environments": environments,
        "coOccurrence": co_occurrence,
        "sharedStacks": shared_stacks,
        "resourceNames": resource_names,
        "platforms": platforms,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize AWS CUR slice into JSON for the report UI.")
    parser.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        default=pathlib.Path("MyProduct/mock_aws_cur_iaas_heavy.csv"),
        help="Path to CUR CSV file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=pathlib.Path("MyProduct/report_data.json"),
        help="Path to write JSON summary",
    )
    args = parser.parse_args()

    cur = load_cur(args.input)
    summary = build_summary(cur)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output} with totalCost=${summary['totalCost']:,.2f}")


if __name__ == "__main__":
    main()
