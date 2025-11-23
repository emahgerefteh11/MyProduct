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

    platforms = [
        {
            "platform": name,
            "cost": round(cost, 2),
            "pctOfTotal": round(pct(cost), 2),
            "rows": platform_count[name],
        }
        for name, cost in top_items(platform_cost, len(platform_cost))
    ]

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
