import csv
import itertools
import json
import math
import pathlib
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_cur(path: pathlib.Path):
    prod_cost: Dict[str, float] = defaultdict(float)
    env_cost: Dict[str, float] = defaultdict(float)
    svc_cost: Dict[str, float] = defaultdict(float)
    prod_svc_cost: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    prod_svc_res: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    prod_rows: Dict[str, int] = defaultdict(int)
    prod_service_sets: Dict[str, Set[str]] = defaultdict(set)

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

    return {
        "prod_cost": prod_cost,
        "env_cost": env_cost,
        "svc_cost": svc_cost,
        "prod_svc_cost": prod_svc_cost,
        "prod_svc_res": prod_svc_res,
        "prod_rows": prod_rows,
        "prod_service_sets": prod_service_sets,
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
    }


def main():
    input_path = pathlib.Path("MyProduct/mock_aws_cur_iaas_heavy.csv")
    output_path = pathlib.Path("MyProduct/report_data.json")

    cur = load_cur(input_path)
    summary = build_summary(cur)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {output_path} with totalCost=${summary['totalCost']:,.2f}")


if __name__ == "__main__":
    main()
