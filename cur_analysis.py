import csv, collections, itertools

path = 'MyProduct/mock_aws_cur_iaas_heavy.csv'

prod_cost = collections.defaultdict(float)
env_cost = collections.defaultdict(float)
svc_cost = collections.defaultdict(float)
prod_svc_cost = collections.defaultdict(lambda: collections.defaultdict(float))
prod_svc_res = collections.defaultdict(lambda: collections.defaultdict(set))
prod_rows = collections.defaultdict(int)
prod_service_sets = collections.defaultdict(set)

with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cost = float(row['lineItem/UnblendedCost'] or 0)
        prod = row.get('tags/Product') or 'UnTagged'
        env = row.get('tags/Environment') or 'UnTagged'
        svc = row['lineItem/ProductCode']
        rid = row.get('resourceId') or ''
        prod_cost[prod] += cost
        env_cost[env] += cost
        svc_cost[svc] += cost
        prod_svc_cost[prod][svc] += cost
        prod_svc_res[prod][svc].add(rid)
        prod_rows[prod] += 1
        prod_service_sets[prod].add(svc)

def top_items(d, n=5):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

total_cost = sum(prod_cost.values())
print(f"Total unblended cost: ${total_cost:,.2f}\n")

print('Top products by cost:')
for prod, cost in top_items(prod_cost, 10):
    pct = cost / total_cost * 100 if total_cost else 0
    print(f"  {prod:20s}  ${cost:12,.2f}  ({pct:5.2f}%)  rows={prod_rows[prod]}  services={len(prod_service_sets[prod])}")

print('\nTop services by cost:')
for svc, cost in top_items(svc_cost, 10):
    pct = cost / total_cost * 100 if total_cost else 0
    print(f"  {svc:20s}  ${cost:12,.2f}  ({pct:5.2f}%)")

print('\nCost by environment:')
for env, cost in top_items(env_cost, 10):
    pct = cost / total_cost * 100 if total_cost else 0
    print(f"  {env:10s}  ${cost:12,.2f}  ({pct:5.2f}%)")

print('\nStacks per product (top 5 products, top 5 services each):')
for prod, _ in top_items(prod_cost, 5):
    print(f"\n{prod}:")
    svc_items = top_items(prod_svc_cost[prod], 5)
    prod_total = prod_cost[prod]
    for svc, cost in svc_items:
        pct = cost / prod_total * 100 if prod_total else 0
        res_count = len(prod_svc_res[prod][svc])
        print(f"  {svc:22s} ${cost:10,.2f} ({pct:5.2f}%) resources={res_count}")

pair_counts = collections.defaultdict(int)
for prod, svc_set in prod_service_sets.items():
    for a, b in itertools.combinations(sorted(svc_set), 2):
        pair_counts[(a,b)] += 1

print('\nService co-occurrence across products (top 15 pairs):')
for (a,b), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {a} + {b}: in {count} products")

stack_signature = collections.defaultdict(list)
for prod, svc_set in prod_service_sets.items():
    stack_signature[frozenset(svc_set)].append(prod)

print('\nProducts sharing identical service stacks:')
any_shared = False
for stack, prods in stack_signature.items():
    if len(prods) > 1:
        any_shared = True
        services = ', '.join(sorted(stack))
        print(f"  {', '.join(sorted(prods))}  -> services: {services}")
if not any_shared:
    print('  (none)')

print('\nCost concentration per product (HHI):')
for prod, svcs in prod_svc_cost.items():
    total = prod_cost[prod]
    shares = [(c/total) for c in svcs.values()] if total else []
    hhi = sum(s*s for s in shares)
    print(f"  {prod:15s} HHI={hhi:0.3f} (1.0 single-service dominated)")
