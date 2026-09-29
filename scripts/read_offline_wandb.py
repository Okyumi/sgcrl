import sys, json, collections
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb

path = sys.argv[1]
out = sys.argv[2]
ds = datastore.DataStore()
ds.open_for_scan(path)
rows = []
kinds = collections.Counter()
while True:
    data = ds.scan_data()
    if data is None:
        break
    rec = pb.Record()
    rec.ParseFromString(data)
    k = rec.WhichOneof("record_type")
    kinds[k] += 1
    if k == "history":
        row = {}
        for it in rec.history.item:
            try:
                row[it.key] = json.loads(it.value_json)
            except Exception:
                row[it.key] = it.value_json
        rows.append(row)
print("record kinds:", dict(kinds))
print("history rows:", len(rows))
keys = collections.Counter(k for r in rows for k in r)
print("num keys:", len(keys))
for k, c in sorted(keys.items()):
    print(f"  {k}: {c}")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", out)
