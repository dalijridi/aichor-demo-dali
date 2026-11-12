import json, os, tempfile, time
from datetime import date, timedelta
import pandas as pd
import delta_sharing

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

TARGET = "speaking_water.monitor_share.v_speaking_water_telemetry"
GATEWAY = "25dbf102-4b9d-4e82-99ec-1dfdfcbc072a"
LAST_DAYS = int(os.environ.get("LAST_DAYS", "2"))          # ~63k rows/day → ~125k rows
ROW_CAP = int(os.environ.get("ROW_CAP", "300000"))          # hard cap on any read

t0 = time.time()
def step(msg):
    print(f"\n[{time.time() - t0:6.1f}s] {msg}", flush=True)

profile = {
    "shareCredentialsVersion": 1,
    "endpoint": os.environ["DELTA_SHARING_ENDPOINT"],
    "bearerToken": os.environ["DELTA_SHARING_BEARER_TOKEN"],
}
path = os.path.join(tempfile.mkdtemp(), "config.share")
with open(path, "w") as f:
    json.dump(profile, f)
url = f"{path}#{TARGET}"

step("metadata")
meta = delta_sharing.get_table_metadata(url)
fields = {f["name"]: f["type"] for f in json.loads(meta.schema_string)["fields"]}
print("columns:", fields)
print("partition columns:", meta.partition_columns)
version = delta_sharing.get_table_version(url)
print("current version:", version)

step("version pinning (10 rows)")
try:
    pinned = delta_sharing.load_as_pandas(url, version=version, limit=10)
    print("OK:", len(pinned), "rows at version", version)
except Exception as e:
    print("NOT supported:", type(e).__name__, str(e)[:300])

# Recent slice: filter on a date partition column if there is one
date_parts = [c for c in meta.partition_columns if fields.get(c) == "date"]
hints = None
if date_parts:
    since = (date.today() - timedelta(days=LAST_DAYS)).isoformat()
    hints = json.dumps({"op": "greaterThanOrEqual", "children": [
        {"op": "column", "name": date_parts[0], "valueType": "date"},
        {"op": "literal", "value": since, "valueType": "date"}]})
    step(f"reading {date_parts[0]} >= {since} (cap {ROW_CAP} rows)")
else:
    step(f"no date partition column → capped read of {ROW_CAP} rows (not necessarily the latest)")

df = delta_sharing.load_as_pandas(url, limit=ROW_CAP, jsonPredicateHints=hints)
print("rows read:", len(df))

step("checks")
time_cols = [c for c, t in fields.items() if t in ("timestamp", "timestamp_ntz", "date")]
if time_cols:
    tc = time_cols[0]
    df = df.sort_values(tc)
    days = pd.to_datetime(df[tc]).dt.date.value_counts().sort_index()
    print(f"{tc}: {df[tc].min()} → {df[tc].max()}")
    print("rows per day:\n", days.to_string())
print("rows outside the gateway:", (~df["device_name"].str.contains(GATEWAY)).sum())
print("device_names:\n", df["device_name"].value_counts().to_string())
name_cols = [c for c, t in fields.items() if "metric" in c and t == "string" and not c.endswith("_v")]
if name_cols:
    print(df.groupby(["device_name", name_cols[0]]).size().sort_values(ascending=False).head(60).to_string())
print("null share per column:\n", df.isna().mean().round(3).to_string())
print("\nlast rows:\n", df.tail(10).to_string())

step("done")
