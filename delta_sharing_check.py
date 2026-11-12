import json, os, tempfile, time
from datetime import datetime, timedelta, timezone
import pandas as pd
import delta_sharing

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

TARGET = "speaking_water.monitor_share.v_speaking_water_telemetry"
GATEWAY = "25dbf102-4b9d-4e82-99ec-1dfdfcbc072a"
LAST_DAYS = int(os.environ.get("LAST_DAYS", "2"))
ROW_CAP = int(os.environ.get("ROW_CAP", "300000"))

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

since = datetime.now(timezone.utc) - timedelta(days=LAST_DAYS)
hints = json.dumps({"op": "greaterThanOrEqual", "children": [
    {"op": "column", "name": "timestamp", "valueType": "timestamp"},
    {"op": "literal", "value": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "valueType": "timestamp"}]})

step(f"reading timestamp >= {since:%Y-%m-%d %H:%M} UTC (hint, cap {ROW_CAP} rows)")
df = delta_sharing.load_as_pandas(url, limit=ROW_CAP, jsonPredicateHints=hints)
print("rows read:", len(df))
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
recent = df[df["timestamp"] >= since].sort_values("timestamp")
print("rows in window:", len(recent), "→ hint honoured" if len(recent) == len(df) else "→ hint ignored or partial")
print("timestamp range read:", df["timestamp"].min(), "→", df["timestamp"].max())

step("checks")
print("rows outside the gateway:", (~df["device_name"].str.contains(GATEWAY)).sum())
print("rows per day:\n", recent["timestamp"].dt.date.value_counts().sort_index().to_string())
print("device_names:\n", df["device_name"].value_counts().to_string())
print("metrics per device:\n", df.groupby(["device_name", "metric_name", "metric_datatype"])
      .size().sort_values(ascending=False).to_string())
print("value column filled, per datatype:\n",
      df.groupby("metric_datatype")[["metric_str_v", "metric_long_v", "metric_double_v", "metric_bool_v"]]
      .agg(lambda s: s.notna().mean()).round(2).to_string())
print("\nlast rows:\n", (recent if len(recent) else df).tail(10).to_string())

step("done")
