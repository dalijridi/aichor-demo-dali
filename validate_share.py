"""Validate the Wilo Delta Share against the 21 Oct 2026 Databricks export.

Runs on AIchor only: the share admits AIchor's outbound IP, not laptops.
Reads DELTA_SHARING_ENDPOINT and DELTA_SHARING_BEARER_TOKEN, the runtime
variables already set in the DeltaSharingProject.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timezone

import delta_sharing
import pandas as pd

TABLE = "speaking_water.monitor_share.v_speaking_water_telemetry"
GATEWAY = "25dbf102-4b9d-4e82-99ec-1dfdfcbc072a"

# Weekly row counts (Monday 00:00Z weeks) from Databricks at the 21 Oct export, 00_counts.csv.
REFERENCE = {
    "2026-06-15": 142228, "2026-06-22": 201594, "2026-06-29": 172994,
    "2026-07-06": 218216, "2026-07-13": 217638, "2026-07-20": 217534,
    "2026-07-27": 227136, "2026-08-03": 395686, "2026-08-10": 462292,
    "2026-08-17": 377016, "2026-08-24": 358070, "2026-08-31": 432048,
    "2026-09-07": 433403, "2026-09-14": 430909, "2026-09-21": 441507,
    "2026-09-28": 428360, "2026-10-05": 403618, "2026-10-12": 444287,
    "2026-10-19": 154046,
}
# The export left out slaves 1 and 3 from 18 Aug to 1 Sep, and stopped on 21 Oct,
# so these weeks should be HIGHER in the share. Every other week should match.
EXPECTED_HIGHER = {"2026-08-17", "2026-08-24", "2026-08-31", "2026-10-19"}
WILO_TOTAL_11_NOV = 7_633_679

# Rows per device in the export (16 Jun -> 21 Oct). The share should be >= each.
DEVICE_FLOOR = {
    f"ModbusRTU1_20_{GATEWAY}": 1_675_181,
    f"ModbusRTU1_10_{GATEWAY}": 3_364_633,
    f"ModbusRTU1_1_{GATEWAY}": 892_852,
    f"ModbusRTU1_3_{GATEWAY}": 140_776,
    f"ModbusRTU1_30_{GATEWAY}": 2_275,
    GATEWAY: 83_054,
}

t0 = time.time()


def step(msg: str) -> None:
    print(f"\n[{time.time() - t0:7.1f}s] {msg}", flush=True)


profile = {
    "shareCredentialsVersion": 1,
    "endpoint": os.environ["DELTA_SHARING_ENDPOINT"],
    "bearerToken": os.environ["DELTA_SHARING_BEARER_TOKEN"],
}
path = os.path.join(tempfile.mkdtemp(), "config.share")
with open(path, "w") as f:
    json.dump(profile, f)
url = f"{path}#{TABLE}"
problems: list[str] = []

step("A. Tables visible in the share")
for t in delta_sharing.SharingClient(path).list_all_tables():
    print("  ", f"{t.share}.{t.schema}.{t.name}")

step("B. Version pinning")
try:
    version = delta_sharing.get_table_version(url)
    print("   current version:", version)
    pinned = delta_sharing.load_as_pandas(url, version=version, limit=10)
    print(f"   pinned read OK ({len(pinned)} rows at version {version})")
except Exception as e:
    print(f"   NOT available: {type(e).__name__}: {str(e)[:300]}")
    problems.append("version pinning unavailable (ask Wilo)")

step("C. Full read (no row cap)")
df = delta_sharing.load_as_pandas(url)
read_s = time.time() - t0
print(f"   {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory")

step("D. Scope and freshness")
ts = pd.to_datetime(df["timestamp"], utc=True)
foreign = int((~df["device_name"].str.contains(GATEWAY)).sum())
lag_h = (datetime.now(timezone.utc) - ts.max()).total_seconds() / 3600
print(f"   time range: {ts.min()} -> {ts.max()}  (newest row is {lag_h:.1f} h old)")
print(f"   rows from other gateways: {foreign}")
if foreign:
    problems.append("rows outside the Brackel gateway")
if lag_h > 24:
    problems.append(f"newest row {lag_h:.0f} h old")
if len(df) < WILO_TOTAL_11_NOV:
    problems.append("fewer rows than Wilo counted on 11 Nov")

step("E. Rows per device vs export (share must be >=)")
counts = df["device_name"].value_counts()
for dev, floor in DEVICE_FLOOR.items():
    got = int(counts.get(dev, 0))
    ok = got >= floor
    print(f"   {dev[:16]:<16} share={got:>10,}  export={floor:>10,}  {'ok' if ok else 'LOW'}")
    if not ok:
        problems.append(f"device {dev[:16]} has fewer rows than the export")
extra = set(counts.index) - set(DEVICE_FLOOR)
if extra:
    print("   devices not in the export:", sorted(extra))

step("F. Weekly counts vs export")
week = (ts.dt.tz_localize(None).dt.normalize()
        - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
wk_counts = week.value_counts()
for wk, ref in REFERENCE.items():
    got = int(wk_counts.get(wk, 0))
    if got == ref:
        flag = "match"
    elif wk in EXPECTED_HIGHER and got > ref:
        flag = "higher, expected"
    else:
        flag = "MISMATCH"
        problems.append(f"week {wk}: {got - ref:+,}")
    print(f"   {wk}: share={got:>8,}  export={ref:>8,}  diff={got - ref:>+8,}  {flag}")
newer = wk_counts[wk_counts.index > max(REFERENCE)].sort_index()
print("   weeks after the export:\n", newer.to_string())

step("G. Duplicates and value columns")
dups = int(df.duplicated(subset=["kafka_topic", "kafka_partition", "kafka_offset",
                                 "device_name", "metric_name", "timestamp"]).sum())
print(f"   duplicate rows (same kafka offset, device, metric, time): {dups}")
if dups:
    problems.append(f"{dups} duplicate rows")
filled = df[["metric_str_v", "metric_long_v", "metric_double_v", "metric_bool_v"]].notna().sum(axis=1)
print(f"   rows with no value in any value column: {int((filled == 0).sum()):,}")

step("H. Spot check: cistern level, last 3 days (slave 20, Current_value)")
lvl = df[(df["device_name"] == f"ModbusRTU1_20_{GATEWAY}") & (df["metric_name"] == "Current_value")]
lvl = lvl.assign(t=pd.to_datetime(lvl["timestamp"], utc=True)).sort_values("t")
recent = lvl[lvl["t"] >= ts.max() - pd.Timedelta(days=3)]
print(f"   {len(recent):,} readings; tail:\n",
      recent[["t", "metric_long_v", "metric_double_v"]].tail(5).to_string())

print(f"\nRead time: {read_s:.0f} s for {len(df):,} rows")
print("RESULT:", "PASS" if not problems else "CHECK -> " + "; ".join(problems))
