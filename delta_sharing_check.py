import json, os, tempfile, urllib.request
import delta_sharing

print("egress IP:", urllib.request.urlopen("https://ifconfig.me/ip").read().decode())

profile = {
    "shareCredentialsVersion": 1,
    "endpoint": os.environ["DELTA_SHARING_ENDPOINT"],
    "bearerToken": os.environ["DELTA_SHARING_BEARER_TOKEN"],
}
path = os.path.join(tempfile.mkdtemp(), "config.share")
with open(path, "w") as f:
    json.dump(profile, f)

client = delta_sharing.SharingClient(path)
tables = client.list_all_tables()
print("tables:", tables)                         # tests the sharing server + recipient IP list

t = tables[0]
url = f"{path}#{t.share}.{t.schema}.{t.name}"
df = delta_sharing.load_as_pandas(url, limit=100)  # tests the signed URLs + storage firewall
print(df.shape); print(df.head())
