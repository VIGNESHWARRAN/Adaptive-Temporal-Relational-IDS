import sys, os
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
import src.ui.app as ui_app
from src.ui.engine import IDSEngine

engine_instance = IDSEngine(
    watch_dir="incoming_flows",
    default_batch_size=2,
    dataset_name="nf_cse_cic_ids2018_v2",
    fusion_type="concat"
)
ui_app.engine = engine_instance

client = TestClient(ui_app.app)

# 1. Test root page
res = client.get("/")
print("GET / status_code:", res.status_code, "Length:", len(res.text))
assert res.status_code == 200, "Root page failed"

# 2. Test status API
res_status = client.get("/api/status")
print("GET /api/status status_code:", res_status.status_code)
assert res_status.status_code == 200, "Status endpoint failed"
status_data = res_status.json()
print(" -> Active file:", status_data["active_file"], "Batch size:", status_data["batch_size"])

# 3. Test trigger API (n=2)
res_trig = client.post("/api/trigger", json={"n": 2})
print("POST /api/trigger status_code:", res_trig.status_code)
assert res_trig.status_code == 200, "Trigger endpoint failed"
trig_data = res_trig.json()
print(" -> Success:", trig_data["success"], "Records count:", len(trig_data["records"]))
for r in trig_data["records"]:
    print(f"    Flow {r['id']}: {r['verdict']} | {r['attack_signature']} | Conf: {r['confidence']}%")

# 4. Test file switch
res_switch = client.post("/api/select_file", json={"filename": "attack_intrusion_traffic.csv"})
print("POST /api/select_file status_code:", res_switch.status_code)
assert res_switch.status_code == 200, "Select file endpoint failed"

# 5. Test trigger on attack file
res_trig_atk = client.post("/api/trigger", json={"n": 2})
print("POST /api/trigger on attacks:", res_trig_atk.status_code)
trig_atk_data = res_trig_atk.json()
for r in trig_atk_data["records"]:
    print(f"    Attack Flow {r['id']}: {r['verdict']} | {r['attack_signature']} | Conf: {r['confidence']}%")

print("\n[ALL API TESTS PASSED SUCCESSFULLY!]")
