import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.ui.engine import IDSEngine

engine = IDSEngine()
print("Engine initialized successfully!")
status = engine.get_status()
print(f"Status: Active File: {status['active_file']} | Total rows: {status['total_rows']}")
res = engine.trigger(n=3)
print(f"Triggered n=3: Success={res['success']} | Records count={len(res['records'])}")
for r in res['records']:
    print(f" -> {r['id']} | Verdict: {r['verdict']} | Attack: {r['attack_signature']} | Conf: {r['confidence']}% | Lat: {r['latency_ms']}ms")
