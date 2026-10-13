import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.ui.engine import IDSEngine

engine = IDSEngine()
engine.set_active_file("attack_intrusion_traffic.csv")
res = engine.trigger(n=4)
print(f"Triggered 4 attack flows from {engine.active_filename}:")
for r in res["records"]:
    print(f"  {r['id']} | Verdict: {r['verdict']} | Attack: {r['attack_signature']} | Conf: {r['confidence']}%")
