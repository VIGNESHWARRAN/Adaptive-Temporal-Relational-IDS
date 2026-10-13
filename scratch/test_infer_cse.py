import pickle
import torch
import json
import numpy as np
import pandas as pd
import glob
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.models.classifiers import MultimodalFusionClassifier

scaler_data = pickle.load(open('results/phase0_data_preparation/scalers_and_encoders/cse_cic_ids2018_scaler.pkl', 'rb'))
scaler = scaler_data['scaler']
features = scaler_data['features']

model = MultimodalFusionClassifier(
    fusion_type='concat',
    temp_input_dim=len(features),
    rel_node_dim=len(features),
    rel_edge_dim=0,
    num_classes=15,
    dropout_rate=0.2
)
state = torch.load('experiments/gradient_adaptive_hybrid/cse_cic_ids2018/concat/best_model.pt', map_location='cpu')
model.load_state_dict(state)
model.eval()

with open('experiments/gradient_adaptive_hybrid/cse_cic_ids2018/concat/dataset_integration_report.json') as f:
    meta = json.load(f)
label_map = meta['label_mapping']

# Load 5 sample rows from raw
p = glob.glob('data/raw/cse-cic-ids2018/*.csv')[0]
df = pd.read_csv(p, nrows=5)
# Strip whitespace from column names
df.columns = [c.strip() for c in df.columns]
X_raw = df[features].values.astype(np.float32)
X_scaled = np.clip(scaler.transform(X_raw), -100.0, 100.0)

t_temp = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
t_rel = torch.tensor(X_scaled, dtype=torch.float32)

with torch.no_grad():
    logits = model(t_temp, t_rel)
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)

for i in range(len(df)):
    pred_cls = label_map.get(str(preds[i]), str(preds[i]))
    print(f"Row {i:2d}: True={df.iloc[i].get('Label')} | Pred={pred_cls} | MaxProb={probs[i, preds[i]]:.4f}")
