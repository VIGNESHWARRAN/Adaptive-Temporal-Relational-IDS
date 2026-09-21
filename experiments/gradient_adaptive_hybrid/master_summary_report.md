# Master Summary Report: Gradient-Based Adaptive Hybrid Loss + 18 Fusion Experiments

Generated across 18 completed experiments.

| Dataset | Fusion Method | Accuracy | Macro F1 | Weighted F1 | Best Epoch | Time (s) | Final alpha (CE) | Final beta (Focal) | Final gamma (WCE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| nf_cse_cic_ids2018_v2 | concat | 0.9945 | 0.9866 | 0.9944 | 6 | 156.9 | 0.2875 | 0.6483 | 0.0643 |
| nf_cse_cic_ids2018_v2 | gated | 0.9947 | 0.9871 | 0.9946 | 6 | 135.3 | 0.2938 | 0.6421 | 0.0641 |
| nf_cse_cic_ids2018_v2 | bilinear | 0.9946 | 0.9870 | 0.9946 | 6 | 325.9 | 0.3334 | 0.5927 | 0.0738 |
| nf_cse_cic_ids2018_v2 | cross_attention | 0.9943 | 0.9863 | 0.9943 | 6 | 177.0 | 0.2815 | 0.6571 | 0.0614 |
| nf_cse_cic_ids2018_v2 | decision_ensemble | 0.9947 | 0.9871 | 0.9946 | 5 | 119.1 | 0.4234 | 0.5147 | 0.0620 |
| nf_cse_cic_ids2018_v2 | quantum_pqc | 0.9918 | 0.9804 | 0.9918 | 6 | 173.4 | 0.4213 | 0.5038 | 0.0749 |
| nf_unsw_nb15_v2 | concat | 0.9880 | 0.9189 | 0.9889 | 4 | 140.7 | 0.4876 | 0.4466 | 0.0657 |
| nf_unsw_nb15_v2 | gated | 0.9880 | 0.9193 | 0.9889 | 3 | 98.4 | 0.4690 | 0.4629 | 0.0681 |
| nf_unsw_nb15_v2 | bilinear | 0.9882 | 0.9203 | 0.9891 | 15 | 451.1 | 0.4107 | 0.5310 | 0.0583 |
| nf_unsw_nb15_v2 | cross_attention | 0.9878 | 0.9180 | 0.9887 | 15 | 238.9 | 0.4153 | 0.5280 | 0.0567 |
| nf_unsw_nb15_v2 | decision_ensemble | 0.9877 | 0.9174 | 0.9886 | 5 | 122.0 | 0.3992 | 0.5321 | 0.0687 |
| nf_unsw_nb15_v2 | quantum_pqc | 0.9875 | 0.9164 | 0.9885 | 3 | 120.3 | 0.4455 | 0.4737 | 0.0808 |
| cse_cic_ids2018 | concat | 0.9446 | 0.8328 | 0.9375 | 12 | 281.9 | 0.4544 | 0.4462 | 0.0994 |
| cse_cic_ids2018 | gated | 0.9458 | 0.8347 | 0.9377 | 14 | 275.3 | 0.4643 | 0.4387 | 0.0970 |
| cse_cic_ids2018 | bilinear | 0.9464 | 0.8305 | 0.9383 | 14 | 667.6 | 0.4624 | 0.4410 | 0.0966 |
| cse_cic_ids2018 | cross_attention | 0.9456 | 0.8325 | 0.9372 | 14 | 339.4 | 0.4634 | 0.4381 | 0.0985 |
| cse_cic_ids2018 | decision_ensemble | 0.9455 | 0.8263 | 0.9356 | 15 | 278.4 | 0.4349 | 0.4681 | 0.0971 |
| cse_cic_ids2018 | quantum_pqc | 0.9414 | 0.7804 | 0.9277 | 13 | 296.2 | 0.4628 | 0.4403 | 0.0969 |
