# ⚠️ Duplicate Feature Vector Leakage Report

**Document Purpose**: Documentation of 28-Feature Vector Duplication in Phase 1 Datasets  
**Date**: September 19, 2026  

## Summary of Findings
- **Dataset**: `CSE-CIC-IDS2018`
- **Total Scanned Rows**: `16,232,943`
- **Unique 28-Feature Vectors**: `11,442,399`
- **Exact Duplicate Rows**: `4,790,544` (**29.51%**)
- **Duplicate Groups**: `700,590`
- **Conflicting Label Groups**: `68,679`

## Experimental Impact & Safeguards
1. **Cross-Split Boundary Crossing**: Random row-level splitting allows identical 28-feature vectors to appear in both Train and Test sets.
2. **Handling Policy**: Duplicate feature vectors are **acknowledged and logged** in `split_manifest.csv`, but are **NOT removed** prior to splitting to preserve standardized dataset benchmark protocols.
3. **Reporting Requirement**: Documented as a methodology limitation in Phase 1 evaluation.
