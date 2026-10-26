# Results

No performance number was computed in this run. The required scientific Python runtime stalled before it could read a record. Values below are intentionally not estimated.

| Experiment | Design | Status | Macro F1 mean | Standard deviation |
|---|---|---|---:|---:|
| E0 | Random beat split, AAMI reference | Not computed | — | — |
| E1 | Set A, Random Forest, DS1 to DS2 | Not computed | — | — |
| E2 | Set B, Random Forest, DS1 to DS2 | Not computed | — | — |
| E3 | Set B, HistGradientBoosting, DS1 to DS2 | Not computed | — | — |
| E4 | Raw-window 1D CNN, DS1 to DS2 | Not computed | — | — |
| E5 | Best model with detected peaks | Not computed | — | — |
| E6 | External test on INCART and SVDB | Not computed | — | — |
| E7 | DS2 128 Hz downsample-resample test | Not computed | — | — |

## Execution state

The local Python 3.14 virtual environment could not create `pip` with `ensurepip`. The available package index returned no package candidates. System package metadata shows the required packages except PyWavelets, but importing the installed numerical stack stalled. The processes started for this failed import were stopped. No source record, legacy code file, or external database file was modified.

## LIMITATIONS

This system is not validated for clinical use. Do not use it for diagnosis, triage, treatment, or patient monitoring. The legacy beat-level result is not evidence of performance on new patients. The fixed DS1/DS2 design has a known potential patient-identity risk because records 201 and 202 can originate from one patient. No classifier result, detector metric, confidence interval, feature-importance result, or external-database result was computed in this run.

## Prediction comparison

`PREDICTIONS.md` was not present. No prediction comparison is available.

## Final summary

1. The pipeline is isolated in `arrhythmia_v2`.
2. It preserves the source project without modification.
3. It implements the specified five-class AAMI mapping.
4. It uses named leads, filtering, and resampling.
5. It preserves annotation and detector peak positions.
6. It implements fixed DS1 and DS2 separation.
7. It includes GroupKFold inside DS1 for model selection.
8. It includes per-record bootstrap metrics when the runtime is usable.
9. No experimental metric was computed in this environment.
10. The system is not validated for clinical use.
