# Engineering Decisions

1. The immutable project root is the local MIT-BIH source. All generated files are in `arrhythmia_v2`.
2. The de Chazal DS1 and DS2 record lists in the task are used as the fixed evaluation split. The lists are disjoint.
3. Records 102, 104, 107, and 217 are excluded from every non-legacy analysis because they are paced records.
4. The lead is selected by its name. The code fails a record safely when the requested lead is absent.
5. ECG is filtered before resampling with a fourth-order, zero-phase 0.5 to 40 Hz Butterworth band-pass filter.
6. Annotated beats are used for E0 to E4, E6, and E7. E5 uses detector peaks matched to annotations.
7. The original annotation symbol is retained independently from the AAMI class label.
8. No external beat is used for training, tuning, scaling, or model selection.
9. Hyperparameter selection is limited to GroupKFold folds inside DS1. The held-out DS2 records are never used for tuning.
10. Records 201 and 202 can represent one patient. This is a known residual leak risk across DS1 and DS2.
11. The local package index did not provide PyWavelets for Python 3.14. The pipeline uses an internal db4 discrete-wavelet transform when `pywt` is unavailable.
12. The local package index did not provide install candidates in this run. The virtual environment is configured to use the already installed scientific packages.
