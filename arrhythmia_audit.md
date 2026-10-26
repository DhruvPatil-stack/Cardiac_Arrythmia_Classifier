# Arrhythmia Classifier Audit
**Audit Date:** 2026-10-02  
**Auditor:** Claude Code (forensic read-only pass)  
**Project:** MIT-BIH Arrhythmia Classifier — scikit-learn Random Forest  
**Directory:** `C:\Users\dhruv\OneDrive\Desktop\Arrythmia_Project`

---

## 1. Evidence Table

| Item | Value | Source file : line | Confidence |
|------|-------|--------------------|------------|
| **DATA — record IDs (FirstTry)** | `['100','106','109','119','209','232']` (6 records) | FirstTry.py : 11 | HIGH |
| **DATA — record IDs (SecondTry / ThirdTry)** | All 48 standard MIT-BIH records | SecondTry.py : 11-16; ThirdTry.py : 10-16 | HIGH |
| **DATA — sampling rate** | 360 Hz | 100.hea : 1 (confirmed across all .hea files) | HIGH |
| **DATA — lead** | Channel 0 only (`p_signal[:, 0]`) | FirstTry.py : 23; SecondTry.py : 50; ThirdTry.py : 39 | HIGH |
| **DATA — loader** | `wfdb.rdrecord()`, `wfdb.rdann()`, local .dat/.atr files | FirstTry.py : 20-21; SecondTry.py : 31-32 | HIGH |
| **LABELS — raw symbols kept (3-class)** | `'N','L','R','/'` → N; `'A','a','S','J','j'` → A; `'V','E'` → V; others discarded | FirstTry.py : 28-35; SecondTry.py : 55-65 | HIGH |
| **LABELS — class mapping (7-class)** | N→N, V/E/r→V, A/a/S/P/J/I/i→A, L→L, R→R, F/f→F, j/e→j; Q/?/B discarded | ExpandedPipeline.py : 117-133 | HIGH |
| **LABELS — class counts (3-class)** | N: 75,033 \| A: 2,546 \| V: 7,129 \| Total: 84,708 | results_all_records.txt : 4-7 | HIGH |
| **LABELS — class counts (7-class)** | N: 75,033 \| L: 8,073 \| R: 7,257 \| V: 7,129 \| A: 2,546 \| F: 802 \| j: 229 \| Total: 101,069 | results_7class.txt : 6-12 | HIGH |
| **PREPROCESSING — filtering** | NOT APPLIED (no bandpass/butter/highpass anywhere) | All .py files — absent | HIGH |
| **PREPROCESSING — R-peak detection** | NOT computed; loaded from cardiologist `.atr` annotations via `wfdb.rdann()` | FirstTry.py : 21; SecondTry.py : 32 | HIGH |
| **PREPROCESSING — beat window** | 90 samples before and 90 after R-peak = 180-sample window (~0.5 s at 360 Hz) | FirstTry.py : 12, 32-33 | HIGH |
| **PREPROCESSING — normalization/scaling** | NOT APPLIED; no StandardScaler/MinMaxScaler present anywhere | All .py files — absent | HIGH |
| **FEATURES — 3-class model (7 features)** | max, min, mean, std, rr_prev, rr_next, rr_prev−rr_next | FirstTry.py : 44-52; SecondTry.py : 65-73 | HIGH |
| **FEATURES — 7-class model (10 features)** | Above 7 + (max−min), rr_prev/rr_next, sum(abs(diff(beat))) | ThirdTry.py : 54-65 | HIGH |
| **FEATURES — matrix shape (3-class)** | 84,708 × 7 total; train ≈ 67,766 × 7; test ≈ 16,942 × 7 | Derived from results_all_records.txt : 4-7 + FirstTry.py : 85-87 | HIGH |
| **FEATURES — matrix shape (7-class)** | 101,069 × 10 total; train ≈ 80,855 × 10; test ≈ 20,214 × 10 | results_7class.txt : 6; ThirdTry.py : 121-123 | HIGH |
| **SPLIT — method (all 3 completed models)** | `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` — beats, NOT patients | FirstTry.py : 85-87; SecondTry.py : 104-106; ThirdTry.py : 121-123 | HIGH |
| **SPLIT — inter- vs intra-patient** | **INTRA-PATIENT** — beats from same patient split across train and test | FirstTry.py : 11-87 (no patient ID tracking) | HIGH |
| **SPLIT — inter-patient design (correct)** | Patient-wise 80/20 split by record ID using `np.unique(pid_all)` | ExpandedPipeline.py : 362-378 | HIGH — but NOT EXECUTED |
| **SPLIT — SMOTE / oversampling** | NOT USED | All .py files — absent | HIGH |
| **MODEL — algorithm** | `RandomForestClassifier` | FirstTry.py : 91; SecondTry.py : 108; ThirdTry.py : 125 | HIGH |
| **MODEL — n_estimators** | 100 (FirstTry); 200 (SecondTry, ThirdTry, ExpandedPipeline) | FirstTry.py : 92; SecondTry.py : 109; ThirdTry.py : 126 | HIGH |
| **MODEL — class_weight** | `'balanced'` in all models | FirstTry.py : 93; SecondTry.py : 110; ThirdTry.py : 127 | HIGH |
| **MODEL — random_state** | 42 in all models | FirstTry.py : 94; SecondTry.py : 111; ThirdTry.py : 128 | HIGH |
| **MODEL — max_depth / other params** | NOT SET (scikit-learn defaults: max_depth=None, min_samples_split=2, max_features='sqrt') | All .py files — not specified | HIGH |
| **MODEL — grid search / tuning** | NOT PERFORMED | All .py files — absent | HIGH |
| **RESULTS — 3-class accuracy** | **0.99** (overall weighted) | results_all_records.txt : 15 | HIGH |
| **RESULTS — 3-class per-class F1** | N: 0.99 \| A: 0.88 \| V: 0.96 | results_all_records.txt : 9-14 | HIGH |
| **RESULTS — 3-class per-class precision** | N: 0.99 \| A: 0.97 \| V: 0.97 | results_all_records.txt : 9-14 | HIGH |
| **RESULTS — 3-class per-class recall** | N: 1.00 \| A: 0.80 \| V: 0.94 | results_all_records.txt : 9-14 | HIGH |
| **RESULTS — 3-class macro F1** | 0.94 | results_all_records.txt : 16 | HIGH |
| **RESULTS — 7-class accuracy** | **0.97** (overall weighted) | results_7class.txt : 24 | HIGH |
| **RESULTS — 7-class per-class F1** | N:0.98 \| L:0.94 \| R:0.96 \| V:0.95 \| A:0.87 \| F:0.75 \| j:0.34 | results_7class.txt : 14-24 | HIGH |
| **RESULTS — 7-class macro F1** | 0.83 | results_7class.txt : 25 | HIGH |
| **RESULTS — confusion matrix** | Printed to console; NOT saved to file | ThirdTry.py : 153-173 | HIGH |
| **RESULTS — ROC / AUC** | NOT COMPUTED | All .py files — absent | HIGH |
| **RESULTS — ExpandedPipeline metrics** | NOT FOUND — pipeline did not complete execution | ExpandedPipeline artifacts: empty | HIGH |
| **ARTIFACTS — saved models** | NONE (.pkl / .joblib) in completed models | Directory listing | HIGH |
| **ARTIFACTS — figures** | ecg_raw.png (63.7 KB, Record 100 raw signal) | ecg_raw.png — file present | HIGH |
| **ARTIFACTS — README** | NOT FOUND | Directory listing | HIGH |
| **ARTIFACTS — requirements.txt** | NOT FOUND | Directory listing | HIGH |

---

## 2. Resume-Safe Claims

The following statements are **fully backed by evidence** and safe to state in interviews or on a resume.

> "Classified cardiac arrhythmia beats from the MIT-BIH Arrhythmia Database (all 48 standard records, 360 Hz, MLII lead) using a scikit-learn RandomForestClassifier with 200 estimators and balanced class weights."

> "Extracted 10 hand-engineered features per beat: R-peak amplitude, S-wave depth, mean voltage, voltage standard deviation, peak-to-trough range, preceding and following RR intervals, RR asymmetry, RR ratio, and total signal variation — computed from a 180-sample (±90-sample) window centered on cardiologist-annotated R-peaks."

> "Implemented a 7-class label scheme (N, V, A, L, R, F, j) derived from MIT-BIH annotation symbols, with class imbalance handled via class_weight='balanced' in the classifier."

> "Produced a 7-class model on 101,069 beats achieving a weighted F1 of 0.97 and a macro F1 of 0.83 on a held-out 20% random beat split (random_state=42, stratify=y)."

> "Designed an inter-patient train/test isolation strategy (patient-wise 80/20 split by record ID) in a subsequent pipeline to address generalization concerns."

---

## 3. Claims I Must NOT Make Yet

| Claim to avoid | Why it is unsupported |
|----------------|-----------------------|
| "99% / 97% accuracy on unseen patients" | The completed models use **intra-patient random splits** — beats from the same subjects appear in both train and test sets. This inflates all reported metrics. These numbers do NOT represent generalization to new patients. |
| "The model generalizes well across patients" | The inter-patient pipeline (ExpandedPipeline.py) was never successfully executed; its metrics do not exist. |
| "Achieved high recall for atrial and junctional arrhythmias" | Recall for class A is 0.80 and for class j is only 0.22 — both are substantially lower than the headline accuracy. |
| "AUC-ROC of X" | No ROC or AUC computation exists anywhere in the project. |
| "Used signal filtering to remove noise" | No filtering of any kind was applied to the raw ECG signal. |
| "Model validated on a held-out patient cohort" | Never done in the completed experiments. |
| "Cross-validated results" | No k-fold or leave-one-patient-out cross-validation was performed. |
| "Trained on multi-database data (afdb, vfdb, etc.)" | ExpandedPipeline.py was written to do this but failed to complete execution; output logs are empty. |

---

## 4. Leakage / Methodology Risks (Ranked by Severity)

### CRITICAL — Intra-Patient Data Leakage
- **What:** `train_test_split` is applied to a pool of individual beats drawn from ALL patients simultaneously. No patient ID is tracked. Beats from the same ECG recording appear in both train and test sets.
- **Location:** FirstTry.py : 11-87; SecondTry.py : 11-106; ThirdTry.py : 10-123
- **Effect:** The model memorizes patient-specific morphology (baseline, amplitude, rhythm patterns) during training and exploits it at test time. This is the primary cause of the 99%/97% headline accuracy. True inter-patient accuracy for Random Forest classifiers on MIT-BIH typically falls to 75-90% depending on feature set.
- **Fix:** Use the inter-patient split already designed in ExpandedPipeline.py : 362-378, or leave-one-patient-out (LOPO) cross-validation.

### HIGH — No Signal Filtering
- **What:** Raw ECG signals are fed directly into feature extraction. Baseline wander (< 0.5 Hz), EMG noise, and power-line interference are not removed.
- **Location:** Preprocessing stage — absent in all .py files
- **Effect:** Features such as max, min, and mean amplitude are polluted by noise and baseline drift, reducing both accuracy and generalizability to real clinical signals.
- **Fix:** Apply a zero-phase bandpass filter (0.5–40 Hz or 0.5–100 Hz) using `scipy.signal.butter` before feature extraction.

### MEDIUM — No Model Persistence
- **What:** Trained models are never serialized to disk (no `joblib.dump` or `pickle.dump` calls).
- **Location:** All .py files — absent
- **Effect:** Reproducing exact results requires re-running training. With fixed random seeds this is deterministic, but wastes time and makes deployment impossible.
- **Fix:** Add `joblib.dump(clf, 'rf_model_7class.joblib')` after training.

### MEDIUM — No Requirements File
- **What:** No `requirements.txt` or `environment.yml` exists.
- **Location:** Directory root — absent
- **Effect:** Future runs may produce different results if package versions change (especially scikit-learn, wfdb, numpy).
- **Fix:** Run `pip freeze > requirements.txt` and commit it.

### LOW — Crude Morphological Features
- **What:** Beat morphology is summarized only by scalar statistics (max, min, mean, std). No wavelet coefficients, PCA projections, or QRS-duration proxies are used.
- **Location:** ThirdTry.py : 54-65
- **Effect:** Low discriminative power for morphologically similar arrhythmias (e.g., L vs R bundle branch block), limiting macro F1 on minority classes. Class j recall of 0.22 is a direct consequence.
- **Fix:** Add wavelet features (`pywt.wavedec`) or raw beat downsampled to fixed length as feature vector.

### LOW — No Cross-Validation
- **What:** Single train/test split only. No k-fold or LOPO CV.
- **Location:** All .py files
- **Effect:** Reported metrics have unknown variance. A single split with a favorable random seed (42) may not be representative.
- **Fix:** Implement 5-fold stratified CV or LOPO CV.

---

## 5. Gaps: What to Run or Check to Close Each NOT FOUND

| NOT FOUND item | What to do |
|----------------|------------|
| **Inter-patient performance metrics** | Fix and execute `ExpandedPipeline.py`. The patient-wise split logic at lines 362-378 is correct; debug why the pipeline stalls (likely a database download failure in the afdb section — check `download_log.txt` after running). Alternatively, add `records_train = ['101','106','107','108',...]` / `records_test = ['100','109','119','200','...]` directly in ThirdTry.py and re-run. |
| **Confusion matrix saved to file** | Add `np.savetxt('confusion_matrix.csv', cm, delimiter=',')` after ThirdTry.py : 153. |
| **ROC / AUC** | Add `from sklearn.metrics import roc_auc_score` and use `predict_proba`. For multi-class: `roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')`. |
| **Requirements.txt** | Run `pip freeze > requirements.txt` in the project virtualenv. |
| **Saved model artifact** | Add `import joblib; joblib.dump(clf, 'rf_7class.joblib')` after fitting in ThirdTry.py. |
| **Signal filtering effect** | Add `scipy.signal.butter` bandpass (0.5–100 Hz) before feature extraction. Re-run and compare results to current unfiltered results. |
| **README** | Write a brief README.md documenting: data source, feature list, model, split strategy, and how to reproduce. |
| **ExpandedPipeline execution** | Check whether wfdb `dl_database()` calls (ExpandedPipeline.py : 175-230) require internet access or a PhysioNet account. Run with verbose output and inspect the first error. |

---

## Appendix: File Inventory

| File | Last Modified | Role |
|------|---------------|------|
| FirstTry.py | Apr 17 | 6-record, 3-class baseline |
| SecondTry.py | Apr 17 | 48-record, 3-class |
| ThirdTry.py | Apr 17 | 48-record, 7-class (most complete) |
| ExpandedPipeline.py | Apr 23 | Multi-DB + inter-patient split (NOT executed) |
| results_all_records.txt | Apr 17 04:21 | 3-class metrics output |
| results_7class.txt | Apr 17 04:31 | 7-class metrics output |
| ecg_raw.png | Apr 17 02:45 | Record 100 visualization |
| output.txt | Apr 21 20:09 | Contains only `0\n1\n2\n3\n4\n` — debug artifact |
| download_log.txt | Apr 23 23:18 | Empty (pipeline stalled) |
| skipped_log.txt | Apr 23 23:18 | Empty |
| unmapped_symbols.txt | Apr 23 23:18 | Empty |
| `data/` (48 records × .dat/.hea/.atr) | — | MIT-BIH local files, read-only |
| `afdb/` | Apr 23 | Partially downloaded AF database |
