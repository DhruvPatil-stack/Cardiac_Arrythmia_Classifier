"""
Expanded ECG Arrhythmia Pipeline
Downloads 8 additional databases, combines with existing MIT-BIH data,
trains a Random Forest with patient-wise split.
"""

import subprocess, sys, os, traceback, warnings
warnings.filterwarnings('ignore')

# ── STEP 1: Install packages ──────────────────────────────────────────────
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'wfdb', 'scikit-learn', 'numpy', 'joblib'],
    capture_output=True
)

import wfdb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import joblib

WORK_DIR = r"C:\Users\dhruv\OneDrive\Desktop\Arrythmia Project"
os.chdir(WORK_DIR)

# ── LOGGING HELPERS ───────────────────────────────────────────────────────
download_log   = open("download_log.txt", "w")
skipped_log    = open("skipped_log.txt", "w")
unmapped_file  = open("unmapped_symbols.txt", "w")

unmapped_seen = set()

def log_download(msg):
    print(msg)
    download_log.write(msg + "\n")
    download_log.flush()

def log_skipped(msg):
    skipped_log.write(msg + "\n")
    skipped_log.flush()

def log_unmapped(sym, db, rec):
    key = (sym, db)
    if key not in unmapped_seen:
        unmapped_seen.add(key)
        unmapped_file.write(f"db={db}  rec={rec}  symbol='{sym}'\n")
        unmapped_file.flush()

# ── STEP 2: DATABASES TO DOWNLOAD ────────────────────────────────────────
EXTRA_DBS = {
    'afdb':     'afdb',
    'vfdb':     'vfdb',
    'nstdb':    'nstdb',
    'edb':      'edb',
    'svdb':     'svdb',
    'incartdb': 'incartdb',
    'ltdb':     'ltdb',
    'cudb':     'cudb',
}

db_dirs = {}        # db_name -> local subdirectory path
failed_dbs = []
success_dbs = []

for db_name, db_id in EXTRA_DBS.items():
    db_dir = os.path.join(WORK_DIR, db_name)
    os.makedirs(db_dir, exist_ok=True)
    # Check if already downloaded (at least 1 .hea file present)
    existing = [f for f in os.listdir(db_dir) if f.endswith('.hea')]
    if existing:
        print(f"[SKIP DOWNLOAD] {db_name}: already has {len(existing)} .hea files")
        db_dirs[db_name] = db_dir
        success_dbs.append(db_name)
        continue
    print(f"[DOWNLOADING] {db_name} ...")
    try:
        wfdb.dl_database(db_id, dl_dir=db_dir)
        hea_count = len([f for f in os.listdir(db_dir) if f.endswith('.hea')])
        if hea_count == 0:
            raise RuntimeError("No .hea files after download")
        log_download(f"SUCCESS: {db_name} — {hea_count} records")
        db_dirs[db_name] = db_dir
        success_dbs.append(db_name)
    except Exception as e:
        msg = f"FAILED: {db_name} — {e}"
        log_download(msg)
        failed_dbs.append(db_name)

print(f"\nDownloaded: {success_dbs}")
print(f"Failed:     {failed_dbs}\n")

# ── STEP 3 & 4: ANNOTATION SURVEY + FEATURE EXTRACTION ───────────────────
window = 90

# BEAT-LEVEL annotation types (rhythm-only markers are NOT beat labels)
# We check each record: if it has any beat symbol we process it.
RHYTHM_ONLY_SYMBOLS = {
    '(', ')', '[', '!', ']', 'x', '|', '~', '+', 's', 'T', '*', 'D',
    '=', '"', '@', 'p', 'u', '^', 'M', 'h', 'H', 'q', 'W', 'K', 'U'
}

# Known beat symbols across all PhysioNet databases
ALL_BEAT_SYMBOLS = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j',
    'n', 'E', '/', 'f', 'Q', '?', 'P', 'I', 'i'
}

# ── CLASS MAPPING: from raw symbol → our 7 target classes ────────────────
# N  = Normal
# V  = Ventricular ectopic (PVC and equivalents)
# A  = Atrial premature (and junctional/supraventricular premature)
# L  = Left bundle branch block
# R  = Right bundle branch block
# F  = Fusion beat
# j  = Junctional escape
CLASS_MAP = {
    # Normal
    'N':  'N', 'n':  'N', '/':  'N',          # normal, pac-with-aberr, paced
    # Ventricular
    'V':  'V', 'E':  'V', 'r':  'V',           # PVC, ventricular escape, R-on-T
    # Atrial premature / supraventricular premature
    'A':  'A', 'a':  'A', 'S':  'A', 'P': 'A', # APB, aberrant APB, SVE, premature
    'J':  'A', 'I':  'A', 'i':  'A',            # junctional premature, interpolated
    # Bundle branch block
    'L':  'L',
    'R':  'R',
    # Fusion
    'F':  'F', 'f':  'F',                        # fusion, paced fusion
    # Junctional escape
    'j':  'j', 'e':  'j',                        # junctional escape, atrial escape
    # Q / ? / B — unclassifiable / aberrant — discard
}

def extract_features(signal, r_peaks, fs):
    """Extract 10-feature vectors for all valid beats in a record."""
    features, valid_indices = [], []
    for i, peak in enumerate(r_peaks):
        if peak - window < 0 or peak + window >= len(signal):
            continue
        beat = signal[peak - window: peak + window]
        rr_prev = (r_peaks[i] - r_peaks[i-1]) / fs if i > 0 else 0
        rr_next = (r_peaks[i+1] - r_peaks[i]) / fs if i < len(r_peaks)-1 else 0
        feat = [
            float(np.max(beat)),
            float(np.min(beat)),
            float(np.mean(beat)),
            float(np.std(beat)),
            float(np.max(beat) - np.min(beat)),
            float(rr_prev),
            float(rr_next),
            float(rr_prev - rr_next),
            float(rr_prev / rr_next) if rr_next > 0 else 0.0,
            float(np.sum(np.abs(np.diff(beat)))),
        ]
        features.append(feat)
        valid_indices.append(i)
    return features, valid_indices

# ── DATA STORAGE ──────────────────────────────────────────────────────────
# Each entry: (features_array, labels_array, patient_id_string)
all_data = []   # list of (features_list, labels_list, patient_id)

# ── STEP 4a: LOAD EXISTING MIT-BIH DATA ──────────────────────────────────
MITBIH_RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

mitbih_loaded = 0
mitbih_symbols_seen = set()
for rec in MITBIH_RECORDS:
    hea = os.path.join(WORK_DIR, f"{rec}.hea")
    dat = os.path.join(WORK_DIR, f"{rec}.dat")
    atr = os.path.join(WORK_DIR, f"{rec}.atr")
    if not (os.path.exists(hea) and os.path.exists(dat) and os.path.exists(atr)):
        continue
    try:
        record = wfdb.rdrecord(os.path.join(WORK_DIR, rec))
        annotation = wfdb.rdann(os.path.join(WORK_DIR, rec), 'atr')
    except Exception as e:
        print(f"  MIT-BIH {rec}: ERROR — {e}")
        continue
    signal = record.p_signal[:, 0]
    fs = record.fs
    symbols = annotation.symbol
    r_peaks = annotation.sample
    mitbih_symbols_seen.update(symbols)

    beat_syms = [s for s in symbols if s in ALL_BEAT_SYMBOLS]
    if not beat_syms:
        log_skipped(f"mitbih/{rec}: no beat annotations, only rhythm markers")
        continue

    feats, valid_idx = extract_features(signal, r_peaks, fs)
    labels = [symbols[i] for i in valid_idx]
    patient_id = f"mitbih_{rec}"
    all_data.append((feats, labels, patient_id))
    mitbih_loaded += 1

print(f"MIT-BIH: loaded {mitbih_loaded} records")
print(f"MIT-BIH unique symbols: {sorted(mitbih_symbols_seen)}\n")

# ── STEP 4b: LOAD EXTRA DATABASES ─────────────────────────────────────────
db_records_processed = {}
db_symbols_found = {}

for db_name, db_dir in db_dirs.items():
    print(f"\n{'='*60}")
    print(f"Processing database: {db_name.upper()}")
    print(f"{'='*60}")

    # Find all annotation extension types in this db
    ann_exts_available = set()
    for f in os.listdir(db_dir):
        if '.' in f:
            ext = f.rsplit('.', 1)[1]
            if ext not in ('hea', 'dat', 'xyz', 'rec'):
                ann_exts_available.add(ext)

    # Prefer known beat-level annotation extensions
    ANN_PRIORITY = ['atr', 'ari', 'ecg', 'ann', 'qrs', 'man', 'auto', 'bak']
    ann_ext = None
    for pref in ANN_PRIORITY:
        if pref in ann_exts_available:
            ann_ext = pref
            break
    if ann_ext is None and ann_exts_available:
        ann_ext = sorted(ann_exts_available)[0]

    print(f"  Annotation extensions found: {ann_exts_available}")
    print(f"  Using extension: {ann_ext}")

    if ann_ext is None:
        log_download(f"SKIP {db_name}: no annotation files found")
        failed_dbs.append(db_name)
        if db_name in success_dbs:
            success_dbs.remove(db_name)
        continue

    # Get record list — try RECORDS file first, then scan .hea files
    records = []
    records_file = os.path.join(db_dir, 'RECORDS')
    if os.path.exists(records_file):
        with open(records_file) as rf:
            records = [line.strip() for line in rf if line.strip()]
    else:
        records = [f[:-4] for f in os.listdir(db_dir) if f.endswith('.hea')]

    # Some RECORDS files contain subdirectory paths — keep as-is
    print(f"  Records found: {len(records)}")

    all_syms_db = set()
    records_ok = 0
    records_skipped = 0

    for rec_name in records:
        rec_path = os.path.join(db_dir, rec_name)
        # Ensure the .dat and annotation files exist
        dat_path = rec_path + '.dat'
        ann_path = rec_path + '.' + ann_ext
        hea_path = rec_path + '.hea'

        if not os.path.exists(hea_path):
            log_skipped(f"{db_name}/{rec_name}: missing .hea")
            records_skipped += 1
            continue
        if not os.path.exists(dat_path):
            log_skipped(f"{db_name}/{rec_name}: missing .dat")
            records_skipped += 1
            continue
        if not os.path.exists(ann_path):
            log_skipped(f"{db_name}/{rec_name}: missing .{ann_ext}")
            records_skipped += 1
            continue

        try:
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, ann_ext)
        except Exception as e:
            log_skipped(f"{db_name}/{rec_name}: READ ERROR — {e}")
            records_skipped += 1
            continue

        symbols = annotation.symbol
        all_syms_db.update(symbols)

        # Check if any beat-level annotations exist
        beat_syms_in_rec = [s for s in symbols if s in ALL_BEAT_SYMBOLS]
        if not beat_syms_in_rec:
            log_skipped(f"{db_name}/{rec_name}: no beat annotations (only rhythm markers {set(symbols)})")
            records_skipped += 1
            continue

        signal = record.p_signal[:, 0]
        fs = record.fs
        r_peaks = annotation.sample

        if len(r_peaks) == 0:
            log_skipped(f"{db_name}/{rec_name}: zero annotations")
            records_skipped += 1
            continue

        feats, valid_idx = extract_features(signal, r_peaks, fs)
        if not feats:
            log_skipped(f"{db_name}/{rec_name}: no valid beats (all edge cases)")
            records_skipped += 1
            continue

        labels = [symbols[i] for i in valid_idx]
        patient_id = f"{db_name}_{rec_name.replace(os.sep,'_')}"
        all_data.append((feats, labels, patient_id))
        records_ok += 1

    db_symbols_found[db_name] = sorted(all_syms_db)
    db_records_processed[db_name] = (records_ok, records_skipped)
    print(f"  Unique symbols: {sorted(all_syms_db)}")
    print(f"  Processed: {records_ok} records | Skipped: {records_skipped}")

# ── STEP 5: MAP ANNOTATIONS → 7 CLASSES ──────────────────────────────────
print(f"\n{'='*60}")
print("MAPPING ANNOTATIONS TO 7 TARGET CLASSES")
print(f"{'='*60}")

X_all, y_all, pid_all = [], [], []

for feats, labels, patient_id in all_data:
    for feat, raw_sym in zip(feats, labels):
        mapped = CLASS_MAP.get(raw_sym, None)
        if mapped is None:
            # Determine which DB this came from for logging
            db_prefix = patient_id.split('_')[0]
            log_unmapped(raw_sym, db_prefix, patient_id)
        else:
            X_all.append(feat)
            y_all.append(mapped)
            pid_all.append(patient_id)

X_all = np.array(X_all, dtype=np.float64)
y_all = np.array(y_all)
pid_all = np.array(pid_all)

print(f"\nTotal beats after mapping: {len(X_all)}")
class_names = {
    'N': 'Normal',
    'V': 'PVC / Ventricular Ectopic',
    'A': 'Atrial Premature Beat',
    'L': 'Left Bundle Branch Block',
    'R': 'Right Bundle Branch Block',
    'F': 'Fusion Beat',
    'j': 'Junctional Escape',
}
print("\nClass distribution:")
for cls in sorted(class_names):
    cnt = np.sum(y_all == cls)
    pct = cnt / len(y_all) * 100 if len(y_all) > 0 else 0
    print(f"  {cls}  {class_names[cls]:35s}: {cnt:7d}  ({pct:.2f}%)")

# ── STEP 6: PATIENT-WISE TRAIN/TEST SPLIT ────────────────────────────────
print(f"\n{'='*60}")
print("PATIENT-WISE 80/20 SPLIT")
print(f"{'='*60}")

unique_patients = np.unique(pid_all)
np.random.seed(42)
np.random.shuffle(unique_patients)
split_idx = int(len(unique_patients) * 0.8)
train_patients = set(unique_patients[:split_idx])
test_patients  = set(unique_patients[split_idx:])

train_mask = np.array([p in train_patients for p in pid_all])
test_mask  = np.array([p in test_patients  for p in pid_all])

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

print(f"  Train patients: {len(train_patients)}  |  Test patients: {len(test_patients)}")
print(f"  Train beats:    {len(X_train)}  |  Test beats:    {len(X_test)}")

# Drop classes with < 10 samples in training set
valid_classes = [c for c in class_names if np.sum(y_train == c) >= 10
                 and np.sum(y_test == c) >= 1]
if len(valid_classes) < len(class_names):
    dropped = set(class_names) - set(valid_classes)
    print(f"\n  NOTE: Dropping classes with insufficient data: {dropped}")
    train_mask2 = np.isin(y_train, valid_classes)
    test_mask2  = np.isin(y_test,  valid_classes)
    X_train, y_train = X_train[train_mask2], y_train[train_mask2]
    X_test,  y_test  = X_test[test_mask2],   y_test[test_mask2]

print(f"\n  Final train: {len(X_train)} beats  |  Final test: {len(X_test)} beats")

# ── STEP 6b: TRAIN ────────────────────────────────────────────────────────
print(f"\nTraining Random Forest (200 trees, balanced, all cores) ...")
clf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Training complete.")

# ── STEP 7: EVALUATE ─────────────────────────────────────────────────────
feature_names = [
    'Max amplitude', 'Min amplitude', 'Mean voltage', 'Std deviation',
    'Peak-trough range', 'RR prev', 'RR next', 'RR diff',
    'RR ratio', 'Signal variation'
]
importances = clf.feature_importances_
report_str  = classification_report(y_test, y_pred, target_names=[class_names.get(c, c) for c in sorted(valid_classes)],
                                    labels=sorted(valid_classes))
labels_sorted = sorted(set(y_test))
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT — EXPANDED MULTI-DATABASE CLASSIFIER")
print(f"{'='*60}")
print(report_str)

print("\n── Feature Importances ──")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = '█' * int(imp * 50)
    print(f"  {name:22s}: {imp:.4f}  {bar}")

print("\n── Confusion Matrix ──")
print(f"{'Predicted →':>14}", end='')
for lbl in labels_sorted:
    print(f"  {lbl:>5}", end='')
print(f"    (actual count)")
print("  " + "─" * (8 * len(labels_sorted) + 20))
for i, lbl in enumerate(labels_sorted):
    actual_count = int(np.sum(y_test == lbl))
    print(f"  Actual {lbl:3s}    ", end='')
    for j, val in enumerate(cm[i]):
        if i == j:
            print(f"  [{val:4d}]", end='')
        elif val > 0:
            print(f"   {val:4d} ", end='')
        else:
            print(f"     · ", end='')
    print(f"    ({actual_count})")

# Save model
joblib.dump(clf, "expanded_rf.joblib")
print("\nModel saved to expanded_rf.joblib")

# Save report
with open("expanded_results.txt", "w") as f:
    f.write("EXPANDED ECG ARRHYTHMIA CLASSIFIER\n")
    f.write("Databases used:\n")
    for db in success_dbs:
        ok, sk = db_records_processed.get(db, (0, 0))
        f.write(f"  {db}: {ok} records processed, {sk} skipped\n")
    if failed_dbs:
        f.write(f"\nFailed databases: {failed_dbs}\n")

    f.write(f"\nTotal patients: {len(unique_patients)}\n")
    f.write(f"  Train patients: {len(train_patients)}\n")
    f.write(f"  Test  patients: {len(test_patients)}\n")
    f.write(f"\nTotal beats (mapped): {len(X_all)}\n")
    f.write(f"  Train beats: {len(X_train)}\n")
    f.write(f"  Test  beats: {len(X_test)}\n")

    f.write("\nClass distribution (full dataset):\n")
    for cls in sorted(class_names):
        cnt = np.sum(y_all == cls)
        f.write(f"  {cls}  {class_names[cls]:35s}: {cnt}\n")

    f.write(f"\n{'='*60}\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write(f"{'='*60}\n")
    f.write(report_str)

    f.write("\nFeature Importances:\n")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        f.write(f"  {name:22s}: {imp:.4f}\n")

    f.write("\nConfusion Matrix:\n")
    f.write("Predicted → " + "  ".join(f"{l:>5}" for l in labels_sorted) + "\n")
    for i, lbl in enumerate(labels_sorted):
        f.write(f"Actual {lbl:3s}  " + "  ".join(f"{v:5d}" for v in cm[i]) + "\n")

    f.write("\nSymbols per database:\n")
    for db, syms in db_symbols_found.items():
        f.write(f"  {db}: {syms}\n")

print("Full results saved to expanded_results.txt")

# ── STEP 8: FINAL SUMMARY ─────────────────────────────────────────────────
download_log.close()
skipped_log.close()
unmapped_file.close()

print(f"\n{'='*70}")
print("FINAL PIPELINE SUMMARY")
print(f"{'='*70}")
print(f"Databases successfully processed : {success_dbs}")
print(f"Databases that FAILED            : {failed_dbs}")
print(f"\nTotal patients                   : {len(unique_patients)}")
total_recs = sum(v[0] for v in db_records_processed.values()) + mitbih_loaded
print(f"Total records processed          : {total_recs}")
print(f"\nBeats per class (full dataset):")
for cls in sorted(class_names):
    cnt = np.sum(y_all == cls)
    print(f"  {cls}  {class_names[cls]:35s}: {cnt:7d}")
print(f"\n{'─'*70}")
print(report_str)
