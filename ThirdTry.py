import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# ── ALL 48 MIT-BIH RECORDS ───────────────────────────────────────────────
all_records = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

window = 90
all_features = []
all_labels = []
downloaded = []
failed = []

# ── LOAD ALL RECORDS (already downloaded from SecondTry) ──────────────────
for rec in all_records:
    if not (os.path.exists(f'{rec}.hea') and os.path.exists(f'{rec}.dat') and os.path.exists(f'{rec}.atr')):
        print(f"Record {rec}: FILES MISSING — skipping")
        failed.append(rec)
        continue

    try:
        record = wfdb.rdrecord(rec)
        annotation = wfdb.rdann(rec, 'atr')
    except Exception as e:
        print(f"Record {rec}: ERROR — {e}")
        failed.append(rec)
        continue

    signal = record.p_signal[:, 0]
    fs = record.fs
    labels = annotation.symbol
    r_peaks = annotation.sample

    beat_count = 0
    for i, peak in enumerate(r_peaks):
        if peak - window < 0 or peak + window >= len(signal):
            continue

        beat = signal[peak - window : peak + window]

        rr_prev = (r_peaks[i] - r_peaks[i-1]) / fs if i > 0 else 0
        rr_next = (r_peaks[i+1] - r_peaks[i]) / fs if i < len(r_peaks)-1 else 0

        feature_vector = [
            np.max(beat),           # R-peak amplitude
            np.min(beat),           # S-wave depth
            np.mean(beat),          # mean voltage
            np.std(beat),           # voltage variability (morphology)
            np.max(beat) - np.min(beat),  # NEW: peak-to-trough range
            rr_prev,               # RR interval before
            rr_next,               # RR interval after
            rr_prev - rr_next,     # RR asymmetry
            rr_prev / rr_next if rr_next > 0 else 0,  # NEW: RR ratio
            np.sum(np.abs(np.diff(beat))),  # NEW: total signal variation (captures QRS width)
        ]

        all_features.append(feature_vector)
        all_labels.append(labels[i])
        beat_count += 1

    downloaded.append(rec)
    print(f"Record {rec}: {beat_count} beats | Types: {set(labels)}")

features = np.array(all_features)
beat_labels = np.array(all_labels)

print(f"\n{'='*60}")
print(f"LOADING COMPLETE — {len(downloaded)} records, {len(features)} total beats")
print(f"{'='*60}")

# ── EXPANDED CLASSIFICATION ───────────────────────────────────────────────
# N  = Normal beat
# V  = Premature ventricular contraction (rogue ventricular cell fires)
# A  = Atrial premature beat (rogue atrial cell fires)
# L  = Left bundle branch block (left bundle highway is broken)
# R  = Right bundle branch block (right bundle highway is broken)
# F  = Fusion beat (PVC and normal beat collide mid-ventricle)
# j  = Junctional escape (AV node takes over as pacemaker)

keep = {'N', 'V', 'A', 'L', 'R', 'F', 'j'}
mask = np.isin(beat_labels, list(keep))
X = features[mask]
y = beat_labels[mask]

print(f"\nExpanded class distribution:")
print(f"{'─'*40}")
class_names = {
    'N': 'Normal',
    'V': 'PVC (Premature Ventricular)',
    'A': 'APB (Atrial Premature Beat)',
    'L': 'Left Bundle Branch Block',
    'R': 'Right Bundle Branch Block',
    'F': 'Fusion Beat',
    'j': 'Junctional Escape'
}
for cls in sorted(keep):
    count = np.sum(y == cls)
    pct = count / len(y) * 100
    print(f"  {cls} — {class_names.get(cls, '???'):35s}: {count:6d} beats ({pct:.1f}%)")

# ── CHECK: skip classes with fewer than 10 beats ─────────────────────────
valid_classes = [cls for cls in keep if np.sum(y == cls) >= 10]
if len(valid_classes) < len(keep):
    dropped = keep - set(valid_classes)
    print(f"\n  WARNING: Dropping classes with <10 beats: {dropped}")
    mask2 = np.isin(y, valid_classes)
    X = X[mask2]
    y = y[mask2]

# ── TRAIN ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # use all CPU cores for speed
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ── RESULTS ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CLASSIFICATION REPORT — 7 ARRHYTHMIA TYPES ACROSS 48 PATIENTS")
print(f"{'='*60}")
print(classification_report(y_test, y_pred))

# ── FEATURE IMPORTANCES ──────────────────────────────────────────────────
feature_names = [
    'Max amp', 'Min amp', 'Mean', 'Std dev', 'Peak-trough range',
    'RR prev', 'RR next', 'RR diff', 'RR ratio', 'Signal variation'
]
importances = clf.feature_importances_

print("\n── Feature Importances ──")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = '█' * int(imp * 50)
    print(f"  {name:20s}: {imp:.3f} {bar}")

# ── CONFUSION MATRIX ─────────────────────────────────────────────────────
labels_sorted = sorted(set(y_test))
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

print("\n── Confusion Matrix ──")
print(f"{'Predicted →':>12}", end='')
for lbl in labels_sorted:
    print(f"{lbl:>7}", end='')
print(f"  {'(actual count)':>14}")
print("  " + "─" * (7 * len(labels_sorted) + 25))

for i, lbl in enumerate(labels_sorted):
    actual_count = np.sum(y_test == lbl)
    print(f"  Actual {lbl}  ", end='')
    for j, val in enumerate(cm[i]):
        if i == j:
            print(f"  [{val:4d}]", end='')  # highlight diagonal
        elif val > 0:
            print(f"   {val:4d}", end='')    # misclassifications
        else:
            print(f"      ·", end='')        # zero = clean
    print(f"     ({actual_count})")

# ── SAVE EVERYTHING ──────────────────────────────────────────────────────
with open("results_7class.txt", "w") as f:
    f.write("ARRHYTHMIA CLASSIFIER — 7 CLASSES, 48 PATIENTS\n")
    f.write(f"Total beats: {len(features)}\n")
    f.write(f"Filtered beats: {len(X)}\n\n")
    f.write("Class distribution:\n")
    for cls in sorted(keep):
        count = np.sum(y == cls)
        f.write(f"  {cls} — {class_names.get(cls, '???')}: {count}\n")
    f.write(f"\n{classification_report(y_test, y_pred)}\n")
    f.write("\nFeature Importances:\n")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        f.write(f"  {name}: {imp:.3f}\n")

print("\nResults saved to results_7class.txt")