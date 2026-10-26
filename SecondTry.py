import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import time

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

# ── DOWNLOAD AND PROCESS EACH RECORD ─────────────────────────────────────
for rec in all_records:
    # Check if files already exist locally
    if os.path.exists(f'{rec}.hea') and os.path.exists(f'{rec}.dat') and os.path.exists(f'{rec}.atr'):
        print(f"Record {rec}: Loading from local files...")
        try:
            record = wfdb.rdrecord(rec)
            annotation = wfdb.rdann(rec, 'atr')
        except Exception as e:
            print(f"  ERROR reading local {rec}: {e}")
            failed.append(rec)
            continue
    else:
        # Download from PhysioNet
        print(f"Record {rec}: Downloading from PhysioNet...")
        try:
            wfdb.dl_database('mitdb', dl_dir='.', records=[rec])
            record = wfdb.rdrecord(rec)
            annotation = wfdb.rdann(rec, 'atr')
            time.sleep(1)  # Be polite to the server
        except Exception as e:
            print(f"  ERROR downloading {rec}: {e}")
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
            np.max(beat),
            np.min(beat),
            np.mean(beat),
            np.std(beat),
            rr_prev,
            rr_next,
            rr_prev - rr_next,
        ]

        all_features.append(feature_vector)
        all_labels.append(labels[i])
        beat_count += 1

    downloaded.append(rec)
    print(f"  Extracted {beat_count} beats | Types: {set(labels)}")

# ── SUMMARY ───────────────────────────────────────────────────────────────
features = np.array(all_features)
beat_labels = np.array(all_labels)

print(f"\n{'='*50}")
print(f"DOWNLOAD COMPLETE")
print(f"Records processed: {len(downloaded)}/{len(all_records)}")
print(f"Records failed: {len(failed)} {failed if failed else ''}")
print(f"Total beats extracted: {len(features)}")
print(f"All beat types found: {sorted(set(beat_labels))}")
print(f"{'='*50}")

# ── FILTER AND TRAIN ──────────────────────────────────────────────────────
keep = {'N', 'V', 'A'}
mask = np.isin(beat_labels, list(keep))
X = features[mask]
y = beat_labels[mask]

print(f"\nFiltered class distribution:")
for cls in sorted(keep):
    print(f"  {cls}: {np.sum(y == cls)} beats")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,       # more trees for larger dataset
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n── Classification Report (ALL 48 RECORDS) ──")
print(classification_report(y_test, y_pred))

feature_names = ['Max amp', 'Min amp', 'Mean', 'Std dev', 'RR prev', 'RR next', 'RR diff']
importances = clf.feature_importances_

print("\n── Feature Importances ──")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")

# ── SAVE RESULTS ──────────────────────────────────────────────────────────
with open("results_all_records.txt", "w") as f:
    f.write(f"Records processed: {len(downloaded)}/{len(all_records)}\n")
    f.write(f"Total beats: {len(features)}\n\n")
    f.write("Class distribution:\n")
    for cls in sorted(keep):
        f.write(f"  {cls}: {np.sum(y == cls)}\n")
    f.write(f"\n{classification_report(y_test, y_pred)}\n")
    f.write("\nFeature Importances:\n")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        f.write(f"  {name}: {imp:.3f}\n")

print("\nResults saved to results_all_records.txt")
