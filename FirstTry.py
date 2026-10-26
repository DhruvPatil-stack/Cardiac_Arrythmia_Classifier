import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── CONFIGURATION ─────────────────────────────────────────────────────────
# Records 106 and 119 have high PVC counts, 100 is normal sinus rhythm
# Together they give us a balanced training dataset
records = ['100', '106', '109', '119', '209', '232']
window = 90  # samples on each side of R-peak (90/360hz = 0.25 seconds)

# ── LOAD AND EXTRACT FEATURES FROM ALL RECORDS ────────────────────────────
all_features = []
all_labels = []

for rec in records:
    # Load ECG signal and cardiologist annotations from local files
    record = wfdb.rdrecord(rec)
    annotation = wfdb.rdann(rec, 'atr')

    signal = record.p_signal[:, 0]   # Lead I voltage over time
    fs = record.fs                    # Sampling frequency = 360 Hz
    labels = annotation.symbol        # Beat type labels (N, V, A, etc.)
    r_peaks = annotation.sample       # Sample index of each R-peak

    print(f"Record {rec}: {len(r_peaks)} beats, types: {set(labels)}")

    for i, peak in enumerate(r_peaks):
        # Skip beats too close to signal edges
        if peak - window < 0 or peak + window >= len(signal):
            continue

        # Extract 180-sample window centered on R-peak
        beat = signal[peak - window : peak + window]

        # RR intervals capture rhythm regularity
        # PVCs have short RR before, long RR after (compensatory pause)
        rr_prev = (r_peaks[i] - r_peaks[i-1]) / fs if i > 0 else 0
        rr_next = (r_peaks[i+1] - r_peaks[i]) / fs if i < len(r_peaks)-1 else 0

        # Feature vector — each value maps to a physical property of the beat
        feature_vector = [
            np.max(beat),           # R-peak amplitude (PVCs are taller)
            np.min(beat),           # S-wave depth (PVCs are deeper)
            np.mean(beat),          # mean voltage of beat window
            np.std(beat),           # voltage variability (PVCs are wider/messier)
            rr_prev,                # RR interval before this beat
            rr_next,                # RR interval after this beat
            rr_prev - rr_next,      # RR asymmetry (the PVC signature)
        ]

        all_features.append(feature_vector)
        all_labels.append(labels[i])

features = np.array(all_features)
beat_labels = np.array(all_labels)

# ── VISUALIZE 10 SECONDS FROM RECORD 100 ─────────────────────────────────
record_vis = wfdb.rdrecord('100')
signal_vis = record_vis.p_signal[:, 0]

plt.figure(figsize=(14, 4))
plt.plot(signal_vis[:3600], color='steelblue', linewidth=0.8)
plt.title("ECG Signal — Record 100 (10 seconds)")
plt.xlabel("Sample index")
plt.ylabel("Voltage (mV)")
plt.tight_layout()
plt.show()

# ── FILTER TO CLINICALLY RELEVANT BEAT TYPES ──────────────────────────────
# N = Normal | V = PVC (dangerous) | A = Atrial premature beat
keep = {'N', 'V', 'A'}
mask = np.isin(beat_labels, list(keep))
X = features[mask]
y = beat_labels[mask]

print("\nClass distribution:")
for cls in keep:
    print(f"  {cls}: {np.sum(y == cls)} beats")

# ── TRAIN CLASSIFIER ──────────────────────────────────────────────────────
# Stratify ensures each class is proportionally represented in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# class_weight='balanced' makes the model penalize missing rare classes more
# This is your sensitivity/specificity tradeoff control
clf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# ── EVALUATE ──────────────────────────────────────────────────────────────
# Never report just accuracy on imbalanced medical data
# F1 per class is what matters — especially recall on V beats
y_pred = clf.predict(X_test)

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred))

# Feature importances tell you which physical properties drove the decisions
feature_names = ['Max amp', 'Min amp', 'Mean', 'Std dev', 'RR prev', 'RR next', 'RR diff']
importances = clf.feature_importances_

print("\n── Feature Importances ──")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")