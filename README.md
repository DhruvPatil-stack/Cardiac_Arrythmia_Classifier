# Cardiac Arrhythmia Classifier

Machine learning pipelines for ECG beat classification on the MIT-BIH Arrhythmia Database (and the AF Database for external testing). The repository holds iterative experiments at the project root plus a cleaner, isolated inter-patient evaluation pipeline under `arrhythmia_v2/`.

## Repository layout

```
.
├── FirstTry.py              # Baseline Random Forest on a handful of records
├── SecondTry.py             # Expanded feature set
├── ThirdTry.py              # Further iteration
├── ExpandedPipeline.py      # Multi-database pipeline (downloads + trains)
├── arrhythmia_audit.md      # Audit notes on the earlier experiments
├── ecg_raw.png              # Sample ECG plot
├── results_*.txt            # Classification reports from earlier runs
└── arrhythmia_v2/           # Isolated inter-patient + external-DB pipeline
    ├── pipeline.py          # Stage-driven pipeline (stages 0–6)
    ├── config.py            # DS1/DS2 record lists, AAMI mapping, paths
    ├── requirements.txt
    ├── README.md
    ├── DECISIONS.md         # Engineering decisions log
    ├── RESULTS.md
    └── logs/stage_*.log     # Per-stage execution logs
```

## Datasets

Raw ECG records are **not** committed. Download them from PhysioNet before running:

- MIT-BIH Arrhythmia Database — https://physionet.org/content/mitdb/
  Place `.dat` / `.atr` / `.hea` files at the project root.
- AF Database — https://physionet.org/content/afdb/
  Place files under `afdb/`.

## `arrhythmia_v2/` — recommended entry point

Inter-patient evaluation using the de Chazal DS1 / DS2 split (paced records 102, 104, 107, 217 excluded), AAMI 5-class mapping (N, S, V, F, Q), 0.5–40 Hz Butterworth filter, resampling to 360 Hz, and external testing on AF DB.

Setup:

```powershell
cd arrhythmia_v2
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the stages in order (each writes `logs/stage_N.log`):

```powershell
.\.venv\Scripts\python.exe pipeline.py --stage 0
.\.venv\Scripts\python.exe pipeline.py --stage 1
.\.venv\Scripts\python.exe pipeline.py --stage 2
.\.venv\Scripts\python.exe pipeline.py --stage 3
.\.venv\Scripts\python.exe pipeline.py --stage 4
.\.venv\Scripts\python.exe pipeline.py --stage 5
.\.venv\Scripts\python.exe pipeline.py --stage 6
```

Python 3.11 or 3.12 is recommended (see `arrhythmia_v2/DECISIONS.md` for the rationale).

## License

No license specified yet.
