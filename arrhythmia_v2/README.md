# Arrhythmia V2

This folder is an isolated inter-patient and external-database evaluation pipeline. It does not modify source records or legacy scripts.

## Requirements

Use a supported Python release with the packages in `requirements.txt`. Python 3.11 or 3.12 is the safe choice because the available Python 3.14 runtime stalled while importing the installed numerical stack.

## Run order

From this folder, run the stages in order:

```powershell
.\.venv\Scripts\python.exe pipeline.py --stage 0
.\.venv\Scripts\python.exe pipeline.py --stage 1
.\.venv\Scripts\python.exe pipeline.py --stage 2
.\.venv\Scripts\python.exe pipeline.py --stage 3
.\.venv\Scripts\python.exe pipeline.py --stage 4
.\.venv\Scripts\python.exe pipeline.py --stage 5
.\.venv\Scripts\python.exe pipeline.py --stage 6
```

Each stage writes `logs/stage_N.log`. Re-running a stage regenerates only files under this folder.

## Evaluation design

The fixed de Chazal DS1 records are training data. The fixed DS2 records are test data. The implementation asserts that the record lists do not overlap. It excludes paced records 102, 104, 107, and 217.

The program filters ECG at 0.5 to 40 Hz, resamples external ECG to 360 Hz, maps annotations to five AAMI classes, and stores original symbols independently. E0 is a beat-level reference. E1 to E4 are inter-record experiments. E5 uses detected peaks. E6 is external testing. E7 tests sampling-rate degradation.
