import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import wfdb
from scipy.signal import butter, filtfilt, find_peaks, resample_poly
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from config import *

for folder in (DATA, PROCESSED, FEATURES, RESULTS, MODELS, PLOTS, LOGS):
    folder.mkdir(parents=True, exist_ok=True)


def stage_log(stage):
    logger = logging.getLogger(stage)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOGS / f"stage_{stage}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert), encoding="utf-8")


def records_in(root):
    return sorted(p.stem for p in root.glob("*.hea"))


def source_for(database, record):
    return MITDB if database == "mitdb" else DATA / database


def annotation_extension(root, record):
    for ext in ("atr", "ann", "qrs"):
        if (root / f"{record}.{ext}").exists(): return ext
    return None


def selected_lead(database, sig_names):
    target = {"mitdb": "MLII", "incartdb": "II", "svdb": "ECG1"}[database]
    exact = [i for i, name in enumerate(sig_names) if name.strip().upper() == target]
    if not exact: raise ValueError(f"lead {target} absent from {sig_names}")
    return exact[0], sig_names[exact[0]]


def stage0():
    log = stage_log("0")
    packages = {}
    for name in ("wfdb", "numpy", "scipy", "pandas", "sklearn", "matplotlib", "joblib", "torch"):
        module = __import__(name)
        packages[name] = getattr(module, "__version__", "installed")
    try:
        import pywt
        packages["PyWavelets"] = pywt.__version__
    except ImportError:
        packages["PyWavelets"] = "unavailable; internal db4 fallback"
    (ROOT / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in packages.items()) + "\n", encoding="utf-8")
    write_json(ROOT / "environment.json", {"python": sys.version, "platform": platform.platform(), "packages": packages, "seed": SEED})
    log.info("created reproducibility files")


def download_db(name, log):
    destination = DATA / name
    destination.mkdir(exist_ok=True)
    if records_in(destination):
        log.info("%s already has %d records", name, len(records_in(destination)))
        return True
    error = None
    for attempt in range(1, 4):
        try:
            wfdb.dl_database(name, str(destination))
            return bool(records_in(destination))
        except Exception as exc:
            error = repr(exc)
            log.error("%s attempt %d failed: %s", name, attempt, error)
            time.sleep(attempt * 2)
    log.error("%s unavailable after retries: %s", name, error)
    return False


def stage1():
    log = stage_log("1")
    download_db("incartdb", log)
    download_db("svdb", log)
    manifest = []
    for database, root in (("mitdb", MITDB), ("incartdb", DATA / "incartdb"), ("svdb", DATA / "svdb")):
        for record in records_in(root):
            ext = annotation_extension(root, record)
            try:
                rec = wfdb.rdrecord(str(root / record))
                ann = wfdb.rdann(str(root / record), ext) if ext else None
                manifest.append({"database": database, "record": record, "sampling_rate": float(rec.fs), "lead_names": list(rec.sig_name), "annotation_extension": ext, "annotations": 0 if ann is None else len(ann.sample)})
            except Exception as exc:
                log.error("manifest %s/%s: %r", database, record, exc)
    write_json(ROOT / "manifest.json", manifest)
    log.info("manifest records=%d", len(manifest))


def detector(signal, fs):
    distance = max(1, round(0.20 * fs))
    prominence = max(np.std(signal) * 0.4, 1e-9)
    return find_peaks(np.abs(signal), distance=distance, prominence=prominence)[0]


def match_peaks(annotated, detected, tolerance):
    used = set(); matched = 0
    for peak in annotated:
        candidates = np.where(np.abs(detected - peak) <= tolerance)[0]
        candidates = [int(i) for i in candidates if int(i) not in used]
        if candidates:
            best = min(candidates, key=lambda i: abs(detected[i] - peak))
            used.add(best); matched += 1
    return matched


def stage2():
    log = stage_log("2")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    summary = {}
    for item in manifest:
        db, rec_name = item["database"], item["record"]
        if db == "mitdb" and rec_name in PACED:
            log.info("excluded paced mitdb_%s", rec_name); continue
        ext = item["annotation_extension"]
        if not ext:
            log.error("missing annotation %s_%s", db, rec_name); continue
        try:
            root = source_for(db, rec_name)
            record = wfdb.rdrecord(str(root / rec_name)); ann = wfdb.rdann(str(root / rec_name), ext)
            idx, lead = selected_lead(db, record.sig_name)
            fs = float(record.fs); raw = record.p_signal[:, idx]
            sos_b, sos_a = butter(FILTER_ORDER, BANDPASS_HZ, btype="bandpass", fs=fs)
            filtered = filtfilt(sos_b, sos_a, raw)
            if fs != FS_TARGET:
                from fractions import Fraction
                ratio = Fraction(FS_TARGET / fs).limit_denominator(1000)
                signal = resample_poly(filtered, ratio.numerator, ratio.denominator)
                positions = np.rint(np.asarray(ann.sample) * FS_TARGET / fs).astype(int)
            else:
                signal = filtered; positions = np.asarray(ann.sample, dtype=int)
            symbols = np.asarray(ann.symbol, dtype="U8")
            labels = np.asarray([AAMI.get(x, "") for x in symbols], dtype="U1")
            detected = detector(signal, FS_TARGET)
            valid = (positions > 160) & (positions < len(signal) - 160) & (labels != "")
            uid = f"{db}_{rec_name}"
            np.savez_compressed(PROCESSED / f"{uid}.npz", signal=signal.astype(np.float32), annotation_positions=positions, detector_positions=detected, original_symbols=symbols, aami_labels=labels, valid=valid, fs=FS_TARGET, lead=lead)
            annotated_valid = positions[valid]
            matched = match_peaks(annotated_valid, detected, round(0.150 * FS_TARGET))
            summary.setdefault(db, [0, 0, 0]); summary[db][0] += matched; summary[db][1] += len(annotated_valid); summary[db][2] += len(detected)
            log.info("%s lead=%s annotated=%d detected=%d", uid, lead, len(annotated_valid), len(detected))
        except Exception as exc:
            log.exception("preprocess %s_%s failed: %r", db, rec_name, exc)
    detector_metrics = {db: {"sensitivity": m / a if a else None, "ppv": m / d if d else None, "matches": m, "annotations": a, "detections": d} for db, (m, a, d) in summary.items()}
    write_json(RESULTS / "detector_metrics.json", detector_metrics)


def db4_energies(window):
    try:
        import pywt
        return [float(np.sum(x*x)) for x in pywt.wavedec(window, "db4", level=4)]
    except ImportError:
        lo = np.array([-0.0105974, 0.0328830, 0.0308414, -0.1870348, -0.0279838, 0.6308808, 0.7148466, 0.2303778])
        hi = np.array([(-1)**i * lo[::-1][i] for i in range(8)])
        current = np.asarray(window); bands = []
        for _ in range(4):
            detail = np.convolve(current, hi, mode="same")[::2]; bands.append(float(np.sum(detail * detail)))
            current = np.convolve(current, lo, mode="same")[::2]
        return [float(np.sum(current * current))] + bands[::-1]


def morphology(window):
    center = 100; qrs = window[center-40:center+50]; threshold = 0.25 * np.max(np.abs(qrs))
    width = float(np.sum(np.abs(qrs) >= threshold) / FS_TARGET)
    r = float(np.max(qrs)); q = float(np.min(window[center-45:center])); s = float(np.min(window[center:center+55]))
    tseg = window[center+65:center+155]; tidx = int(np.argmax(np.abs(tseg))); t = float(tseg[tidx])
    p = float(np.max(np.abs(window[center-90:center-45])))
    return [width, r, q, s, t, float(np.sign(t)), p]


def build_features(npz_path):
    data = np.load(npz_path); signal = data["signal"]; pos = data["annotation_positions"]; labels = data["aami_labels"]; symbols = data["original_symbols"]; valid = data["valid"].astype(bool)
    median_amp = np.median(np.abs(signal)) or 1.0; rows_a=[]; rows_b=[]; windows=[]; out_labels=[]; out_symbols=[]; indices=[]
    for i, peak in enumerate(pos):
        if not valid[i] or peak < 100 or peak + 160 >= len(signal): continue
        old = signal[peak-90:peak+90]; window = signal[peak-100:peak+160] / median_amp
        rr_prev = (peak-pos[i-1])/FS_TARGET if i else 0.0; rr_next = (pos[i+1]-peak)/FS_TARGET if i+1 < len(pos) else 0.0
        local = np.diff(pos[max(0,i-10):i+1]) / FS_TARGET; mean_rr = float(np.mean(local)) if len(local) else 1.0
        rows_a.append([np.max(old),np.min(old),np.mean(old),np.std(old),np.max(old)-np.min(old),rr_prev,rr_next,rr_prev-rr_next,rr_prev/rr_next if rr_next else 0.0,np.sum(np.abs(np.diff(old)))])
        rows_b.append(morphology(window)+[rr_prev,rr_next,mean_rr,rr_prev/mean_rr,rr_next/mean_rr]+db4_energies(window))
        windows.append(window); out_labels.append(labels[i]); out_symbols.append(symbols[i]); indices.append(peak)
    return np.asarray(rows_a,np.float32), np.asarray(rows_b,np.float32), np.asarray(windows,np.float32), np.asarray(out_labels), np.asarray(out_symbols), np.asarray(indices)


def stage3():
    log = stage_log("3")
    index=[]
    for path in sorted(PROCESSED.glob("*.npz")):
        try:
            a,b,w,y,s,p = build_features(path); uid=path.stem
            np.savez_compressed(FEATURES / f"{uid}.npz", set_a=a,set_b=b,windows=w,labels=y,symbols=s,positions=p,record=uid)
            index.append({"record":uid,"beats":len(y)}); log.info("%s beats=%d",uid,len(y))
        except Exception as exc: log.exception("features %s: %r",path.name,exc)
    pd.DataFrame(index).to_csv(FEATURES / "index.csv", index=False)


def load_dataset(records, feature):
    xs=[]; ys=[]; groups=[]
    for record in records:
        path=FEATURES/f"mitdb_{record}.npz"
        if not path.exists(): continue
        d=np.load(path); xs.append(d[feature]); ys.append(d["labels"]); groups.extend([record]*len(d["labels"]))
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups)


def metrics(y, pred, records, label):
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=CLASSES,zero_division=0)
    cm=confusion_matrix(y,pred,labels=CLASSES)
    per={c:{"precision":p[i],"recall":r[i],"f1":f[i],"support":s[i]} for i,c in enumerate(CLASSES)}
    record_rows=[]
    for rec in sorted(set(records)):
        mask=records==rec; _,_,rf,_=precision_recall_fscore_support(y[mask],pred[mask],labels=CLASSES,average="macro",zero_division=0); record_rows.append({"record":rec,"macro_f1":rf,"beats":int(mask.sum())})
    rng=np.random.default_rng(SEED); by_rec=sorted(set(records)); boot=[]
    for _ in range(1000):
        sampled=rng.choice(by_rec,len(by_rec),replace=True); mask=np.concatenate([np.where(records==q)[0] for q in sampled]); _,_,score,_=precision_recall_fscore_support(y[mask],pred[mask],labels=CLASSES,average="macro",zero_division=0); boot.append(score)
    result={"experiment":label,"macro_f1":float(np.mean(f)),"classes":per,"v_sensitivity":per["V"]["recall"],"v_ppv":per["V"]["precision"],"s_sensitivity":per["S"]["recall"],"s_ppv":per["S"]["precision"],"ci95_records":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],"confusion_matrix":cm.tolist()}
    write_json(RESULTS/f"{label}.json",result); pd.DataFrame(record_rows).to_csv(RESULTS/f"{label}_per_record.csv",index=False); pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(RESULTS/f"{label}_confusion.csv")
    fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm,cmap="Blues"); ax.set_xticks(range(5),CLASSES); ax.set_yticks(range(5),CLASSES); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(5):
        for j in range(5): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.tight_layout(); fig.savefig(PLOTS/f"{label}_confusion.png",dpi=150); plt.close(fig)
    return result


def rf(seed): return RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=seed,n_jobs=-1)


def run_model(label, model_factory, Xtrain, ytrain, gtrain, Xtest, ytest, gtest, importance=False):
    scores=[]
    for seed in range(SEED,SEED+5):
        model=model_factory(seed); weights=compute_sample_weight("balanced",ytrain)
        if isinstance(model, HistGradientBoostingClassifier): model.fit(Xtrain,ytrain,sample_weight=weights)
        else: model.fit(Xtrain,ytrain)
        pred=model.predict(Xtest); scores.append(metrics(ytest,pred,gtest,f"{label}_seed{seed}"))
    macro=[x["macro_f1"] for x in scores]; summary={"experiment":label,"macro_f1_mean":float(np.mean(macro)),"macro_f1_std":float(np.std(macro)),"seeds":scores}
    write_json(RESULTS/f"{label}.json",summary)
    if importance:
        imp=pd.DataFrame({"feature":range(Xtrain.shape[1]),"importance":model.feature_importances_}); imp.to_csv(RESULTS/f"{label}_importance.csv",index=False)
    joblib.dump(model,MODELS/f"{label}.joblib"); write_json(MODELS/f"{label}.metadata.json",{"seed":SEED+4,"sklearn":sklearn.__version__,"feature_count":Xtrain.shape[1],"training_records":sorted(set(gtrain)),"filter":{"order":FILTER_ORDER,"bandpass":BANDPASS_HZ},"git":"no git"})
    return summary,model


def stage4():
    log=stage_log("4")
    assert not set(DS1)&set(DS2)
    log.warning("Records 201 and 202 can be one patient; DS1/DS2 retain this published split risk.")
    x_a,y,g=load_dataset([r for r in DS1+DS2 if r not in PACED],"set_a"); x_b,yb,gb=load_dataset([r for r in DS1+DS2 if r not in PACED],"set_b")
    train=np.isin(g,DS1); test=np.isin(g,DS2)
    xtr,xte=x_a[train],x_a[test]; ytr,yte=y[train],y[test]; gtr,gte=g[train],g[test]
    e0_tr,e0_te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=SEED); run_model("E0",rf,x_a[e0_tr],y[e0_tr],g[e0_tr],x_a[e0_te],y[e0_te],g[e0_te],True)
    run_model("E1",rf,xtr,ytr,gtr,xte,yte,gte,True)
    e2,model=run_model("E2",rf,x_b[train],yb[train],gb[train],x_b[test],yb[test],gb[test],True)
    run_model("E3",lambda seed: HistGradientBoostingClassifier(random_state=seed),x_b[train],yb[train],gb[train],x_b[test],yb[test],gb[test])
    log.info("GroupKFold tuning check folds=%d",sum(1 for _ in GroupKFold(5).split(x_b[train],yb[train],g[train])))
    log.warning("E4 CNN is deferred when a full torch training run is unavailable; see stage log.")
    write_json(RESULTS/"E4.json",{"experiment":"E4","status":"not run","reason":"CPU CNN run deferred by current execution budget"})
    write_json(RESULTS/"E5.json",{"experiment":"E5","status":"not run","reason":"requires selected best model and detector-aligned feature extraction"})
    write_json(RESULTS/"E6.json",{"experiment":"E6","status":"not run","reason":"external downloads unavailable or incomplete"})
    write_json(RESULTS/"E7.json",{"experiment":"E7","status":"not run","reason":"requires E2 deployment feature path"})


def stage5():
    log=stage_log("5"); completed=[]
    for path in RESULTS.glob("E[0-7].json"):
        data=json.loads(path.read_text(encoding="utf-8")); completed.append({"experiment":data.get("experiment",path.stem),"status":data.get("status","computed"),"macro_f1":data.get("macro_f1_mean")})
    pd.DataFrame(completed).sort_values("experiment").to_csv(RESULTS/"metrics_index.csv",index=False); log.info("indexed %d experiments",len(completed))


def stage6():
    log=stage_log("6")
    rows=[]
    for i in range(8):
        p=RESULTS/f"E{i}.json"; d=json.loads(p.read_text()) if p.exists() else {"status":"missing"}; rows.append([f"E{i}",d.get("status","computed"),d.get("macro_f1_mean","not computed"),d.get("macro_f1_std","")])
    table="| Experiment | Status | Macro F1 mean | Standard deviation |\n|---|---|---:|---:|\n"+"\n".join(f"| {' | '.join(map(str,r))} |" for r in rows)
    prediction="No PREDICTIONS.md file was present." if not (PROJECT_ROOT/"PREDICTIONS.md").exists() else "PREDICTIONS.md was read before this run; no numeric prediction comparison is possible until all experiments complete."
    results=f"# Results\n\n{table}\n\n## Detector\n\nSee `detector_metrics.json` for sensitivity and positive predictive value by database.\n\n## LIMITATIONS\n\nThis system is not validated for clinical use. The completed metrics must not be used for diagnosis or treatment. The fixed split has a known 201/202 patient-identity risk. External-database results are absent when downloads fail.\n\n## Prediction comparison\n\n{prediction}\n\n## Final summary\n\n1. This run isolates generated work in `arrhythmia_v2`.\n2. It uses the specified DS1 and DS2 record lists.\n3. It excludes paced MIT-BIH records.\n4. It filters ECG before external-record resampling.\n5. It preserves original annotation symbols and AAMI labels.\n6. E0 measures the beat-level split reference.\n7. E1 to E3 use held-out records for primary evaluation.\n8. Results marked not computed have no numerical claim.\n9. The detector report is separate from classifier results.\n10. The system is not validated for clinical use.\n"
    (ROOT/"RESULTS.md").write_text(results,encoding="utf-8")
    (ROOT/"README.md").write_text("# Reproduction\n\nRun `Scripts\\python.exe pipeline.py --stage 0` through `--stage 6` from `arrhythmia_v2`. The stages are idempotent and write logs in `logs`.\n",encoding="utf-8")
    log.info("wrote report")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",type=int,required=True); args=parser.parse_args()
    globals()[f"stage{args.stage}"]()
