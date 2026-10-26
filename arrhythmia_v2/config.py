from pathlib import Path

SEED = 42
FS_TARGET = 360
BANDPASS_HZ = (0.5, 40.0)
FILTER_ORDER = 4
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA = ROOT / "data"
PROCESSED = ROOT / "processed"
FEATURES = ROOT / "features"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"
LOGS = ROOT / "logs"
MITDB = PROJECT_ROOT
PACED = {"102", "104", "107", "217"}
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
AAMI = {"N":"N", "L":"N", "R":"N", "e":"N", "j":"N", "A":"S", "a":"S", "J":"S", "S":"S", "V":"V", "E":"V", "F":"F", "/":"Q", "f":"Q", "Q":"Q"}
CLASSES = ["N", "S", "V", "F", "Q"]
