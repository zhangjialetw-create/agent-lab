from pathlib import Path

# __file__ 是这个 Python 文件的位置；不受终端当前目录影响。
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NOTES = ROOT / "notes"
RUNS = ROOT / "runs"

