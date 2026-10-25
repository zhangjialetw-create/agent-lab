import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
path = root / "data" / "sessions.json"
text = path.read_text(encoding="utf-8-sig")
records = json.loads(text)
print("文件：", path.name)
print("记录数：", len(records))
print("第一条主题：", records[0]["topic"])