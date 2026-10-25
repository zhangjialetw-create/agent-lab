records = [
    {"topic": "python", "minutes": 25},
    {"topic": "终端", "minutes": 35},
    {"topic": "复习", "minutes": 15},
]
total = 0 
for record in records:
    total = total + record["minutes"]
    print(record["topic"], record["minutes"])
print("总分钟：", total)