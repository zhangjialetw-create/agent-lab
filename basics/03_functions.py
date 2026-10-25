def summarize(records):
    count = len(records)
    total = 0
    for record in records:
        total = total + record["minutes"]
    if count == 0:
        average = 0.0
    else:
        average = total / count
    return {"count":count, "total": total, "average": average}


if __name__ == "__main__":
    result = summarize([{"topic":"python", "minutes":25}])
    print(result)
    print(summarize([]))