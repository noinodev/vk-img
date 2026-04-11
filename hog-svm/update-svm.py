import numpy as np

feature_length = 4608
records = []

with open("vectors/negative_mined.hog", "rb") as f:
    while True:
        score_bytes = f.read(4)
        if not score_bytes or len(score_bytes) < 4:
            break
        score = np.frombuffer(score_bytes, dtype=np.float32)[0]
        vec_bytes = f.read(feature_length * 4)
        if len(vec_bytes) < feature_length * 4:
            break
        vec = np.frombuffer(vec_bytes, dtype=np.float32)
        records.append((score, vec))

print(f"total false positives collected: {len(records)}")

# sort by score descending, take top 100
records.sort(key=lambda x: x[0], reverse=True)
top = records[:100]

print(f"top score: {top[0][0]:.4f}")
print(f"100th score: {top[-1][0]:.4f}")

# append to negative.hog
with open("vectors/negative.hog", "ab") as f:
    for score, vec in top:
        vec.tofile(f)

print("done, appended 100 hard negatives to negative.hog")