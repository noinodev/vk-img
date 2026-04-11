import numpy as np

weights = np.fromfile("svm_weights.bin", dtype=np.float32)
bias = np.fromfile("svm_bias.bin", dtype=np.float32)[0]
vec = np.fromfile("vectors/test.hog", dtype=np.float32)

score = np.dot(vec, weights) + bias
print("score:", score)
print("prediction:", "positive" if score > 0 else "negative")