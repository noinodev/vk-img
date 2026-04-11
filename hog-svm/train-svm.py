import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import struct

feature_length = 4608

def load_vectors(path):
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(-1, feature_length)

pos = load_vectors("vectors/positive.hog")
neg = load_vectors("vectors/negative.hog")

X = np.vstack([pos, neg])
y = np.array([1] * len(pos) + [-1] * len(neg))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

print(f"accuracy: {accuracy_score(y_test, clf.predict(X_test)):.3f}")

print("positive samples:", len(pos))
print("negative samples:", len(neg))
print("positive min/max:", pos.min(), pos.max())
print("negative min/max:", neg.min(), neg.max())
print("any nan in pos:", np.any(np.isnan(pos)))
print("any nan in neg:", np.any(np.isnan(neg)))
print("first positive vector:", pos[0,:10])
print("first negative vector:", neg[0,:10])

zero_rows = np.all(neg == 0, axis=1)
print("zero vectors in negative:", zero_rows.sum())

unique_neg = np.unique(neg, axis=0)
print("unique negative vectors:", len(unique_neg))

# save weights as flat binary for C
w = clf.coef_[0].astype(np.float32)
b = np.array([clf.intercept_[0]], dtype=np.float32)
w.tofile("svm_weights.bin")
b.tofile("svm_bias.bin")

print(f"support vectors: {len(clf.support_vectors_)}")
print(f"weights saved: {len(w)} floats")

print("bias:", clf.intercept_[0])
#print("first 100 weights:", clf.coef_[0][:100])