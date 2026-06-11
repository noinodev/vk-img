import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import io

table = pq.read_table(input("input parquet: "))
num_rows = len(table)
shuffled_indices = np.random.permutation(num_rows)
shuffled_table = table.take(shuffled_indices)

coarse = shuffled_table["label"].to_numpy() # mnist format
fine = coarse
images = shuffled_table["image"]

#coarse = shuffled_table["coarse"].to_numpy() # cifar format
#fine = coshuffled_table["fine"].to_numpy()arse
#images = shuffled_table["img"]

count = len(coarse)

sample = Image.open(io.BytesIO(images[0].as_py()["bytes"])).convert("RGB")
sample = np.array(sample, dtype=np.uint8)

height, width, channels = sample.shape
image_stride = sample.size
label_stride = 2

header = np.zeros(16, dtype=np.uint32)
header[0] = count
header[1] = label_stride
header[2] = image_stride
header[3] = label_stride + image_stride
header[4] = width
header[5] = height
header[6] = 1
header[7] = channels

out_path = input("file output as: ")

with open(out_path, "wb") as f:
    f.write(header.tobytes())

    for i in range(count):
        f.write(np.uint8(coarse[i]).tobytes())
        f.write(np.uint8(fine[i]).tobytes())
    
    for i in range(count):
        img = Image.open(io.BytesIO(images[i].as_py()["bytes"])).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)

        arr = np.ascontiguousarray(np.array(img, dtype=np.uint8))
        f.write(arr.ravel().tobytes())