import numpy as np
from PIL import Image
import random

FILE = input("file: ")

with open(FILE, "rb") as f:
    header = np.frombuffer(f.read(16 * 4), dtype=np.uint32)

    count        = int(header[0])
    label_stride = int(header[1])
    image_stride = int(header[2])
    total_stride = int(header[3])

    width    = int(header[4])
    height   = int(header[5])
    channels = int(header[7])

    idx = random.randint(0, count - 1)
    print("reading index:", idx)

    base = 16 * 4
    record_offset = base + idx * total_stride

    f.seek(record_offset)

    # 2 byte label
    coarse = int.from_bytes(f.read(1), "little")
    fine   = int.from_bytes(f.read(1), "little")

    # image
    img_bytes = f.read(image_stride)

    arr = np.frombuffer(img_bytes, dtype=np.uint8)

    if channels == 3:
        arr = arr.reshape((height, width, channels))
    else:
        arr = arr.reshape((height, width))

    arr = np.ascontiguousarray(arr)

    img = Image.fromarray(arr)
    img.show()

    print("coarse:", coarse, "fine:", fine)