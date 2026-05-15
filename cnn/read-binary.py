import numpy as np
from PIL import Image
import random

FILE = input("file: ")

with open(FILE, "rb") as f:
    header = np.frombuffer(f.read(16 * 4), dtype=np.uint32)

    count        = header[0]
    label_stride = header[1]
    image_stride = header[2]
    total_stride = header[3]

    width    = header[4]
    height   = header[5]
    channels = header[7]

    idx = random.randint(0, count - 1)
    print("reading index:", idx)

    # --- compute offset ---
    base = 16 * 4  # header size in bytes
    record_offset = base + idx * total_stride

    f.seek(record_offset)

    coarse = int.from_bytes(f.read(1), "little")
    fine   = int.from_bytes(f.read(1), "little")

    img_bytes = f.read(image_stride)

    arr = np.frombuffer(img_bytes, dtype=np.uint8)

    if channels == 3:
        arr = arr.reshape((height, width, channels))
    else:
        arr = arr.reshape((height, width))

    img = Image.fromarray(arr)
    img.show()

    print("coarse:", coarse, "fine:", fine)