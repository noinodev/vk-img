local img = require("img")
local hog = require("hog")

local gpu = img.gpu_init()
print("gpu:",gpu,"ok!")

local file = "data_ped/pedestrians/crop_000010a.png"
local image = img.create_from_image(file)
local encoded = hog.encode(gpu,image)
local vectors = hog.vectorize(gpu,image,encoded,8,0,0,64,128)
hog.export(vectors,file,"vectors/")
img.destroy(vectors)
img.destroy(image)