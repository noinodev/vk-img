local function files_in(dir)
    local t = {}
    local p = io.popen("ls " .. dir)
    for file in p:lines() do
        t[#t+1] = dir .. file
    end
    p:close()
    return t
end

local img = require("img")
local hog = require("hog-svm/hog")
local gpu = img.gpu_init()

for i, file in ipairs(files_in("datasets/inria/pedestrians/")) do
    local image = img.create_from_image(file)
    local encoded = hog.encode(gpu, image)
    local vectors = hog.vectorize(gpu, encoded, 8, 64, 128, 16, 16)
    --hog.export(vectors, file, "vectors/")
    img.write_raw(vectors,"hog-svm/vectors/positive.hog","ab")
    img.destroy(vectors)
    img.destroy(image)
end

math.randomseed(os.time())
for i, file in ipairs(files_in("datasets/inria/no_pedestrians/")) do
    local image = img.create_from_image(file)
    local width = img.width(image)-64
    local height = img.height(image)-128

    local encoded = hog.encode(gpu, image)
    local vectors = hog.vectorize(gpu, encoded, 8, 64, 128, math.random(width),math.random(height))
    img.write_raw(vectors,"hog-svm/vectors/negative.hog","ab")
    img.destroy(vectors)
    img.destroy(image)
end