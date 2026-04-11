local img = require("img")
local hog = require("hog-svm/hog")

local gpu = img.gpu_init()
print("gpu:",gpu,"ok!")

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
local hog = require("hog")
local gpu = img.gpu_init()

local cellsize = 8
local ww = 64
local wh = 128
local window_width = ww // cellsize
local window_height = wh // cellsize

local count = 0
local max_negatives = 10000 -- cap it so you don't skew class balance

for i, file in ipairs(files_in("datasets/inria/no_pedestrians/")) do
    if count >= max_negatives then break end

    local image = img.create_from_image(file)
    local width = img.width(image)
    local height = img.height(image)
    local encoded = hog.encode(gpu, image)
    local scores = hog.inference(gpu, image, encoded, cellsize, ww, wh)

    local block_width = width // cellsize - 1
    local block_height = height // cellsize - 1
    local total_window_width = block_width - window_width
    local total_window_height = block_height - window_height
    local window_count = total_window_width * total_window_height

    for j = 0, window_count - 1 do
        if count >= max_negatives then break end
        local score = img.get_float(scores, j)
        if score > 0 then
            local wx = (j % total_window_width) * cellsize
            local wy = (j // total_window_width) * cellsize
            local encoded2 = hog.encode(gpu, image)
            local vec = hog.vectorize(gpu, encoded2, cellsize, ww, wh, wx, wy)
            local confidence = img.create_fill(1,1,1,1,score) -- malloc a single float YES SIR!!
            img.write_raw(confidence, "hog-svm/vectors/negative_mined.hog","ab")
            img.write_raw(vec, "hog-svm/vectors/negative_mined.hog", "ab")
            img.destroy(vec)
            img.destroy(confidence)
            count = count + 1
        end
    end

    img.destroy(scores)
    img.destroy(image)
    print("file", i, "hard negatives so far:", count)
end

print("total hard negatives collected:", count)