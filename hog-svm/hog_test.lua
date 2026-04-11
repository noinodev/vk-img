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
print("gpu:",gpu,"ok!")

for i, file in ipairs(files_in("datasets/inria/pedestrians/")) do
    local image = img.create_from_image(file)
	local encoded = hog.encode(gpu,image)
	local scores = hog.inference(gpu,image,encoded,8,64,128,0,0)
    img.destroy(scores)
    img.destroy(image)
end

for i, file in ipairs(files_in("datasets/inria/no_pedestrians/")) do
    local image = img.create_from_image(file)
	local encoded = hog.encode(gpu,image)
	local scores = hog.inference(gpu,image,encoded,8,64,128,0,0)
    img.destroy(scores)
    img.destroy(image)
end

--local file = "data_ped/pedestrians/crop_000010a.png"
--local file = "data_ped/no_pedestrians/00000002a.png"
--local file = "res/gold.jpg"
--local file = "res/camera.jpeg"
--img.write_raw(scores,"SCOReS!.bin","wb")
--hog.export(image_output,file,"vectors/")
