local img = require("img")
local cnn = require("cnn/cnn")

local gpu = img.gpu_init()

local i = 0
local file = "cnn/cifar100.bin"
local page_size = 1024*1024

local atlas = img.create_atlas(file,page_size,i)
local image_first = img.create_from_atlas(atlas,0) -- first image in atlas
local width = img.width(image_first)
local height = img.height(image_first)
local depth = img.depth(image_first)
local channels = img.channels(image_first)

local input_gpu_uniform = 0
local input_gpu = img.gpu_allocate_image(gpu,input_gpu_uniform,width,height,depth,channels)
local upload = img.gpu_upload(gpu,input_gpu,image_first)

-- define CNN layout, allocate all buffers etc
-- draw the rest of the owl

cnn.flatten(gpu,image_first,input_gpu_uniform) -- convert w,h,d,c to tensor layout w,h,d*c (channels become image layers/depth)
-- convolve, pool, convolve, pool ... softmax
-- repeat backwards

while atlas do

	local image_count = img.atlas_size(atlas)
	for j=0, image_count-1 do
		local image = img.create_from_atlas(atlas,j)

		img.gpu_map_host(gpu,upload,image)
		img.gpu_dispatch() -- copy host->device, feed forward, backprop, weights persist
	end

	img.destroy_atlas(atlas)
	i = i+1
	atlas = img.create_atlas(file,page_size,i)
end

-- download weights