local img = require("img")
local cnn = require("cnn/cnn")

local gpu = img.gpu_init()

local i = 0
local file = "datasets/cifar100/train.bin"
local page_size = 1024*1024

--local greyscale = img.gpu_load_program(gpu,"glsl/greyscale.comp",8,8,1)

local atlas = img.create_atlas(file,page_size,i)
local image_first = img.create_from_atlas(atlas,1) -- first image in atlas
local width,height,depth,channels = img.dim(image_first);
print(img.dim(image_first))

local image_staging = img.create(width,height,depth,4,0)
img.remap(image_first,image_staging)

local input_gpu_uniform = 0
local input_gpu = img.gpu_allocate_image(gpu,input_gpu_uniform,width,height,depth,4)
local upload = img.gpu_upload(gpu,input_gpu,image_staging)

local output_gpu_uniform = 1
local output_gpu = img.gpu_allocate_image(gpu,output_gpu_uniform,100,1,1,1)
local image_out = img.create(100,1,1,1,0)
local download = img.gpu_download(gpu,output_gpu,image_out)

cnn.load(gpu)
cnn.to_3d(gpu,image_staging,input_gpu,2)
--[[cnn.convolve(gpu,
cnn.maxpool
cnn.con]]



--img.write_as_image(image_first,"atlas.bmp")

-- define CNN layout, allocate all buffers etc
-- draw the rest of the owl

--cnn.flatten(gpu,image_first,input_gpu_uniform) -- convert w,h,d,c to tensor layout w,h,d*c (channels become image layers/depth)
-- convolve, pool, convolve, pool ... softmax
-- repeat backwards

while atlas do
	local image_count = img.atlas_size(atlas)
	for j=0, image_count-1 do
		local image = img.create_from_atlas(atlas,j)
		img.remap(image,image_staging)

		print(img.meta(image_staging))

		img.gpu_map_host(gpu,upload,image_staging)
		img.gpu_dispatch(gpu) -- copy host->device, feed forward, backprop, weights persist

		img.remap(image_staging,image_out)
		img.write_as_image(image_out,"atlas.bmp")

		io.read()
		
	end

	img.destroy_atlas(atlas)
	i = i+1
	atlas = img.create_atlas(file,page_size,i)
end


-- download weights