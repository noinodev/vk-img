local img = require("img")
local gpu = img.gpu_init()

local input = img.create_from_image("res/gold.jpg")
local width = img.width(input)
local height = img.height(input)
local depth = img.depth(input)
local channels = img.channels(input)
local output = img.create_zero(width//4,height//4,depth,channels)

local brightness = img.gpu_load_program(gpu,"glsl/brightness.comp",8,8,1)
local greyscale = img.gpu_load_program(gpu,"glsl/greyscale.comp",8,8,1)
local downscale = img.gpu_load_program(gpu,"glsl/downscale.comp",8,8,1)

-- greyscale
do
	local input_uniform = 0
	local input_gpu = img.gpu_allocate_image(gpu,input_uniform,width,height,depth,channels)
	local output_uniform = 1
	local output_gpu = img.gpu_allocate_image(gpu,output_uniform,width,height,depth,channels)

	local staging_upload = img.gpu_upload(gpu,input_gpu,input)
	local pass = img.gpu_add_stage(gpu,greyscale,width,height,depth)
	img.gpu_add_stage_data(gpu,pass,input_uniform)
	img.gpu_add_stage_data(gpu,pass,output_uniform)

	local staging_download = img.gpu_download(gpu,output_gpu,input)

	img.gpu_dispatch(gpu)
	img.gpu_reset(gpu)
end

img.write_as_image(input,"greyscale.bmp")

-- brighten
do
	local input_uniform = 0
	local input_gpu = img.gpu_allocate_image(gpu,input_uniform,width,height,depth,channels)
	local output_uniform = 1
	local output_gpu = img.gpu_allocate_image(gpu,output_uniform,width,height,depth,channels)

	local staging_upload = img.gpu_upload(gpu,input_gpu,input)
	local pass = img.gpu_add_stage(gpu,brightness,width,height,depth)
	img.gpu_add_stage_data(gpu,pass,input_uniform)
	img.gpu_add_stage_data(gpu,pass,output_uniform)
	img.gpu_add_stage_data(gpu,pass,.1)

	local staging_download = img.gpu_download(gpu,output_gpu,input)

	img.gpu_dispatch(gpu)
	img.gpu_reset(gpu)
end

img.write_as_image(input,"brightness.bmp")

-- downscale

do
	local input_uniform = 0
	local input_gpu = img.gpu_allocate_image(gpu,input_uniform,width,height,depth,channels)
	local output_uniform = 1
	local output_gpu = img.gpu_allocate_image(gpu,output_uniform,width//4,height//4,depth,channels)

	local staging_upload = img.gpu_upload(gpu,input_gpu,input)
	local pass = img.gpu_add_stage(gpu,downscale,width//4,height//4,depth)
	img.gpu_add_stage_data(gpu,pass,input_uniform)
	img.gpu_add_stage_data(gpu,pass,output_uniform)
	img.gpu_add_stage_data(gpu,pass,.25)

	local staging_download = img.gpu_download(gpu,output_gpu,output)

	img.gpu_dispatch(gpu)
	img.gpu_reset(gpu)
end

img.write_as_image(output,"downscale.bmp")

img.destroy(input)
img.destroy(output)