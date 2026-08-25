Simple Vulkan compute engine for image processing and GP-GPU tasks. Adapted from Vulkan renderer using `vkr.h` utility. Example scripts and shaders for basic image processing, HOG-SVM classification, and a half-working CNN implementation (training has correctness issue and doesn't work properly, but inference works fine using pre-trained weights). Originally designed as CPU-only image processing utility for coursework, slowly built into Vulkan compute GP-GPU runtime as a proof-of-concept.

Building requires Vulkan SDK, cglm, stb-image headers, shaderc and Lua 5.5. Makefile provided, but needs modification to suit your system.

Lua scripting layer for compute interface. Example usage:
```lua
local img = require("img")

local gpu = img.gpu_init()

local input = img.create_from_image("res/gold.jpg")
local width = img.width(input)
local height = img.height(input)
local depth = img.depth(input)
local channels = img.channels(input)

local greyscale = img.gpu_load_program(gpu,"glsl/greyscale.comp",8,8,1)

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

img.destroy(input)
img.destroy(output)
```
