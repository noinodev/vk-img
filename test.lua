local img = require("img")
local gpu = img.gpu_init()

local input = img.create_from_image("res/gold.jpg")
local width = img.width(input)
local height = img.height(input)
local depth = img.depth(input)
local channels = img.channels(input)
local output = img.create_zero(width/2,height/2,depth,channels)

local input_uniform = 0
local input_gpu = img.gpu_allocate_image(gpu,input_uniform,width,height,depth,channels)
local output_uniform = 1
local output_gpu = img.gpu_allocate_image(gpu,output_uniform,width/2,height/2,depth,channels)

img.gpu_upload(gpu,input_gpu,input)
img.gpu_download(gpu,output_gpu,output)

local downscale = img.gpu_load_program(gpu,"glsl/downscale.comp",8,8,1)
local pass = img.gpu_add_stage(gpu,downscale,width/2,height/2,depth)
img.gpu_add_stage_data(gpu,pass,input_uniform)
img.gpu_add_stage_data(gpu,pass,output_uniform)
img.gpu_add_stage_data(gpu,pass,.5)

img.gpu_dispatch(gpu)

img.write_as_image(output,"downscale.bmp")

img.destroy(input)
img.destroy(output)