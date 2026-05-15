local img = require("img")

local API = {}

function API.convolve(gpu,input_width,input_height,input_depth,kernels)


end

function API.maxpool(gpu,input_width,input_height)

end

function API.softmax(gpu)

end

function API.flatten(gpu,image,image_uniform)

	local width = img.width(image)
	local height = img.height(image)
	local depth = img.depth(image)
	local channels = img.channels(image)
	local sizeof_float = 4
	local flat_length = width*height*depth*channels*sizeof(float)

	--local uniform_input = 0
	--local gpu_buffer_input = img.gpu_allocate_buffer(gpu,uniform_input,flat_length)
	local uniform_output = 0
	local gpu_image_output = img.gpu_allocate_image(gpu,uniform_output,width,height,depth*channels,1)

	do
        local program = img.gpu_load_program(gpu,"glsl/cnn/flatten.comp",8,8,1)
        local pass = img.gpu_add_stage(gpu,program,width,height,1)

        img.gpu_add_stage_data(gpu,pass,image_uniform)
        img.gpu_add_stage_data(gpu,pass,uniform_output)

        img.gpu_add_stage_data(gpu,pass,width)
        img.gpu_add_stage_data(gpu,pass,height)
    end

end


return API