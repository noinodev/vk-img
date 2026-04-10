local img = require("img")

local API = {}

function API.encode(gpu,input) -- encode an image
    local width = img.width(input)
    local height = img.height(input)
    local depth = img.depth(input)
    local channels = img.channels(input)

    local input_uniform = 0
    local input_gpu = img.gpu_allocate_image(gpu,input_uniform,width,height,depth,channels)
    local output_uniform = 1
    local output_gpu = img.gpu_allocate_image(gpu,output_uniform,width,height,depth,channels)

    local staging_upload = img.gpu_upload(gpu,input_gpu,input)

    do
        local program = img.gpu_load_program(gpu,"glsl/greyscale.comp",8,8,1)
        local pass = img.gpu_add_stage(gpu,program,width,height,depth)
        img.gpu_add_stage_data(gpu,pass,input_uniform)
        img.gpu_add_stage_data(gpu,pass,output_uniform)
    end

    do
        local sobel_kernel_x = {-1,0,1,-2,0,2,-1,0,1}
        local sobel_kernel_y = {-1,-2,-1,0,0,0,1,2,1}

        local program = img.gpu_load_program(gpu,"glsl/sobel.comp",8,8,1)
        local pass = img.gpu_add_stage(gpu,program,width,height,depth)

        img.gpu_add_stage_data(gpu,pass,output_uniform)
        img.gpu_add_stage_data(gpu,pass,input_uniform)
        img.gpu_add_stage_data(gpu,pass,sobel_kernel_x)
        img.gpu_add_stage_data(gpu,pass,sobel_kernel_y)
    end

    local cellsize = 8
    local bins = 9
    local cell_width = width // cellsize
    local cell_height = height // cellsize

    local histogram_uniform = 2
    local gpu_image_histogram = img.gpu_allocate_image(gpu,histogram_uniform,cell_width,cell_height,bins,channels)

    do
        local program = img.gpu_load_program(gpu,"glsl/hog.comp",1,1,1)
        local pass = img.gpu_add_stage(gpu,program,cell_width,cell_height,1)

        img.gpu_add_stage_data(gpu,pass,input_uniform)
        img.gpu_add_stage_data(gpu,pass,histogram_uniform)
        img.gpu_add_stage_data(gpu,pass,cellsize)
    end

    local block_width = cell_width - 1
    local block_height = cell_height -1
    local block_depth = bins * 4
    local block_uniform = 3
    local gpu_image_blocks = img.gpu_allocate_image(gpu,block_uniform,block_width,block_height,block_depth,channels)

    do
        local program = img.gpu_load_program(gpu,"glsl/normalize.comp",1,1,1)
        local pass = img.gpu_add_stage(gpu,program,block_width,block_height,1)

        img.gpu_add_stage_data(gpu,pass,histogram_uniform)
        img.gpu_add_stage_data(gpu,pass,block_uniform)
    end

    return block_uniform
end

function API.vectorize(gpu,image,hog,cellsize,wx,wy,ww,wh) -- linearize a window of an encoded image

    local width = img.width(image)
    local height = img.height(image)
    local window_width = ww / cellsize
    local window_height = wh / cellsize
    local window_left = ((wx*.5+.5)*width) // cellsize
    local window_top = ((wy*.5+.5)*height) // cellsize
    local feature_length = window_width * window_height * 36
    local sizeof_float = 4
    local vector_uniform = 0
    local gpu_buffer_vectors = img.gpu_allocate_buffer(gpu,vector_uniform,feature_length*sizeof_float)

    do
        local program = img.gpu_load_program(gpu,"glsl/extract.comp",36,1,1)
        local pass = img.gpu_add_stage(gpu,program,window_width,window_height,1)

        img.gpu_add_stage_data(gpu,pass,hog)
        img.gpu_add_stage_data(gpu,pass,vector_uniform)
        img.gpu_add_stage_data(gpu,pass,window_left)
        img.gpu_add_stage_data(gpu,pass,window_top)
        img.gpu_add_stage_data(gpu,pass,9)
        img.gpu_add_stage_data(gpu,pass,window_width)
    end

    local output = img.create_zero(feature_length,1,1,1)
    local download = img.gpu_download(gpu,gpu_buffer_vectors,output)

    img.gpu_dispatch(gpu)
    img.gpu_reset(gpu)

    return output
end

function API.export(buffer,file,location)
    local filename = file:match("([^/]+)$")
    img.write_as_binary(buffer,location..filename..".hog")
end

function API.inference(gpu,image,hog,cellsize,ww,wh)

end

return API

--local gpu = img.gpu_init()

--hog(gpu,"data_ped/pedestrians/crop_000010a.png",0,0,64,128)