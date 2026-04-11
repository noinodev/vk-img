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
    local sobel_uniform = 6
    local sobel_gpu = img.gpu_allocate_image(gpu,sobel_uniform,width,height,depth,channels)

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

        local program = img.gpu_load_program(gpu,"glsl/hog/sobel.comp",8,8,1)
        local pass = img.gpu_add_stage(gpu,program,width,height,depth)

        img.gpu_add_stage_data(gpu,pass,output_uniform)
        img.gpu_add_stage_data(gpu,pass,sobel_uniform)
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
        local program = img.gpu_load_program(gpu,"glsl/hog/hog.comp",1,1,1)
        local pass = img.gpu_add_stage(gpu,program,cell_width,cell_height,1)

        img.gpu_add_stage_data(gpu,pass,sobel_uniform)
        img.gpu_add_stage_data(gpu,pass,histogram_uniform)
        img.gpu_add_stage_data(gpu,pass,cellsize)
    end
    
    local block_width = cell_width - 1
    local block_height = cell_height -1
    local block_depth = bins * 4
    local block_uniform = 3
    local gpu_image_blocks = img.gpu_allocate_image(gpu,block_uniform,block_width,block_height,block_depth,channels)

    do
        local program = img.gpu_load_program(gpu,"glsl/hog/normalize.comp",1,1,1)
        local pass = img.gpu_add_stage(gpu,program,block_width,block_height,1)

        img.gpu_add_stage_data(gpu,pass,histogram_uniform)
        img.gpu_add_stage_data(gpu,pass,block_uniform)
    end

    return block_uniform
end

function API.vectorize(gpu,blocks_uniform,cellsize,ww,wh,wx,wy) -- linearize a window of an encoded image

    local window_width = ww // cellsize
    local window_height = wh // cellsize
    local window_left = wx // cellsize
    local window_top = wy // cellsize

    local feature_length = window_width * window_height * 36
    --print(feature_length)
    local sizeof_float = 4
    local vector_uniform = 0
    local gpu_buffer_vectors = img.gpu_allocate_buffer(gpu,vector_uniform,feature_length*sizeof_float)

    do
        local program = img.gpu_load_program(gpu,"glsl/hog/extract.comp",1,1,1) -- workgroup size divisor for pixel images; not used here
        local pass = img.gpu_add_stage(gpu,program,window_width,window_height,1) -- dispatch size, but this is rounded to the nearest workgroup? should that change?

        img.gpu_add_stage_data(gpu,pass,blocks_uniform) -- i passed the wrong variable here before -> was bigger than 1 float -> pc.vectors was garbage -> driver crash
        img.gpu_add_stage_data(gpu,pass,vector_uniform)
        img.gpu_add_stage_data(gpu,pass,window_left)
        img.gpu_add_stage_data(gpu,pass,window_top)
        img.gpu_add_stage_data(gpu,pass,window_width)
    end

    local final_hog_output = img.create_zero(feature_length,1,1,1)
    local download = img.gpu_download(gpu,gpu_buffer_vectors,final_hog_output)

    img.gpu_dispatch(gpu)
    img.gpu_reset(gpu)

    return final_hog_output
end

function API.inference(gpu,input,blocks_uniform,cellsize,ww,wh)
    local width = img.width(input)
    local height = img.height(input)
    local depth = img.depth(input)
    local channels = img.channels(input)

    local window_width = ww // cellsize
    local window_height = wh // cellsize

    local block_width = img.width(input)//cellsize - 1
    local block_height = img.height(input)//cellsize - 1
    local total_window_width = block_width-window_width
    local total_window_height = block_height-window_height
    local window_count = total_window_width*total_window_height
    local blocks_per_window = window_width*window_height

    local feature_length = window_width * window_height * 36
    local sizeof_float = 4

    local weights = img.create_zero(feature_length,1,1,1)
    img.read_raw(weights,"hog-svm/svm_weights.bin")
    --img.write_as_image(weights,"weights_debug.bmp")
    local weights_readback = img.create_zero(feature_length,1,1,1)

    local f = io.open("hog-svm/svm_bias.bin", "rb")
    local bias = string.unpack("f", f:read(4))
    f:close()
    print("bias",bias)

    local weights_uniform = 2
    local scores_uniform = 3
    local overlay_uniform = 9
    local weights_gpu = img.gpu_allocate_buffer(gpu, weights_uniform, feature_length * sizeof_float)
    local scores_gpu = img.gpu_allocate_buffer(gpu, scores_uniform, window_count * sizeof_float)
    local overlay_gpu = img.gpu_allocate_image(gpu,overlay_uniform,width,height,depth,channels)
    img.gpu_upload(gpu, weights_gpu, weights)
    img.gpu_upload(gpu,overlay_gpu,input)

    do
        local program = img.gpu_load_program(gpu,"glsl/hog/inference.comp",1,1,1) -- workgroup size divisor for pixel images; not used here
        local pass = img.gpu_add_stage(gpu,program,1,1,window_count) -- dispatch size, but this is rounded to the nearest workgroup? should that change?

        img.gpu_add_stage_data(gpu,pass,blocks_uniform) -- i passed the wrong variable here before -> was bigger than 1 float -> pc.vectors was garbage -> driver crash
        img.gpu_add_stage_data(gpu,pass,weights_uniform)
        img.gpu_add_stage_data(gpu,pass,scores_uniform)
        img.gpu_add_stage_data(gpu,pass,overlay_uniform)
        img.gpu_add_stage_data(gpu,pass,bias)
        img.gpu_add_stage_data(gpu,pass,window_width)
        img.gpu_add_stage_data(gpu,pass,window_height)
        img.gpu_add_stage_data(gpu,pass,total_window_width)
    end

    local scores = img.create_zero(window_count,1,1,1)
    local download = img.gpu_download(gpu,scores_gpu,scores)
    img.gpu_download(gpu,weights_gpu,weights_readback)
    img.gpu_download(gpu,overlay_gpu,input)

    img.gpu_dispatch(gpu)
    img.gpu_reset(gpu)

    --img.write_as_image(weights_readback,"weights_r.bmp")

    --img.write_as_image(scores,"scores.bmp")
    img.write_as_image(input,"hog-svm/input.bmp")

    return scores
end

return API