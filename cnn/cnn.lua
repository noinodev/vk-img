local img = require("img")

setmetatable(_G, {
    __index = function(t, k)
        error("undefined variable: " .. tostring(k), 2)
    end
})

local API = {}
local buffers = {}
local program_to3d, program_to2d, program_conv, program_pool, program_linear, program_softmax, program_classify
local program_lossgrad, program_conv_backward_input, progam_conv_backward_weights, program_conv_backward_bias
local program_pool_backward, program_linear_backward_input, program_linear_backward_weights, program_linear_backward_bias
local program_adam

--[[
	load all GLSL programs. program will crash if you dont call this first
]]
function API.load(gpu)
	program_to3d = img.gpu_load_program(gpu,"glsl/cnn/to3d.comp",8,8,1)
	program_to2d = img.gpu_load_program(gpu,"glsl/cnn/to2d.comp",8,8,1)

	program_conv = img.gpu_load_program(gpu,"glsl/cnn/forward/convolve.comp",8,8,4)
	program_pool = img.gpu_load_program(gpu,"glsl/cnn/forward/pool.comp",8,8,1)
	program_linear = img.gpu_load_program(gpu,"glsl/cnn/forward/linear.comp",8,1,1)
	program_softmax = img.gpu_load_program(gpu,"glsl/cnn/forward/softmax.comp",8,1,1)

	program_lossgrad = img.gpu_load_program(gpu,"glsl/cnn/backward/lossgrad.comp",8,8,1)

	program_conv_backward_input = img.gpu_load_program(gpu,"glsl/cnn/backward/convolve_backward_input.comp",1,1,1)
	program_conv_backward_weights = img.gpu_load_program(gpu,"glsl/cnn/backward/convolve_backward_weights.comp",1,1,1)
	program_conv_backward_bias = img.gpu_load_program(gpu,"glsl/cnn/backward/convolve_backward_bias.comp",1,1,1)

	program_pool_backward = img.gpu_load_program(gpu,"glsl/cnn/backward/pool_backward.comp",8,8,1)

	program_linear_backward_input = img.gpu_load_program(gpu,"glsl/cnn/backward/linear_backward_input.comp",8,1,1)
	program_linear_backward_weights = img.gpu_load_program(gpu,"glsl/cnn/backward/linear_backward_weights.comp",8,8,1)
	program_linear_backward_bias = img.gpu_load_program(gpu,"glsl/cnn/backward/linear_backward_bias.comp",64,1,1)

	program_adam = img.gpu_load_program(gpu,"glsl/cnn/backward/adam.comp",256,1,1)
end

--[[
	convolve the input tensor, from the starting image. increase depth sometimes. magic 64x2^n
]]
function API.conv(gpu,in_act,out_act,weights,bias,w,h,ch_in,ch_out)
	print("conv forward")
	local pass = img.gpu_add_stage(gpu,program_conv,w,h,ch_out)

	img.gpu_push_uint(gpu, pass, in_act)
    img.gpu_push_uint(gpu, pass, out_act)
    img.gpu_push_uint(gpu, pass, weights)
    img.gpu_push_uint(gpu, pass, bias)
    img.gpu_push_uint(gpu, pass, w)
    img.gpu_push_uint(gpu, pass, h)
    img.gpu_push_uint(gpu, pass, ch_in)
    img.gpu_push_uint(gpu, pass, ch_out)
	img.gpu_push_uint(gpu, pass, 1) -- relu on

	return pass
end

--[[
	max pooling. reduces width and height dimension by taking max. saves index of max for training. 
	might make a separate program to make this step faster? though a 2x2 downsample is not expensive
]]
function API.pool(gpu,in_act,out_act,indices,w,h,ch)
	print("pool forward")
	local pass = img.gpu_add_stage(gpu,program_pool,w//2,h//2,ch)

	img.gpu_push_uint(gpu, pass, in_act)
    img.gpu_push_uint(gpu, pass, out_act)
    img.gpu_push_uint(gpu, pass, indices)
    img.gpu_push_uint(gpu, pass, w)
    img.gpu_push_uint(gpu, pass, h)
    img.gpu_push_uint(gpu, pass, ch)

	return pass
end

--[[
	linear feedforward, basically a normal neural network pass. relu toggles sometimes. 
	also has an edge case from when a texture changes to a buffer for first fully connected layer.
]]
function API.linear(gpu,in_act,out_act,weights,bias,in_dim,out_dim,sc_dim,relu,prev)
	print("linear forward")
	local pass = img.gpu_add_stage(gpu,program_linear,out_dim,1,1)

	img.gpu_push_uint(gpu, pass, in_act)
    img.gpu_push_uint(gpu, pass, out_act)
    img.gpu_push_uint(gpu, pass, weights)
    img.gpu_push_uint(gpu, pass, bias)
    img.gpu_push_uint(gpu, pass, in_dim)
    img.gpu_push_uint(gpu, pass, out_dim)
	img.gpu_push_uint(gpu, pass, sc_dim)
    img.gpu_push_uint(gpu, pass, relu and 1 or 0)
	img.gpu_push_uint(gpu, pass, prev and 1 or 0) -- prev activation is / is not texture

	return pass
end

--[[
	softmax function. decide which label is most likely
]]
function API.softmax(gpu,in_act,out_act,dim)
	print("softmax forward")
	local pass = img.gpu_add_stage(gpu,program_softmax,dim,1,1)

	img.gpu_push_uint(gpu, pass, in_act)
	img.gpu_push_uint(gpu, pass, out_act)
	img.gpu_push_uint(gpu, pass, dim)

	return pass
end

--[[
	unused. convert volume texture (depth=3) back to rgb texture
]]
function API.to_2d(gpu,width,height,input_uniform,output_uniform) -- convert 3d back to rgba8 for testing
	print("tensor to image")
	local pass = img.gpu_add_stage(gpu,program_to2d,width,height,1)

	img.gpu_push_uint(gpu,pass,input_uniform)
	img.gpu_push_uint(gpu,pass,output_uniform)

	img.gpu_push_uint(gpu,pass,width)
	img.gpu_push_uint(gpu,pass,height)

	return pass
end

--[[
	convert 2D rgba8 texture into 3D r32 texture / tensor
]]
function API.to_3d(gpu,width,height,channels,input_uniform,output_uniform)
	print("image to tensor")
	local pass = img.gpu_add_stage(gpu,program_to3d,width,height,1)

	img.gpu_push_uint(gpu,pass,input_uniform)
	img.gpu_push_uint(gpu,pass,output_uniform)

	img.gpu_push_uint(gpu,pass,width)
	img.gpu_push_uint(gpu,pass,height)

	--img.gpu_add_stage_data(gpu, pass, N)

	return pass
end


--[[
	construct forward pass: 
	 - all gpu object allocations
	 - all command buffer recordings
	 - input binding is 3D volume image
	 - first binding is first usable binding, as this function consumes many indices
]]
function API.build(gpu, definition, input_binding, first_binding)
	local binding = first_binding
	local function next_binding()
		local b = binding
		binding = binding + 1
		return b
	end

    local net = {layers={}, input = input_binding}

	for i,L in ipairs(definition) do
		local layer = { def = L }
		local prev_act = i == 1 and net.input or net.layers[i-1].act


		if L.type == "conv" then
			layer.act = next_binding()
			layer.weights = next_binding()
			layer.bias = next_binding()

			layer.gpu_a = img.gpu_allocate_image(gpu, layer.act, L.w, L.h, L.ch_out, 1)
			layer.gpu_w = img.gpu_allocate_buffer(gpu, layer.weights, L.ch_out * L.ch_in * 9)
			layer.gpu_b = img.gpu_allocate_buffer(gpu, layer.bias, L.ch_out)
			layer.size_w = L.ch_out * L.ch_in * 9
			layer.size_b = L.ch_out

			layer.program = API.conv(gpu, prev_act, layer.act, layer.weights, layer.bias, L.w, L.h, L.ch_in, L.ch_out)
		elseif L.type == "pool" then
			layer.act = next_binding()
			layer.indices = next_binding()

			layer.gpu_a = img.gpu_allocate_image(gpu, layer.act, L.w//2, L.h//2, L.ch, 1)
			layer.gpu_i = img.gpu_allocate_buffer(gpu, layer.indices, (L.w//2) * (L.h//2) * L.ch)

			layer.program = API.pool(gpu, prev_act, layer.act, layer.indices, L.w, L.h, L.ch)
		elseif L.type == "linear" then
			layer.act     = next_binding()
			layer.weights = next_binding()
			layer.bias    = next_binding()

			layer.gpu_a = img.gpu_allocate_buffer(gpu, layer.act, L.out_dim)
			layer.gpu_w = img.gpu_allocate_buffer(gpu, layer.weights, L.out_dim * L.in_dim)
			layer.gpu_b = img.gpu_allocate_buffer(gpu, layer.bias, L.out_dim)
			layer.size_w = L.out_dim * L.in_dim
			layer.size_b = L.out_dim

			local dim = 1
			if net.layers[i-1].type == "pool" then dim = math.sqrt(L.in_dim / net.layers[i-1].def.ch) | 0 end
			print("fuck you",dim)

			layer.program = API.linear(gpu, prev_act, layer.act, layer.weights, layer.bias, L.in_dim, L.out_dim, dim,  L.relu, L.prev)

		elseif L.type == "softmax" then
			layer.act = next_binding()

			layer.gpu_a = img.gpu_allocate_buffer(gpu, layer.act, L.dim)

			layer.program = API.softmax(gpu, prev_act, layer.act, L.dim)
		end

		net.layers[i] = layer
	end

	net.next_free_binding = binding
	return net
end

--[[
	subtract 1 from correct label lmao
]]
function API.loss_grad(gpu,act,grad_act, dim)
	print("loss grad")
	local pass = img.gpu_add_stage(gpu,program_lossgrad,dim,1,1)

	img.gpu_push_uint(gpu, pass, act)
	img.gpu_push_uint(gpu, pass, grad_act)
	img.gpu_push_uint(gpu, pass, 12345)
	img.gpu_push_uint(gpu, pass, dim)

	return pass
end

--[[
	activation gradient. reverses kernel
]]
function API.conv_backward_input(gpu, next_grad_act, grad_act, weights, act, w, h, ch_in, ch_out)
    local pass = img.gpu_add_stage(gpu, program_conv_backward_input, w, h, ch_in)

    img.gpu_push_uint(gpu, pass, next_grad_act)
    img.gpu_push_uint(gpu, pass, grad_act)
    img.gpu_push_uint(gpu, pass, weights)
    img.gpu_push_uint(gpu, pass, act)
    img.gpu_push_uint(gpu, pass, ch_in  | 0)
    img.gpu_push_uint(gpu, pass, ch_out | 0)
    img.gpu_push_uint(gpu, pass, h      | 0)  -- in_h
    img.gpu_push_uint(gpu, pass, w      | 0)  -- in_w
    img.gpu_push_uint(gpu, pass, h      | 0)  -- out_h (same for same-pad stride=1)
    img.gpu_push_uint(gpu, pass, w      | 0)  -- out_w
    img.gpu_push_uint(gpu, pass, 3      | 0)  -- kH
    img.gpu_push_uint(gpu, pass, 3      | 0)  -- kW
    img.gpu_push_uint(gpu, pass, 1      | 0)  -- pad
    img.gpu_push_uint(gpu, pass, 1      | 0)  -- relu always on for conv
end

--[[
	updates weights based on gradient
]]
function API.conv_backward_weights(gpu, prev_act, next_grad_act, grad_weights, w, h, ch_in, ch_out)
	print("conv back weight")
    local pass = img.gpu_add_stage(gpu, program_conv_backward_weights, ch_out, ch_in, 1)

    img.gpu_push_uint(gpu, pass, prev_act)
    img.gpu_push_uint(gpu, pass, next_grad_act)
    img.gpu_push_uint(gpu, pass, grad_weights)
    img.gpu_push_uint(gpu, pass, ch_in)
    img.gpu_push_uint(gpu, pass, ch_out)
    img.gpu_push_uint(gpu, pass, h)   -- in_h
    img.gpu_push_uint(gpu, pass, w)   -- in_w
    img.gpu_push_uint(gpu, pass, h)   -- out_h (same as in_h for same-pad stride=1)
    img.gpu_push_uint(gpu, pass, w)   -- out_w
    img.gpu_push_uint(gpu, pass, 3)   -- kH
    img.gpu_push_uint(gpu, pass, 3)   -- kW
    img.gpu_push_uint(gpu, pass, 1)   -- pad
end

--[[

]]
function API.conv_backward_bias(gpu, next_grad_act, grad_bias, w, h, ch_out)
	print("conv back bias")
	local pass = img.gpu_add_stage(gpu,program_conv_backward_bias,ch_out,1,1)

	img.gpu_push_uint(gpu, pass, next_grad_act)
	img.gpu_push_uint(gpu, pass, grad_bias)
	img.gpu_push_uint(gpu, pass, w)
	img.gpu_push_uint(gpu, pass, h)
	img.gpu_push_uint(gpu, pass, ch_out)
end

function API.pool_backward(gpu, next_grad_act, grad_act, indices, w, h, ch)
    local pass = img.gpu_add_stage(gpu, program_pool_backward, w//2, h//2, ch)
    img.gpu_push_uint(gpu, pass, next_grad_act)
    img.gpu_push_uint(gpu, pass, grad_act)
    img.gpu_push_uint(gpu, pass, indices)
    img.gpu_push_uint(gpu, pass, w//2)  -- out_w
    img.gpu_push_uint(gpu, pass, h//2)  -- out_h
    img.gpu_push_uint(gpu, pass, ch)
    img.gpu_push_uint(gpu, pass, w)     -- in_w for winner unpacking
end

function API.linear_backward_input(gpu, next_grad_act, grad_act, weights, in_dim, out_dim, relu, act)
	print("lin back in")
	local pass = img.gpu_add_stage(gpu,program_linear_backward_input,in_dim,1,1)

	img.gpu_push_uint(gpu, pass, next_grad_act)
	img.gpu_push_uint(gpu, pass, grad_act)
	img.gpu_push_uint(gpu, pass, weights)
	img.gpu_push_uint(gpu, pass, in_dim)
	img.gpu_push_uint(gpu, pass, out_dim)
	img.gpu_push_uint(gpu, pass, relu and 1 or 0)
	img.gpu_push_uint(gpu, pass, act)
end

function API.linear_backward_weights(gpu, prev_act, next_grad_act, grad_weights, in_dim, out_dim, prev)
	print("lin back weight")
	local pass = img.gpu_add_stage(gpu,program_linear_backward_weights,in_dim,out_dim,1)

	img.gpu_push_uint(gpu, pass, prev_act)
	img.gpu_push_uint(gpu, pass, next_grad_act)
	img.gpu_push_uint(gpu, pass, grad_weights)
	img.gpu_push_uint(gpu, pass, in_dim)
	img.gpu_push_uint(gpu, pass, out_dim)
	img.gpu_push_uint(gpu, pass, prev and 1 or 0)
end

function API.linear_backward_bias(gpu, next_grad_act, grad_bias, out_dim)
	print("lin back bias")
	local pass = img.gpu_add_stage(gpu,program_linear_backward_bias,out_dim,1,1)

	img.gpu_push_uint(gpu, pass, next_grad_act)
	img.gpu_push_uint(gpu, pass, grad_bias)
	img.gpu_push_uint(gpu, pass, out_dim)
end

function API.adam(gpu, buf1, buf2, buf3, buf4, dim, lr, eps)
	print("adam")
	local pass = img.gpu_add_stage(gpu,program_adam,dim,1,1)

	img.gpu_push_uint(gpu, pass, buf1)
	img.gpu_push_uint(gpu, pass, buf2)
	img.gpu_push_uint(gpu, pass, buf3)
	img.gpu_push_uint(gpu, pass, buf4)
	img.gpu_push_uint(gpu, pass, dim)

	img.gpu_push_float(gpu,pass,lr)
	img.gpu_push_float(gpu,pass,eps)

	img.gpu_push_float(gpu,pass,1234) -- updated dynamically from elsewhere
	img.gpu_push_float(gpu,pass,5678)

	return pass
end

function API.train(gpu, net, first_binding, lr, eps)

	local binding = first_binding
	local function next_binding()
		local b = binding
		binding = binding + 1
		return b
	end

	local n = #net.layers

	local end_layer = net.layers[n]
	end_layer.grad_act = next_binding()
    end_layer.gpu_ga   = img.gpu_allocate_buffer(gpu, end_layer.grad_act, end_layer.def.dim)
	local loss_grad = API.loss_grad(gpu,end_layer.act,end_layer.grad_act, end_layer.def.dim)
	local max_size = 0

	for i = n-1, 1, -1 do
		local layer = net.layers[i]
		local next_layer = net.layers[i+1]
		local L = layer.def
		local prev_act = i==1 and net.input or net.layers[i-1].act

		if L.type == "conv" then

			layer.grad_act     = next_binding()  -- image3D same shape as act
			layer.grad_weights = next_binding()  -- SSBO same shape as weights
			layer.grad_bias    = next_binding()  -- SSBO same shape as bias
			layer.adam_mw      = next_binding()  -- SSBO same shape as weights
			layer.adam_vw      = next_binding()  -- SSBO same shape as weights
			layer.adam_mb      = next_binding()  -- SSBO same shape as bias
			layer.adam_vb      = next_binding()  -- SSBO same shape as bias

			layer.gpu_ga = img.gpu_allocate_image(gpu, layer.grad_act, L.w, L.h, L.ch_out, 1)
			layer.gpu_gw = img.gpu_allocate_buffer(gpu, layer.grad_weights, L.ch_out * L.ch_in * 9)
			layer.gpu_gb = img.gpu_allocate_buffer(gpu, layer.grad_bias, L.ch_out)
			layer.gpu_mw = img.gpu_allocate_buffer(gpu, layer.adam_mw, L.ch_out * L.ch_in * 9)
			layer.gpu_vw = img.gpu_allocate_buffer(gpu, layer.adam_vw, L.ch_out * L.ch_in * 9)
			layer.gpu_mb = img.gpu_allocate_buffer(gpu, layer.adam_mb, L.ch_out)
			layer.gpu_vb = img.gpu_allocate_buffer(gpu, layer.adam_vb, L.ch_out)

			if L.ch_out * L.ch_in * 9 > max_size then max_size = L.ch_out * L.ch_in * 9 end

			API.conv_backward_input(gpu, next_layer.grad_act, layer.grad_act, layer.weights, layer.act, L.w, L.h, L.ch_in, L.ch_out)
			API.conv_backward_weights(gpu,prev_act, next_layer.grad_act, layer.grad_weights, L.w, L.h, L.ch_in, L.ch_out)
			API.conv_backward_bias(gpu, next_layer.grad_act, layer.grad_bias, L.w, L.h, L.ch_out)
		
		elseif L.type == "pool" then
			layer.grad_act = next_binding()
			-- allocate at FULL input size, not output size
			layer.gpu_ga = img.gpu_allocate_image(gpu, layer.grad_act, L.w, L.h, L.ch, 1)

			API.pool_backward(gpu, next_layer.grad_act, layer.grad_act, layer.indices, L.w, L.h, L.ch, L.next)

		elseif L.type == "linear" then
			layer.grad_act     = next_binding()
			layer.grad_weights = next_binding()
			layer.grad_bias    = next_binding()
			layer.adam_mw      = next_binding()
			layer.adam_vw      = next_binding()
			layer.adam_mb      = next_binding()
			layer.adam_vb      = next_binding()

			layer.gpu_ga = img.gpu_allocate_buffer(gpu, layer.grad_act, L.out_dim)
			layer.gpu_gw = img.gpu_allocate_buffer(gpu, layer.grad_weights, L.out_dim * L.in_dim)
			layer.gpu_gb = img.gpu_allocate_buffer(gpu, layer.grad_bias, L.out_dim)
			layer.gpu_mw = img.gpu_allocate_buffer(gpu, layer.adam_mw, L.out_dim * L.in_dim)
			layer.gpu_vw = img.gpu_allocate_buffer(gpu, layer.adam_vw, L.out_dim * L.in_dim)
			layer.gpu_mb = img.gpu_allocate_buffer(gpu, layer.adam_mb, L.out_dim)
			layer.gpu_vb = img.gpu_allocate_buffer(gpu, layer.adam_vb, L.out_dim)

			API.linear_backward_input(gpu, next_layer.grad_act, layer.grad_act, layer.weights, L.in_dim, L.out_dim, L.relu, layer.act)
			API.linear_backward_weights(gpu, prev_act, next_layer.grad_act, layer.grad_weights, L.in_dim, L.out_dim, L.prev)
			API.linear_backward_bias(gpu, next_layer.grad_act, layer.grad_bias, L.out_dim)
        end
	end

		-- adam
	for i,layer in ipairs(net.layers) do
		local L = layer.def
		if L.type == "conv" then
			layer.adam1 = API.adam(gpu, layer.weights, layer.grad_weights, layer.adam_mw, layer.adam_vw, L.ch_out * L.ch_in * 9 , lr, eps)
			layer.adam2 = API.adam(gpu, layer.bias, layer.grad_bias, layer.adam_mb, layer.adam_vb, L.ch_out, lr, eps)
		elseif L.type == "linear" then
			layer.adam1 = API.adam(gpu, layer.weights, layer.grad_weights, layer.adam_mw, layer.adam_vw, L.out_dim * L.in_dim, lr, eps)
			layer.adam2 = API.adam(gpu, layer.bias, layer.grad_bias, layer.adam_mb, layer.adam_vb, L.out_dim, lr, eps)	
		end
	end

	local zeros = img.create(max_size, 1, 1, 1, 3)  -- calloc'd so already zero
	-- cpu side is free, just reuse it

	for i, layer in ipairs(net.layers) do
		local L = layer.def
		if L.type == "conv" then
			img.gpu_upload_now(gpu, layer.gpu_ga, zeros)
			img.gpu_upload_now(gpu, layer.gpu_gw, zeros)
			img.gpu_upload_now(gpu, layer.gpu_gb, zeros)
			img.gpu_upload_now(gpu, layer.gpu_mw, zeros)
			img.gpu_upload_now(gpu, layer.gpu_vw, zeros)
			img.gpu_upload_now(gpu, layer.gpu_mb, zeros)
			img.gpu_upload_now(gpu, layer.gpu_vb, zeros)
		elseif L.type == "linear" then
			img.gpu_upload_now(gpu, layer.gpu_ga, zeros)
			img.gpu_upload_now(gpu, layer.gpu_gw, zeros)
			img.gpu_upload_now(gpu, layer.gpu_gb, zeros)
			img.gpu_upload_now(gpu, layer.gpu_mw, zeros)
			img.gpu_upload_now(gpu, layer.gpu_vw, zeros)
			img.gpu_upload_now(gpu, layer.gpu_mb, zeros)
			img.gpu_upload_now(gpu, layer.gpu_vb, zeros)
		elseif L.type == "pool" then
			img.gpu_upload_now(gpu, layer.gpu_ga, zeros)  -- the scatter target
		end
	end

	img.destroy(zeros)

	return loss_grad
end

function API.weights_save(gpu, net, folder)
    print("saving trained weights")

    local conv_idx = 0
    local fc_idx = 0

    for i, layer in ipairs(net.layers) do
        local L = layer.def

        if L.type == "conv" then
            local w = img.create(layer.size_w,1,1,1,3) -- or whatever alloc method matches your system
            local b = img.create(layer.size_b,1,1,1,3)

            img.gpu_download_now(gpu, layer.gpu_w, w)
            img.gpu_download_now(gpu, layer.gpu_b, b)

            img.write_raw(w, string.format("%sconv%d_w.bin", folder, conv_idx), "wb")
            img.write_raw(b, string.format("%sconv%d_b.bin", folder, conv_idx), "wb")

            conv_idx = conv_idx + 1

        elseif L.type == "linear" then
            local w = img.create(layer.size_w,1,1,1,3)
            local b = img.create(layer.size_b,1,1,1,3)

            img.gpu_download_now(gpu, layer.gpu_w, w)
            img.gpu_download_now(gpu, layer.gpu_b, b)

            img.write_raw(w, string.format("%sfc_%d.weight.bin", folder, fc_idx*3), "wb")
            img.write_raw(b, string.format("%sfc_%d.bias.bin", folder, fc_idx*3), "wb")

            fc_idx = fc_idx + 1
        end
    end
end

function API.weights_load(gpu, net, folder)
	print("loading pretrained weights")
    local conv_idx = 0
    local fc_idx = 0
    for i, layer in ipairs(net.layers) do
        local L = layer.def
        if L.type == "conv" then
            local w = img.create_raw(string.format("%sconv%d_w.bin",folder, conv_idx))
            local b = img.create_raw(string.format("%sconv%d_b.bin",folder, conv_idx))
            img.gpu_upload_now(gpu, layer.gpu_w, w)
            img.gpu_upload_now(gpu, layer.gpu_b, b)
            conv_idx = conv_idx + 1
        elseif L.type == "linear" then
            local w = img.create_raw(string.format("%sfc_%d.weight.bin",folder, fc_idx*3))
            local b = img.create_raw(string.format("%sfc_%d.bias.bin",folder, fc_idx*3))
            img.gpu_upload_now(gpu, layer.gpu_w, w)
            img.gpu_upload_now(gpu, layer.gpu_b, b)
            fc_idx = fc_idx + 1
        end
    end
end

function API.weights_random(gpu,net)
	print("loading random weights")
    for i, layer in ipairs(net.layers) do
        local L = layer.def
        if L.type == "conv" then
			local size = L.ch_out * L.ch_in * 9
            local w = img.create(size,1,1,1,3)
			img.randomize_kaiming_normal(w,L.ch_in * 9,size)
            local b = img.create(L.ch_out,1,1,1,3)
            img.gpu_upload_now(gpu, layer.gpu_w, w)
            img.gpu_upload_now(gpu, layer.gpu_b, b)
			img.destroy(w)
			img.destroy(b)
        elseif L.type == "linear" then
			local size = L.in_dim * L.out_dim
            local w = img.create(size,1,1,1,3) -- float32
			img.randomize_kaiming_normal(w,L.in_dim,size)
            local b = img.create(L.out_dim,1,1,1,3)
            img.gpu_upload_now(gpu, layer.gpu_w, w)
            img.gpu_upload_now(gpu, layer.gpu_b, b)
			img.destroy(w)
			img.destroy(b)
        end
    end
end

return API