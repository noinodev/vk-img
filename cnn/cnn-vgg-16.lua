local img = require("img")
local cnn = require("cnn/cnn")

local gpu = img.gpu_init()

local file = "datasets/cifar100/test.bin"
--local file = "datasets/imagenette2-320/train.bin"
local page_size = 500

local atlas = img.create_atlas(file,page_size,0)
local image_first = img.create_from_atlas(atlas,0) -- first image in atlas
local width,height,depth,channels = img.dim(image_first);
local atlas_size = img.atlas_size(atlas)
--print(img.dim(image_first))

--local image_batch = img.create(width,height,depth,3,0)
local image_staging = img.create(width,height,depth,4,0)
img.remap(image_first,image_staging)

img.destroy_atlas(atlas)

local input_gpu_uniform = 0
local input_gpu = img.gpu_allocate_image(gpu,input_gpu_uniform,width,height,depth,4)
local upload = img.gpu_upload(gpu,input_gpu,image_staging)

--local output_gpu_uniform = 1 -- gpu uniform / texture id
--local output_gpu = img.gpu_allocate_image(gpu,output_gpu_uniform,width,height,1,4) -- gpu texture

local gpu_image_output = img.gpu_allocate_image(gpu,2,width,height,channels,1)
local image_class = img.create(100,1,1,1,3) -- softmax output basically

local w, h = width, height

--[[
local vgg_16 = {
    -- block 1
    { type="conv", w=w, h=h, ch_in=3, ch_out=64  },
    { type="conv", w=w, h=h, ch_in=64, ch_out=64  },
    { type="pool", w=w, h=h, ch=64, next = true },
    -- block 2
    { type="conv", w=w/2, h=h/2, ch_in=64, ch_out=128 },
    { type="conv", w=w/2, h=h/2, ch_in=128, ch_out=128 },
    { type="pool", w=w/2, h=h/2, ch=128, next = true },
    -- block 3
    { type="conv", w=w/4, h=h/4, ch_in=128, ch_out=256 },
    { type="conv", w=w/4, h=h/4, ch_in=256, ch_out=256 },
    { type="conv", w=w/4, h=h/4, ch_in=256, ch_out=256 },
    { type="pool", w=w/4, h=h/4, ch=256, next = true },
    -- block 4
    { type="conv", w=w/8, h=h/8, ch_in=256, ch_out=512 },
    { type="conv", w=w/8, h=h/8, ch_in=512, ch_out=512 },
    { type="conv", w=w/8, h=h/8, ch_in=512, ch_out=512 },
    { type="pool", w=w/8, h=h/8, ch=512, next = true }, -- 2x2x512 = 2048
    -- block 5
    { type="conv", w=w/16, h=h/16, ch_in=512, ch_out=512 },
    { type="conv", w=w/16, h=h/16, ch_in=512, ch_out=512 },
    { type="conv", w=w/16, h=h/16, ch_in=512, ch_out=512 },
    { type="pool", w=w/16, h=h/16, ch=512, next = false},
    -- fc
    { type="linear", in_dim=512, out_dim=512, relu=true, prev=true}, -- 2048 linear
    { type="linear", in_dim=512, out_dim=512, relu=true, prev=false },
    { type="linear", in_dim=512, out_dim=100, relu=false, prev=false },
    -- output
    { type="softmax", dim=100 },
}
]]
local vgg_7 = {
    -- block 1
    { type="conv", w=w, h=h, ch_in=3, ch_out=64  },
    { type="pool", w=w, h=h, ch=64, next = true },
    -- block 2
    { type="conv", w=w/2, h=h/2, ch_in=64, ch_out=128 },
    { type="pool", w=w/2, h=h/2, ch=128, next = true },
    -- block 3
    { type="conv", w=w/4, h=h/4, ch_in=128, ch_out=256 },
    { type="pool", w=w/4, h=h/4, ch=256, next = true },
    -- block 4
    { type="conv", w=w/8, h=h/8, ch_in=256, ch_out=512 },
    { type="pool", w=w/8, h=h/8, ch=512, next = true },
    -- block 5
    { type="conv", w=w/16, h=h/16, ch_in=512, ch_out=512 },
    { type="pool", w=w/16, h=h/16, ch=512, next = false},
    -- fc
    { type="linear", in_dim=512, out_dim=512, relu=true, prev=true},
    { type="linear", in_dim=512, out_dim=100, relu=false, prev=false },
    -- output
    { type="softmax", dim=100 },
}

cnn.load(gpu)
cnn.to_3d(gpu,width,height,channels,0,2)

local net = cnn.build(gpu, vgg_7, 2, 4)
local train = 0
train = cnn.train(gpu,net,net.next_free_binding, 1e-4, 1e-8 )
--cnn.weights_random(gpu,net)
cnn.weights_load(gpu,net,"vgg-short/")

local output_buffer = net.layers[#net.layers].gpu_a
local download = img.gpu_download(gpu,output_buffer,image_class)

print("OK!!")

local epochs = 500
local adam_t = 1
local last_correct = 0

for k=0,epochs-1 do
    local count = 0
    local correct = 0
    local i = 0
    atlas = img.create_atlas(file,page_size,i)
    while atlas do
        --count = 0
        --correct = 0
        local image_count = img.atlas_count(atlas)
        for j=0, image_count-1 do
            image_batch = img.create_from_atlas(atlas,j)
            img.remap(image_batch,image_staging)

            local coarse,fine = img.meta(image_batch)


            img.gpu_map_host(gpu,upload,image_staging)
            local adam_bias1 = 1 - 0.9^adam_t
            local adam_bias2 = 1 - 0.999^adam_t
            if train > 0 then 
                img.gpu_set_stage_data(gpu,train,fine,2) 

                adam_t = adam_t + 1

                for i,layer in ipairs(net.layers) do
                    local L = layer.def
                    if L.type == "conv" or L.type == "linear" then
                        img.gpu_set_stage_data(gpu,layer.adam1,adam_bias1,7) -- bias1
                        img.gpu_set_stage_data(gpu,layer.adam1,adam_bias2,8) -- bias2

                        img.gpu_set_stage_data(gpu,layer.adam2,adam_bias1,7) -- bias1
                        img.gpu_set_stage_data(gpu,layer.adam2,adam_bias2,8) -- bias2
                    end
                end
            end

            local time = img.gpu_dispatch(gpu) -- copy host->device, feed forward, backprop, weights persist

            local label,value = img.class_1d(image_class)
            if label == fine then correct = correct+1 end
            count = count+1
            --print(string.format("epoch: %d, page: %d image: %d/%d true: %d guess: %d conf: %f (%d/%d = %f) (%fms) adam %f, %f",k,i,count,atlas_size,fine,label,value,correct,count,correct/(count),time,adam_bias1,adam_bias2))
            print(string.format("\rlabel: %d, guess: %d, confidence: %f           ..",fine,label,value))
            if j%10 == 0 then 
                io.write("\r[e: "..k+1 .. "/" .. epochs .. "  i: " .. count .. "/" .. atlas_size .. " a: " .. correct/count .. "]")
                io.flush()
            end
            --io.read()
        end

        --print(string.format("epoch complete: %d/%d correct (%f)",correct,count,correct/count))

        img.destroy_atlas(atlas)
        i = i+1
        atlas = img.create_atlas(file,page_size,i) -- using 0 to restrict current test to 50 images
    end
    print(string.format("\nepoch complete: %d/%d correct (%f)",correct,count,correct/count))
    --local e = io.read("save weights?")
    cnn.weights_save(gpu,net,"vgg-short/")

    --[[if correct < last_correct then
        print("model going backwards",last_correct,correct)

        local s = io.read("stop? y/n")
        if s == 'y' then break end
    end]]

    last_correct = correct
end

print(string.format("training complete"))


-- download weights