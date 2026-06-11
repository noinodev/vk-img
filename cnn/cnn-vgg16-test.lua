local img = require("img")
local cnn = require("cnn/cnn")

local gpu = img.gpu_init()

local file = "datasets/cifar100/test.bin"
local page_size = 1000

local atlas = img.create_atlas(file,page_size,0)
local image_first = img.create_from_atlas(atlas,0) -- first image in atlas
local width,height,depth,channels = img.dim(image_first);
local atlas_size = img.atlas_size(atlas)
local image_staging = img.create(width,height,depth,4,0)
img.remap(image_first,image_staging)

img.destroy_atlas(atlas)

local input_gpu_uniform = 0
local input_gpu = img.gpu_allocate_image(gpu,input_gpu_uniform,width,height,depth,4)
local upload = img.gpu_upload(gpu,input_gpu,image_staging)

local gpu_image_output = img.gpu_allocate_image(gpu,2,width,height,channels,1)
local image_class = img.create(100,1,1,1,3) -- softmax output basically

local w, h = width, height


local vgg_16 = {
    -- block 1
    { type="conv", w=w, h=h, ch_in=3, ch_out=64  },
    { type="relu", w=w, h=h, ch=64  },
    { type="conv", w=w, h=h, ch_in=64, ch_out=64  },
    { type="relu", w=w, h=h, ch=64  },
    { type="pool", w=w, h=h, ch=64, next = true },
    -- block 2
    { type="conv", w=w/2, h=h/2, ch_in=64, ch_out=128 },
    { type="relu", w=w/2, h=h/2, ch=128  },
    { type="conv", w=w/2, h=h/2, ch_in=128, ch_out=128 },
    { type="relu", w=w/2, h=h/2, ch=128  },
    { type="pool", w=w/2, h=h/2, ch=128, next = true },
    -- block 3
    { type="conv", w=w/4, h=h/4, ch_in=128, ch_out=256 },
    { type="relu", w=w/4, h=h/4, ch=256  },
    { type="conv", w=w/4, h=h/4, ch_in=256, ch_out=256 },
    { type="relu", w=w/4, h=h/4, ch=256  },
    { type="conv", w=w/4, h=h/4, ch_in=256, ch_out=256 },
    { type="relu", w=w/4, h=h/4, ch=256  },
    { type="pool", w=w/4, h=h/4, ch=256, next = true },
    -- block 4
    { type="conv", w=w/8, h=h/8, ch_in=256, ch_out=512 },
    { type="relu", w=w/8, h=h/8, ch=512  },
    { type="conv", w=w/8, h=h/8, ch_in=512, ch_out=512 },
    { type="relu", w=w/8, h=h/8, ch=512  },
    { type="conv", w=w/8, h=h/8, ch_in=512, ch_out=512 },
    { type="relu", w=w/8, h=h/8, ch=512  },
    { type="pool", w=w/8, h=h/8, ch=512, next = true }, -- 2x2x512 = 2048
    -- block 5
    { type="conv", w=w/16, h=h/16, ch_in=512, ch_out=512 },
    { type="relu", w=w/16, h=h/16, ch=512  },
    { type="conv", w=w/16, h=h/16, ch_in=512, ch_out=512 },
    { type="relu", w=w/16, h=h/16, ch=512  },
    { type="conv", w=w/16, h=h/16, ch_in=512, ch_out=512 },
    { type="relu", w=w/16, h=h/16, ch=512  },
    { type="pool", w=w/16, h=h/16, ch=512, next = false},
    -- fc
    { type="linear", in_dim=512, out_dim=512, relu=true, prev=true}, -- 2048 linear
    { type="linear", in_dim=512, out_dim=512, relu=true, prev=false },
    { type="linear", in_dim=512, out_dim=100, relu=false, prev=false },
    -- output
    { type="softmax", dim=100 },
}

cnn.load(gpu)
cnn.to_3d(gpu,width,height,channels,0,2)

local net = cnn.build(gpu, vgg_16, 2, 4)
cnn.weights_load(gpu,net,"cnn/weights/vgg-16/")

local output_buffer = net.layers[#net.layers].gpu_a
local download = img.gpu_download(gpu,output_buffer,image_class)

print("OK!!")

local count = 0
local correct = 0
local i = 0
atlas = img.create_atlas(file,page_size,i)
while atlas do
    local image_count = img.atlas_count(atlas)
    for j=0, image_count-1 do
        image_batch = img.create_from_atlas(atlas,j)
        img.remap(image_batch,image_staging)

        local coarse,fine = img.meta(image_batch)
        img.gpu_map_host(gpu,upload,image_staging)

        local time = img.gpu_dispatch(gpu) -- copy host->device, feed forward, backprop, weights persist

        local label,value,_ = img.class_1d(image_class,fine)
        if label == fine then correct = correct+1 end
        count = count+1
        io.write(string.format("label: %d, guess: %d, confidence: %f, accuracy: %d/%d (%f), time: %fms",fine,label,value,correct,count,correct/count,time))
        io.flush()
        io.read()
    end

    img.destroy_atlas(atlas)
    i = i+1
    atlas = img.create_atlas(file,page_size,i) -- using 0 to restrict current test to 50 images
end
print(string.format("\ntest complete: %d/%d correct (%f)",correct,count,correct/count))