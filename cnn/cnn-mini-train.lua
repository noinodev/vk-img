local img = require("img")
local cnn = require("cnn/cnn")

local gpu = img.gpu_init()

local file = "datasets/mnist/test.bin"
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
local image_class = img.create(10,1,1,1,3) -- softmax output basically

local w, h = width, height -- 28x28
local nn = {
    -- block 1
    { type="conv", w=w, h=h, ch_in=1, ch_out=64  },
    { type="relu", w=w, h=h, ch=64 },
    { type="pool", w=w, h=h, ch=64, next = true },
    -- block 2
    { type="conv", w=w/2, h=h/2, ch_in=64, ch_out=128 },
    { type="relu", w=w/2, h=h/2, ch=128 },
    { type="pool", w=w/2, h=h/2, ch=128, next = true },
    -- block 3
    { type="conv", w=w/4, h=h/4, ch_in=128, ch_out=256 },
    { type="relu", w=w/4, h=h/4, ch=256 },
    { type="pool", w=w/4, h=h/4, ch=256, next = true }, -- implicit dim halving in max pool*/
    -- fc
    { type="linear", in_dim=256*(w//8)*(h//8), out_dim=256, relu=true, prev=true},
    { type="linear", in_dim=256, out_dim=10, relu=false, prev=false },
    -- output
    { type="softmax", dim=10 },
}

cnn.load(gpu)
cnn.to_3d(gpu,width,height,channels,0,2)

local net = cnn.build(gpu, nn, 2, 4)
local train = cnn.train(gpu,net,net.next_free_binding, 1e-3, 1e-7 )
cnn.weights_random(gpu,net)

local output_buffer = net.layers[#net.layers].gpu_a
local download = img.gpu_download(gpu,output_buffer,image_class)

print("OK!!")

local function get_shuffled_indices(max_count)
    local indices = {}
    for i = 0, max_count - 1 do
        indices[i + 1] = i
    end
    -- Fisher-Yates shuffle
    for i = #indices, 2, -1 do
        local r = math.random(i)
        indices[i], indices[r] = indices[r], indices[i]
    end
    return indices
end

local epochs = 50
local adam_t = 1
local last_correct = 0

local total_pages = 0
local test_atlas = img.create_atlas(file, page_size, total_pages)
while test_atlas do
    total_pages = total_pages + 1
    img.destroy_atlas(test_atlas)
    test_atlas = img.create_atlas(file, page_size, total_pages)
end

print("Found total dataset pages: " .. total_pages)
math.randomseed(os.time())

local running_loss = nil
local smoothed_acc = nil
local total_iter = 0

for k = 0, epochs - 1 do
    local count = 0
    local correct = 0
    
    local shuffled_pages = get_shuffled_indices(total_pages)

    for p = 1, #shuffled_pages do
        local page_idx = shuffled_pages[p]
        local atlas = img.create_atlas(file, page_size, page_idx)
        
        if atlas then
            local image_count = img.atlas_count(atlas)
            local shuffled_images = get_shuffled_indices(image_count)

            for idx = 1, #shuffled_images do
                local j = shuffled_images[idx]
                
                image_batch = img.create_from_atlas(atlas, j)
                img.remap(image_batch, image_staging)

                local coarse, fine = img.meta(image_batch)

                img.gpu_map_host(gpu, upload, image_staging)
                local adam_bias1 = 1 - 0.9^adam_t
                local adam_bias2 = 1 - 0.999^adam_t
                img.gpu_set_stage_data(gpu, train, fine, 2) 

                adam_t = adam_t + 1

                for i, layer in ipairs(net.layers) do
                    local L = layer.def
                    if L.type == "conv" or L.type == "linear" then
                        img.gpu_set_stage_data(gpu, layer.adam1, adam_bias1, 7)
                        img.gpu_set_stage_data(gpu, layer.adam1, adam_bias2, 8)

                        img.gpu_set_stage_data(gpu, layer.adam2, adam_bias1, 7)
                        img.gpu_set_stage_data(gpu, layer.adam2, adam_bias2, 8)
                    end
                end

                local time = img.gpu_dispatch(gpu)

                -- process results
                local label, value, true_label_prob = img.class_1d(image_class, fine)
                
                if label == fine then correct = correct + 1 end
                count = count + 1
                total_iter = total_iter + 1

                local scalar_loss = -math.log(math.max(true_label_prob, 1e-15))
                local current_acc = (label == fine) and 1.0 or 0.0

                -- average loss
                if not running_loss then
                    running_loss = scalar_loss
                    smoothed_acc = current_acc
                else
                    running_loss = (0.99 * running_loss) + (0.01 * scalar_loss)
                    smoothed_acc = (0.99 * smoothed_acc) + (0.01 * current_acc)
                end

                -- logging
                io.write(string.format(
                    "\r[Epoch: %d/%d  Img: %d/%d]  Loss: %.4f (raw: %.2f) | Acc: %5.1f%% | Current -> Lbl: %d Gs: %d Conf: %4.0f%%",
                    k + 1, epochs,
                    count, atlas_size,
                    running_loss, scalar_loss,
                    smoothed_acc * 100,
                    fine, label, value * 100
                ))
                io.flush()
            end

            img.destroy_atlas(atlas)
        end
    end

    print(string.format("\nepoch complete: %d/%d correct (%f)", correct, count, correct / count))
    cnn.weights_save(gpu, net, "cnn/weights/vgg-mnist/")

    last_correct = correct
    print("")
end

print(string.format("training complete"))


-- download weights