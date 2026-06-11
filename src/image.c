#include "image.h"
#include "vulkan/vulkan_core.h"
#include <endian.h>
#include <stdio.h>

size_t img_get_stride(img_t* img){
    switch(img->type){
        case IMG_TYPE_BYTE: return sizeof(unsigned char);
        case IMG_TYPE_SHORT: return sizeof(unsigned short);
        case IMG_TYPE_INT: return sizeof(unsigned int);
        case IMG_TYPE_FLOAT: return sizeof(float);
        case IMG_TYPE_DOUBLE: return sizeof(double);
    }
}

void img_alloc(img_t* img){
    assert(img);
    assert(img->width > 0);
    assert(img->height > 0);
    assert(img->depth > 0);
    assert(img->channels > 0);
    
    size_t stride = img_get_stride(img);
    size_t size = img->width * img->height * img->depth * img->channels;

    img->memory = (uint8_t*)calloc(size,stride);

    assert(img->memory);
}

void img_destroy(img_t* img){
    assert(img);
    assert(img->memory);
    
    free(img->memory);

    *img = (img_t){0};
}

int img_validate(img_t* img){
    if(img->memory == NULL) return 0;
    else return 1;
}

size_t img_get_size(img_t* img){
    return img->width*img->height*img->depth*img->channels;
}

img_t img_create(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels, uint32_t type){
    img_t out = {0};
    out.width = width;
    out.height = height;
    out.depth = depth;
    out.channels = channels;
    out.type = type;

    //printf("%d %d %d %d %d \n",width,height,depth,channels,type); // prints 686,386,1,4,0 (correct)
    //printf("%d %d %d %d %d \n",out.width,out.height,out.depth,out.channels,out.type); // prints 0,0,0,0,0

    img_alloc(&out);
    //printf("img_create_zero\n");
    return out;
}

void img_fill(img_t* img, double r, double g, double b, double a){
    double cols[] = {r,g,b,a};
    size_t stride = img_get_stride(img);
    size_t size = img_get_size(img);
    size_t channels = img->channels;

    switch(img->type){
        case IMG_TYPE_BYTE:{
            for(size_t i = 0; i < size; i++){
                ((unsigned char*)img->memory)[i] = (unsigned char)cols[i%channels];
            }
        } break;

        case IMG_TYPE_SHORT:{
            for(size_t i = 0; i < size; i++){
                ((unsigned short*)img->memory)[i] = (unsigned short)cols[i%channels];
            }
        } break;

        case IMG_TYPE_INT:{
            for(size_t i = 0; i < size; i++){
                ((unsigned int*)img->memory)[i] = (unsigned int)cols[i%channels];
            }
        } break;

        case IMG_TYPE_FLOAT:{
            for(size_t i = 0; i < size; i++){
                ((float*)img->memory)[i] = (float)cols[i%channels];
            }
        } break;

        case IMG_TYPE_DOUBLE:{
            for(size_t i = 0; i < size; i++){
                ((double*)img->memory)[i] = (double)cols[i%channels];
            }
        } break;
    }
}

void img_remap(img_t* src, img_t* dst){
    if(
        src->width != dst->width ||
        src->height != dst->height ||
        src->depth != dst->depth
    ){
        printf("remap fail: src and dst must be same dimensions\n");
        return;
    }

    size_t size = img_get_size(src)/src->channels;
    size_t min_channels = dst->channels < src->channels ? dst->channels : src->channels;
    size_t src_stride = img_get_stride(src);
    size_t dst_stride = img_get_stride(dst);

    for(size_t i = 0; i < size; i++){
        for(size_t j = 0; j < min_channels; j++){
            dst->memory[(i*dst->channels+j)*dst_stride] = src->memory[(i*src->channels+j)*src_stride];
        }
    }
}

img_t img_create_from_image(const char* file, uint32_t channels){
    int w,h,c;
    stbi_uc* pixels = stbi_load(file, &w, &h, &c, channels);
    if(channels != 0) c = channels;

    img_t out = img_create(w,h,1,c,IMG_TYPE_BYTE);
    size_t size = img_get_size(&out);
    memcpy(out.memory,pixels,size);

    stbi_image_free(pixels);
    printf("img_create_from_image\n");
    return out;
}

void img_write_as_image(img_t* img, const char* file){
    printf("img_write_as_image '%s' \n",file);

    if(img->depth != 1){
        printf("stb only supports 2d image writes\n");
        abort();
        exit(-1);
    }
    unsigned char* pixels = malloc(img->width*img->height*img->depth*img->channels*sizeof(unsigned char));
    for(uint32_t i = 0; i < img->width*img->height*img->depth*img->channels; i++){
        pixels[i] = (uint8_t*)img->memory[i];
    }

    stbi_write_bmp(file,img->width,img->height,img->channels,pixels);
    free(pixels);
}

img_t img_create_from_binary(const char* file){
    FILE* f = fopen(file,"r");
    img_t out;
    fread(&out,sizeof(img_t),1,f);
    fread(out.memory,sizeof(float),out.width*out.height*out.depth*out.channels,f);
    fclose(f);
    return out;
}

img_t img_create_from_binary_raw(const char* file){
    FILE* f = fopen(file,"r");
    fseek(f,0,SEEK_END);
    size_t size = (size_t)ftell(f);

    img_t out = {0};
    out.width = size;
    out.height = 1;
    out.depth = 1;
    out.channels = 1;
    img_alloc(&out);

    printf("binary_raw '%s' size: %zu\n",file,size);

    fseek(f,0,SEEK_SET);
    fread(out.memory,1,size,f);
    fclose(f);
    return out;
}

void img_write_as_binary(img_t* img, const char* file){
    FILE* f = fopen(file,"wb");
    fwrite(img,sizeof(img),1,f);
    fwrite(img->memory, sizeof(float), img->width*img->height*img->depth*img->channels, f);
    fclose(f);
}

void img_write_as_binary_raw(img_t* img, const char* file, const char* mode){
    FILE* f = fopen(file,mode);
    fwrite(img->memory, sizeof(float), img->width*img->height*img->depth*img->channels, f);
    fclose(f);
}

atlas_t img_create_atlas_from_binary(const char* file, size_t blocks, size_t block){
    FILE* f = fopen(file,"rb");
    if(!f) return (atlas_t){0};

    uint32_t headers[ATLAS_SERIAL_HEADER_SIZE] = {0};
    fread(headers,sizeof(uint32_t),ATLAS_SERIAL_HEADER_SIZE,f);

    uint32_t total_count = headers[ATLAS_SERIAL_COUNT];

    uint32_t label_stride = headers[ATLAS_SERIAL_STRIDE_META];
    uint32_t image_stride = headers[ATLAS_SERIAL_STRIDE_IMAGE];
    uint32_t total_stride = headers[ATLAS_SERIAL_STRIDE_TOTAL];

    size_t header_size = ATLAS_SERIAL_HEADER_SIZE*sizeof(uint32_t);
    size_t labels_size = total_count * label_stride;
    size_t image_start = header_size + labels_size;

    //printf("label: %d image: %d total: %d\n",label_stride,image_stride,total_stride);



    fseek(f,0,SEEK_END);
    size_t fsize = (size_t)ftell(f);

    size_t elements_in_block = blocks; // total number of blocks that fit in requested allocation
    size_t block_memory_size = elements_in_block * image_stride;

    size_t offset = image_start+block_memory_size*block;

    if(offset >= fsize){
        fclose(f);
        return (atlas_t){0};
    }else{
        if (offset + block_memory_size > fsize) {
            block_memory_size = fsize - offset;
            elements_in_block = block_memory_size / image_stride;
            block_memory_size = elements_in_block * image_stride;
        }

        uint8_t* labels = calloc(labels_size,1);
        fseek(f,header_size,SEEK_SET);
        fread(labels,1,labels_size,f);

        uint8_t* memory = calloc(block_memory_size,1);
        fseek(f,offset,SEEK_SET);
        fread(memory,1,block_memory_size,f);

        atlas_t atlas = {
            .memory = memory,
            .labels = labels,
            .start = elements_in_block * block,
            .size = total_count,

            .count = elements_in_block,
            .label_stride = label_stride,
            .image_stride = image_stride,
            .total_stride = total_stride,

            .width = headers[ATLAS_SERIAL_IMAGE_WIDTH],
            .height = headers[ATLAS_SERIAL_IMAGE_HEIGHT],
            .depth = headers[ATLAS_SERIAL_IMAGE_DEPTH],
            .channels = headers[ATLAS_SERIAL_IMAGE_CHANNELS]
        };

        fclose(f);
        return atlas;
    }
}

void img_destroy_atlas(atlas_t* atlas){
    if(atlas->memory) free(atlas->memory);
    *atlas = (atlas_t){0};
}

img_t img_create_from_atlas(atlas_t* atlas, uint32_t index){
    img_t out = {0};
    out.width = atlas->width;
    out.height = atlas->height;
    out.depth = atlas->depth;
    out.channels = atlas->channels;
    out.type = IMG_TYPE_BYTE;

    out.memory = ((uint8_t*)atlas->memory)+atlas->image_stride*index;

    for(uint32_t i = 0; i < atlas->label_stride; i++){
        uint32_t idx = (atlas->start+index)*atlas->label_stride + i;
        out.meta[i] = atlas->labels[idx];
    }
    return out;
}

void img_copy_atlas_batch(atlas_t* atlas, img_t* image, uint32_t index, uint32_t count){
    memcpy(image->memory,((uint8_t*)atlas->memory)+atlas->image_stride*index,atlas->image_stride*count);
    /*for(size_t i = 0; i < count; i++){
        memcpy(image->memory+atlas->image_stride*i,((uint8_t*)atlas->memory)+atlas->image_stride*(index+i),atlas->image_stride);
    }*/
}

uint32_t img_peek_atlas(atlas_t* atlas, uint32_t index, uint32_t label){
    return *(uint32_t*)(((uint8_t*)atlas->memory)+atlas->total_stride*index+label);
}

// gpu thing

#include <dlfcn.h>
#include "renderdoc_app.h"
RENDERDOC_API_1_1_2* rdoc = NULL;

img_gpu_t img_gpu_init(){

	// at startup, before VkInstance creation
	void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD); // linux
	// HMODULE mod = GetModuleHandleA("renderdoc.dll"); // windows
	if(mod) {
		pRENDERDOC_GetAPI fn = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
		fn(eRENDERDOC_API_Version_1_1_2, (void**)&rdoc);
	}

    img_gpu_t gpu = {0};
    vkr_state* vkr = &gpu.vkr;
    int res = 0;
    res = vkr_init(vkr);

    VkFenceCreateInfo info_fence = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };
    //VkFence fence;
    vkCreateFence(vkr->device,&info_fence,NULL,&gpu.fence);

    gpu.cmd = vkr->command_buffer[0];//vkr_stc_begin(gpu->vkr.device,gpu->vkr.command_pool);

    const vkr_descriptor_info img_gpu_descriptors[] = {
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, MAX_TEXTURES},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, MAX_TEXTURES}, // alt image
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, MAX_TEXTURES}
    };

    // descriptors
    vkr_generate_descriptor_state(vkr,sizeof(img_gpu_descriptors)/sizeof(img_gpu_descriptors[0]),img_gpu_descriptors);
    vkr_generate_pipeline_layout(vkr);

    return gpu;
}

img_gpu_program_t img_gpu_load_program_glsl(img_gpu_t* gpu, const char* shader_file, uint32_t width, uint32_t height, uint32_t depth){
    FILE* f = fopen(shader_file,"r");
    if(!f){
        printf("failed to open shader file: %s\n",shader_file);
        exit(-1);
    }

    fseek(f,0,SEEK_END);
    long fsize = ftell(f);
    fseek(f,0,SEEK_SET);

    char* glsl = malloc(fsize*sizeof(char)+1);
    fread(glsl,1,fsize,f);
    fclose(f);

    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    shaderc_compile_options_t options = shaderc_compile_options_initialize();

    shaderc_compilation_result_t module = shaderc_compile_into_spv(
        compiler,
        glsl, fsize,
        shaderc_compute_shader,
        shader_file,
        "main",
        options
    );

    if(shaderc_result_get_compilation_status(module) != shaderc_compilation_status_success) {
        printf("glslc compile fail %s\n", shaderc_result_get_error_message(module));
        exit(-1);
    }

    size_t size = shaderc_result_get_length(module);
    const uint32_t* spv = shaderc_result_get_bytes(module);

    // vk pso creation

    VkShaderModule shader_module = vkr_shader_module_create(gpu->vkr.device, spv, size);
	VkPipeline pipeline = vkr_generate_pipeline_compute(&gpu->vkr,shader_module);
    vkDestroyShaderModule(gpu->vkr.device,shader_module,NULL);
    free((void*)spv);
    free(glsl);

    img_gpu_program_t out = {0};
    out.pipeline = pipeline;
    out.workgroup = (VkExtent3D){width,height,depth};

    return out;
}

size_t img_gpu_allocate_image(img_gpu_t* gpu, uint32_t binding, uint32_t width, uint32_t height, uint32_t depth, uint32_t channels){
    size_t count = gpu->device.count++;
    img_gpu_buffer_t* buffer = &gpu->device.buffer[count];
    buffer->type = IMG_GPU_TYPE_IMAGE;

    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    if(channels == 1) format = VK_FORMAT_R32_SFLOAT;
    if(channels == 2) format = VK_FORMAT_R8_UINT;

    buffer->image = vkr_create_texture(
        &gpu->vkr,width,height,depth,format, VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,VK_IMAGE_ASPECT_COLOR_BIT
    );

    vkr_bind_view_compute(&gpu->vkr, 1, buffer->image.view, binding);
    vkr_bind_view_compute(&gpu->vkr, 2, buffer->image.view, binding);

    printf("allocated gpu image: w:%i h:%i d:%i c:%i\n",width,height,depth,channels);

    return count;
}

size_t img_gpu_allocate_buffer(img_gpu_t* gpu, uint32_t binding, size_t size){
    //img_gpu_buffer_t out = {0};
    size_t count = gpu->device.count++;
    img_gpu_buffer_t* buffer = &gpu->device.buffer[count];
    buffer->type = IMG_GPU_TYPE_BUFFER;
    buffer->buffer.size = size*sizeof(float);

    vkr_alloc(
        gpu->vkr.device,gpu->vkr.physical_device, buffer->buffer.size, 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &buffer->buffer.buffer, &buffer->buffer.memory
    );

    vkr_bind_buffer_compute(&gpu->vkr,3,buffer->buffer.buffer,binding,buffer->buffer.size);

    return count;
}

/*void img_gpu_bind_buffer(img_gpu_t* gpu, size_t index){
    img_gpu_buffer_t* buffer = &gpu->device.buffer[index];
    vkr_bind_buffer_compute(&gpu->vkr,IMG_GPU_TYPE_BUFFER,buffer->buffer.buffer,0,buffer->buffer.size);
}*/

size_t img_gpu_upload_now(img_gpu_t* gpu, size_t dest, void* src, size_t size){
    VkBuffer hostBuffer;
    VkDeviceMemory hostMemory;

    vkr_alloc(gpu->vkr.device,gpu->vkr.physical_device,size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &hostBuffer, &hostMemory);

    void* data;
    vkMapMemory(gpu->vkr.device, hostMemory, 0, size, 0, &data);
        memcpy(data, src, size);
    vkUnmapMemory(gpu->vkr.device, hostMemory);

    VkCommandBuffer cmd = vkr_stc_begin(gpu->vkr.device,gpu->vkr.command_pool);

    img_gpu_buffer_t* device = &gpu->device.buffer[dest];
    if(device->type == IMG_GPU_TYPE_IMAGE){
        vkr_texture* texture = &device->image;

        vkr_texture_transition_many(cmd,1,&texture->image,texture->layout,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,vkr_texture_subresource_default());
        texture->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        vkr_copy_buffer_to_image(&gpu->vkr,cmd,hostBuffer,texture->image, texture->width, texture->height, texture->depth);
        vkr_texture_transition_many(cmd,1,&texture->image,texture->layout,VK_IMAGE_LAYOUT_GENERAL,vkr_texture_subresource_default());
    }else if(device->type == IMG_GPU_TYPE_BUFFER){
        vkr_buffer* buffer = &device->buffer;
        vkr_copy_buffer(&gpu->vkr,cmd,hostBuffer,buffer->buffer,buffer->size);
    }else printf("copying what exactly?\n");


    vkr_stc_end(gpu->vkr.device,gpu->vkr.command_pool,gpu->vkr.queue[VKR_QUEUE_GRAPHICS], cmd);
    printf("single time upload complete: %zu bytes\n",size);

    vkDestroyBuffer(gpu->vkr.device,hostBuffer,NULL);
    vkFreeMemory(gpu->vkr.device,hostMemory,NULL);

    return 1;
}

size_t img_gpu_upload(img_gpu_t* gpu, size_t dest, void* src, size_t size){
    VkBuffer hostBuffer;
    VkDeviceMemory hostMemory;

    vkr_alloc(gpu->vkr.device,gpu->vkr.physical_device,size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &hostBuffer, &hostMemory);

    void* data;
    vkMapMemory(gpu->vkr.device, hostMemory, 0, size, 0, &data);
        memcpy(data, src, size);
    vkUnmapMemory(gpu->vkr.device, hostMemory);

    size_t count = gpu->host.count++;
    gpu->host.device_ptr[count] = dest;
    gpu->host.host_ptr[count] = NULL;
    gpu->host.mapped_ptr[count] = data;
    gpu->host.buffer[count] = (img_gpu_buffer_t){
        .type = IMG_GPU_TYPE_BUFFER,
        .buffer = {
            .buffer = hostBuffer,
            .memory = hostMemory,
            .size = size
        }
    };

    return count;
}

void img_gpu_map_host_buffer(img_gpu_t* gpu, size_t index, void* src){
    img_gpu_buffer_t* staging = &gpu->host.buffer[index]; 
    size_t size = staging->buffer.size;
    //VkDeviceMemory hostMemory = staging->buffer.memory;

    //void* data;
    //vkMapMemory(gpu->vkr.device, hostMemory, 0, size, 0, &data);
        memcpy(gpu->host.mapped_ptr[index], src, size);
    //vkUnmapMemory(gpu->vkr.device, hostMemory);
}

size_t img_gpu_download_now(img_gpu_t* gpu, size_t src, void* dest, size_t size){
    VkBuffer hostBuffer;
    VkDeviceMemory hostMemory;

    vkr_alloc(gpu->vkr.device,gpu->vkr.physical_device,size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &hostBuffer, &hostMemory);

    VkCommandBuffer cmd = vkr_stc_begin(gpu->vkr.device,gpu->vkr.command_pool);

    img_gpu_buffer_t* device = &gpu->device.buffer[src];
    if(device->type == IMG_GPU_TYPE_IMAGE){
        vkr_texture* texture = &device->image;

        vkr_texture_transition_many(cmd,1,&texture->image,texture->layout,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,vkr_texture_subresource_default());
        vkr_copy_image_to_buffer(&gpu->vkr,cmd,hostBuffer,texture->image,texture->width, texture->height, texture->depth);
        printf("copied device local image to host visible buffer\n");
        texture->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }else if(device->type == IMG_GPU_TYPE_BUFFER){
        vkr_buffer* buffer = &device->buffer;
        vkr_copy_buffer(&gpu->vkr,cmd,buffer->buffer,hostBuffer,buffer->size);
        //printf("copied device local buffer to host visible buffer\n");
    }else printf("copying what exactly?\n");


    vkr_stc_end(gpu->vkr.device,gpu->vkr.command_pool,gpu->vkr.queue[VKR_QUEUE_GRAPHICS], cmd);
    printf("single time download complete: %zu bytes\n",size);

    void* data;
    vkMapMemory(gpu->vkr.device, hostMemory, 0, size, 0, &data);
        memcpy(dest, data, size);
    vkUnmapMemory(gpu->vkr.device, hostMemory);

    vkDestroyBuffer(gpu->vkr.device,hostBuffer,NULL);
    vkFreeMemory(gpu->vkr.device,hostMemory,NULL);

    return 1;
}

size_t img_gpu_download(img_gpu_t* gpu, size_t src, void* dest, size_t size){
    VkBuffer hostBuffer;
    VkDeviceMemory hostMemory;

    vkr_alloc(gpu->vkr.device,gpu->vkr.physical_device,size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &hostBuffer, &hostMemory);

    size_t count = gpu->host.count++;
    gpu->host.device_ptr[count] = src;
    gpu->host.host_ptr[count] = dest;
    gpu->host.buffer[count] = (img_gpu_buffer_t){
        .type = IMG_GPU_TYPE_BUFFER,
        .buffer = {
            .buffer = hostBuffer,
            .memory = hostMemory,
            .size = size
        }
    };

    return count;
}

void img_gpu_map_device_buffer(img_gpu_t* gpu, size_t index, void* dest){
    gpu->host.host_ptr[index] = dest;
}

size_t img_gpu_add_stage(img_gpu_t* gpu, img_gpu_program_t* program, uint32_t width, uint32_t height, uint32_t depth){
    size_t count = gpu->stages.count++;
    gpu->stages.pass[count].program = program;
    gpu->stages.pass[count].push_size = 0;
    gpu->stages.pass[count].width = width;
    gpu->stages.pass[count].height = height;
    gpu->stages.pass[count].depth = depth;

    return count;
}

void img_gpu_add_stage_data(img_gpu_t* gpu, size_t pass, void* data, size_t size){
    size_t size_align = (size + (IMG_GPU_PUSH_ALIGNMENT - 1)) & ~(IMG_GPU_PUSH_ALIGNMENT - 1);
    size_t size_current = gpu->stages.pass[pass].push_size;
    if(size_current + size_align > IMG_GPU_PUSH_MAX){
        printf("exceeding push constant range\n");
        return;
    }

    //*(uint32_t*)(gpu->stages.pass[pass].push_data + size_current) = *(uint32_t*)data;
    memcpy((uint32_t*)(gpu->stages.pass[pass].push_data + size_current),(uint32_t*)data,size);
    gpu->stages.pass[pass].push_size += size_align;
}

void img_gpu_set_stage_data(img_gpu_t* gpu, size_t pass, void* data, size_t size, size_t offset){
    memcpy((uint32_t*)(gpu->stages.pass[pass].push_data)+offset,(uint32_t*)data,size);
}

int captures = 0;

float img_gpu_dispatch(img_gpu_t* gpu){
    if(gpu->cmd_sealed == 0){
        VkCommandBufferBeginInfo beginInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
        };

        vkBeginCommandBuffer(gpu->cmd, &beginInfo);
        vkCmdResetQueryPool(gpu->cmd, gpu->vkr.query_pool, 0, 2);
        vkCmdWriteTimestamp(gpu->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, gpu->vkr.query_pool, 0);

        for(size_t i = 0; i < gpu->host.count; i++){
            if(gpu->host.host_ptr[i] == NULL){
                size_t device_idx = gpu->host.device_ptr[i];
                img_gpu_buffer_t* device = &gpu->device.buffer[device_idx];
                //if(buffer->type != IMG_GPU_TYPE_IMAGE) continue;

                //vkr_texture* texture = &buffer->image;

                //vkr_texture_transition_many(cmd,1,&texture->image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,vkr_texture_subresource_default());
                //texture->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

                //vkr_copy_buffer_to_image(&gpu->vkr,cmd,gpu->host.buffer[i].buffer.buffer,texture->image, texture->width, texture->height, texture->depth);
                if(device->type == IMG_GPU_TYPE_IMAGE){
                    vkr_texture* texture = &device->image;

                    vkr_texture_transition_many(gpu->cmd,1,&texture->image,texture->layout,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,vkr_texture_subresource_default());
                    texture->layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

                    vkr_copy_buffer_to_image(&gpu->vkr,gpu->cmd,gpu->host.buffer[i].buffer.buffer,texture->image, texture->width, texture->height, texture->depth);
                    //printf("copied host buffer to device image\n");
                }else if(device->type == IMG_GPU_TYPE_BUFFER){
                    vkr_buffer* buffer = &device->buffer;
                    vkr_copy_buffer(&gpu->vkr,gpu->cmd,gpu->host.buffer[i].buffer.buffer,buffer->buffer,buffer->size);
                    //printf("copied host buffer to device buffer\n");
                }else printf("copying what exactly?\n");

                //gpu->host.device_ptr[i] = NULL;
            }
        }

        for(size_t i = 0; i < gpu->device.count; i++){
            img_gpu_buffer_t* buffer = &gpu->device.buffer[i];
            if(buffer->type != IMG_GPU_TYPE_IMAGE) continue;

            vkr_texture* texture = &buffer->image;

            vkr_texture_transition_many(gpu->cmd,1,&texture->image,texture->layout,VK_IMAGE_LAYOUT_GENERAL,vkr_texture_subresource_default());
            texture->layout = VK_IMAGE_LAYOUT_GENERAL;
        }

        vkCmdBindDescriptorSets(
            gpu->cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            gpu->vkr.pipeline_state.layout,
            0,
            1, &gpu->vkr.descriptor_state.sets[0],
            0, NULL
        );

        for(size_t i = 0; i < gpu->stages.count; i++){
            img_gpu_pass_t* pass = &gpu->stages.pass[i];
            uint32_t width = (pass->width + (pass->program->workgroup.width-1)) / pass->program->workgroup.width;
            uint32_t height = (pass->height + (pass->program->workgroup.height-1)) / pass->program->workgroup.height;
            uint32_t depth = (pass->depth + (pass->program->workgroup.depth-1)) / pass->program->workgroup.depth;
            vkCmdBindPipeline(gpu->cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pass->program->pipeline);
            vkCmdPushConstants(gpu->cmd,gpu->vkr.pipeline_state.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,pass->push_size, pass->push_data);
            vkCmdDispatch(gpu->cmd,width,height,depth);

            VkMemoryBarrier mb = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
            };

            vkCmdPipelineBarrier(
                gpu->cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1, &mb,
                0, NULL,
                0, NULL
            );
        }

        for(size_t i = 0; i < gpu->host.count; i++){
            if(gpu->host.host_ptr[i] != NULL){
                size_t device_idx = gpu->host.device_ptr[i];
                img_gpu_buffer_t* device = &gpu->device.buffer[device_idx];
                if(device->type == IMG_GPU_TYPE_IMAGE){
                    vkr_texture* texture = &device->image;

                    vkr_texture_transition_many(gpu->cmd,1,&texture->image,texture->layout,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,vkr_texture_subresource_default());
                    vkr_copy_image_to_buffer(&gpu->vkr,gpu->cmd,gpu->host.buffer[i].buffer.buffer,texture->image,texture->width, texture->height, texture->depth);
                    printf("copied device local image to host visible buffer\n");
                    texture->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                }else if(device->type == IMG_GPU_TYPE_BUFFER){
                    vkr_buffer* buffer = &device->buffer;
                    vkr_copy_buffer(&gpu->vkr,gpu->cmd,buffer->buffer,gpu->host.buffer[i].buffer.buffer,buffer->size);
                    //printf("copied device local buffer to host visible buffer\n");
                }else printf("copying what exactly?\n");

                //gpu->host.host_ptr[i] = NULL;
            }
        }

        vkCmdWriteTimestamp(gpu->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, gpu->vkr.query_pool, 1);

        vkEndCommandBuffer(gpu->cmd);
        //gpu->cmd_sealed = 1;
    }

    const int total_caps = 3;
    if(rdoc) rdoc->StartFrameCapture(NULL, NULL);
    //vkr_stc_end(gpu->vkr.device,gpu->vkr.command_pool,gpu->vkr.queue[VKR_QUEUE_GRAPHICS],cmd);
    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &gpu->cmd,
    };

    vkQueueSubmit(gpu->vkr.queue[VKR_QUEUE_GRAPHICS], 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(gpu->vkr.queue[VKR_QUEUE_GRAPHICS]);

    uint64_t timestamps[2];
    vkGetQueryPoolResults(
        gpu->vkr.device, gpu->vkr.query_pool, 0, 2,
        sizeof(timestamps), timestamps, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    );

    float period = gpu->vkr.properties.limits.timestampPeriod;
    float ms = (timestamps[1] - timestamps[0]) * period / 1e6f;
    //printf("GPU time: %.3fms\n", ms);
    //vkWaitForFences(gpu->vkr.device, 1, &gpu->fence, VK_TRUE, UINT64_MAX);
    //vkResetFences(gpu->vkr.device, 1, &gpu->fence);
    if(rdoc){
        rdoc->EndFrameCapture(NULL, NULL);
        captures++;
        if(captures > total_caps) exit(0);
    }

    for(size_t i = 0; i < gpu->host.count; i++){
        if(gpu->host.host_ptr[i] != NULL){
            vkr_buffer* buffer = &gpu->host.buffer[i].buffer;

            void* data = gpu->host.mapped_ptr[i];
            if(gpu->host.mapped_ptr[i] == NULL){
                vkMapMemory(gpu->vkr.device, buffer->memory, 0, buffer->size, 0, &gpu->host.mapped_ptr[i]);
                    memcpy(gpu->host.host_ptr[i], gpu->host.mapped_ptr[i], buffer->size);
                vkUnmapMemory(gpu->vkr.device, buffer->memory);
            }else memcpy(gpu->host.host_ptr[i], gpu->host.mapped_ptr[i], buffer->size);
        }
    }

    vkResetCommandBuffer(gpu->cmd,0);
    return ms;
}

void img_gpu_reset(img_gpu_t* gpu){
    for(size_t i = 0; i < gpu->device.count; i++) {
        img_gpu_buffer_t* b = &gpu->device.buffer[i];
        if(b->type == IMG_GPU_TYPE_IMAGE) {
            vkr_destroy_texture(&gpu->vkr, &b->image);
        } else if(b->type == IMG_GPU_TYPE_BUFFER) {
            vkr_destroy_buffer(&gpu->vkr, &b->buffer);
        }
    }
    
    for(size_t i = 0; i < gpu->host.count; i++) {
        img_gpu_buffer_t* b = &gpu->host.buffer[i];
        vkr_destroy_buffer(&gpu->vkr, &b->buffer);
    }

    vkResetCommandBuffer(gpu->cmd,0);
    gpu->cmd_sealed = 0;
    gpu->device.count = 0;
    gpu->host.count = 0;
    gpu->stages.count = 0;
}

// processors

/*img_t img_program_greyscale(img_t input, int argc, char** argv){
    img_t out = img_create_zero(input.width,input.height,input.depth,input.channels);
    for(size_t i = 0; i < input.width*input.height*input.depth; i++){
        size_t j = input.channels * i;
        float grey = input.memory[j]*.2126 + input.memory[j+1]*.7152 + input.memory[j+2]*.0722;

        out.memory[j] = grey;
        out.memory[j+1] = grey;
        out.memory[j+2] = grey;
        if(input.channels > 3) out.memory[j+3] = input.memory[j+3]; // preserve alpha;
    }

    return out;
}

img_t img_program_brightness(img_t input, int argc, char** argv){
    float amount = atof(argv[1]);
    img_t out = img_create_zero(input.width,input.height,input.depth,input.channels);
    for(size_t i = 0; i < input.width*input.height*input.depth; i++){
        size_t j = input.channels * i;

        out.memory[j] = input.memory[j]+amount;
        out.memory[j+1] = input.memory[j+1]+amount;
        out.memory[j+2] = input.memory[j+2]+amount;
        if(input.channels > 3) out.memory[j+3] = input.memory[j+3]; // preserve alpha;
    }

    return out;
}

img_t img_program_clamp(img_t input, int argc, char** argv){
    float min = atof(argv[1]);
    float max = atof(argv[2]);
    img_t out = img_create_zero(input.width,input.height,input.depth,input.channels);
    for(size_t i = 0; i < input.width*input.height*input.depth; i++){
        size_t j = input.channels * i;

        out.memory[j] = fmax(fmin(input.memory[j],max),min);
        out.memory[j+1] = fmax(fmin(input.memory[j+1],max),min);
        out.memory[j+2] = fmax(fmin(input.memory[j+2],max),min);
        if(input.channels > 3) out.memory[j+3] = input.memory[j+3]; // preserve alpha;
    }

    return out;
}

img_t img_program_window(img_t input, int argc, char** argv){
    float min = atof(argv[1]);
    float max = atof(argv[2]);

    img_t out = img_create_zero(input.width,input.height,input.depth,input.channels);
    for(size_t i = 0; i < input.width*input.height*input.depth; i++){
        size_t j = input.channels * i;

        out.memory[j] = (fmax(fmin(input.memory[j], max), min) - min)/(max-min);
        out.memory[j+1] = (fmax(fmin(input.memory[j+1], max), min) - min)/(max-min);
        out.memory[j+2] = (fmax(fmin(input.memory[j+2], max), min) - min)/(max-min);
        if(input.channels > 3) out.memory[j+3] = input.memory[j+3]; // preserve alpha;
    }

    return out;
}

img_t img_program_histogram_rgb(img_t input, int argc, char** argv){
    uint32_t width = 256;
    uint32_t height = 256;
    float yscale = 1;

    uint32_t* histogram = calloc(width*3,sizeof(uint32_t)); // r,g,b

    for(size_t i = 0; i < input.width*input.height*input.depth; i++){
        size_t j = input.channels * i;

        uint32_t r = ++histogram[(uint32_t)(fmin(fmax(input.memory[j],0.),1.)*255)];
        uint32_t g = ++histogram[width + (uint32_t)(fmin(fmax(input.memory[j+1],0.),1.)*255)];
        uint32_t b = ++histogram[2 * width + (uint32_t)(fmin(fmax(input.memory[j+2],0.),1.)*255)];

        yscale = fmax(yscale,r/(float)height);
        yscale = fmax(yscale,g/(float)height);
        yscale = fmax(yscale,b/(float)height);
    }

    float fill[3] = {1,1,1};
    img_t out = img_create_fill(width,height,1,3,fill);

    printf("[");
    for(uint32_t x = 0; x < width; x++){
        printf("%d ",histogram[x]);
        for(uint32_t y = 0; y < height; y++){
            uint32_t sy = (uint32_t)floor(y*yscale);

            uint32_t r = histogram[x];
            uint32_t g = histogram[width + x];
            uint32_t b = histogram[2 * width + x];

            uint32_t h[3] = {histogram[x],histogram[width + x],histogram[2 * width + x]};
            int ch[3] = {0,1,2};

            for(int i=0;i<3;i++){
                for(int j=i+1;j<3;j++){
                    if(h[i] < h[j]){
                        uint32_t th = h[i]; h[i] = h[j]; h[j] = th;
                        int tc = ch[i]; ch[i] = ch[j]; ch[j] = tc;
                    }
                }
            }

            for(int k=0;k<3;k++){
                uint32_t height_bar = h[k];
                int channel = ch[k];

                if(sy <= height_bar){
                    size_t idx = (x + (height - 1 - y) * width) * 3;

                    for(size_t l = 0; l < 3; l++) out.memory[idx+l] = 0.;
                    out.memory[idx+channel] = 1.;
                }
            }
        }
    }
    printf("]\n");

    free(histogram);

    return out;
}

img_t img_program_histogram(img_t input, int argc, char** argv){
    uint32_t width = 256;
    uint32_t height = 256;
    float yscale = 1;

    uint32_t* histogram = calloc(width,sizeof(uint32_t));

    for(size_t i = 0; i < input.width*input.height*input.depth; i++){
        size_t j = input.channels * i;
        float grey = input.memory[j]*.2126 + input.memory[j+1]*.7152 + input.memory[j+2]*.0722;
        uint32_t intensity = ++histogram[(uint32_t)(fmin(fmax(grey,0.),1.)*255)];
        yscale = fmax(yscale,intensity/(float)height);
    }

    float fill[3] = {1,1,1};
    img_t out = img_create_fill(width,height,1,3,fill);

    printf("[");
    for(uint32_t x = 0; x < width; x++){
        printf("%d ",histogram[x]);
        for(uint32_t y = 0; y < height; y++){
            uint32_t sy = (uint32_t)floor(y*yscale);

            uint32_t intensity = histogram[x];

            uint32_t idx = (x+(height-1-y)*width) * 3;

            if(intensity >= sy) out.memory[idx+1] = 0.;
            
        }
    }
    printf("]\n");

    free(histogram);

    return out;
}

img_t img_program_otsu(img_t input, int argc, char** argv){
    uint32_t bins = 256;

    uint32_t* histogram = calloc(bins,sizeof(uint32_t));

    const float coeff[] = {.2126,.7152,.0722,0.}; // rgba
    const size_t coeff_n = sizeof(coeff)/sizeof(coeff[0]);

    uint32_t channels = 1;
    img_t out = img_create_zero(input.width,input.height,input.depth,channels);

    size_t pixel_count = input.width*input.height*input.depth;

    for(size_t i = 0; i < pixel_count; i++){
        size_t j = input.channels * i;

        float grey = 0.;

        for(size_t k = 0; k < coeff_n; k++){
            uint32_t ch = (k < input.channels) ? k : input.channels-1;
            grey += input.memory[j+ch] * coeff[k];
        }
        histogram[(uint32_t)(fmin(fmax(grey,0.),1.)*(bins-1))] += 1; // intensity histogram
        out.memory[i] = grey; // intermediate image (conv to greyscale)
    }

    double intensity_sum = 0;
    for(uint32_t t = 0; t < bins; t++) intensity_sum += (((double)t) / (bins)) * histogram[t];

    double threshold = 0;
    double sumB = 0;
    double wB = 0;
    double maxVar = 0;

    for(uint32_t t = 0; t < bins; t++){
        wB += histogram[t];
        if(wB == 0) continue;

        double wF = pixel_count-wB;
        if(wF == 0) break;

        sumB += (((double)t) / (bins)) * histogram[t];
        double mB = sumB/wB;
        double mF = (intensity_sum-sumB) / wF;

        double varBetween = wB*wF*(mB-mF)*(mB-mF);

        if(varBetween > maxVar){
            maxVar = varBetween;
            threshold = t;
        }
    }

    float thr = (float)threshold / (bins - 1);
    for(size_t i = 0; i < input.width*input.height*input.depth; i++) out.memory[i] = out.memory[i] > thr ? 1.0f : 0.0f;

    free(histogram);

    return out;
}


img_t img_program_convolve(img_t input, int argc, char** argv){

    uint32_t mode = 0;
    if(argc > 1){
        if(strcmp(argv[1],"crop")) mode = 0;
        else if(strcmp(argv[1],"wrap")) mode = 1;
        else if(strcmp(argv[1],"zero")) mode = 2;
        else if(strcmp(argv[1],"repeat")) mode = 3;
        else if(strcmp(argv[1],"extend")) mode = 4;
    }

    uint32_t kernel_width = argc > 2 ? atoi(argv[2]) : 5;
    uint32_t kernel_height = argc > 3 ? atoi(argv[3]) : 5;

    float kernel_5x5[] = {
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1
    };

    for(size_t i = 4; i < argc; i++){
        if(argc > i) kernel_5x5[i] = atof(argv[i]);
    }

    uint32_t normalize = 1;
    img_t out = img_create_zero(input.width,input.height,input.depth,input.channels);
    for(int iy = 0; iy < input.height; iy++){
        for(int ix = 0; ix < input.width; ix++){
            float accumulated[4] = {0};
            float totalWeight = 0;

            for(int kx = 0; kx < kernel_width; kx++){
                for(int ky = 0; ky < kernel_height; ky++){
                    int kxo = kx-kernel_width/2;
                    int kyo = ky-kernel_height/2;

                    int x = ix+kxo;
                    int y = iy+kyo;

                    size_t fi = kx+ky*kernel_width;
                    float w = kernel_5x5[fi];
                    
                    if(x < 0 || x >= input.width || y < 0 || y >= input.height){
                        if(mode == 0){
                            continue;
                        }else if(mode == 1){
                            x = abs(x) % (2*input.width);
                            if(x >= input.width) x = 2*input.width - x - 1;
                            y = abs(y) % (2*input.height);
                            if(y >= input.height) y = 2*input.height - y - 1;
                        }else if(mode == 2){
                            totalWeight += w;
                            continue;
                        }else if(mode == 3){
                            x = ((x % input.width)+input.width) % input.width;
                            y = ((y % input.height)+input.height) % input.height;
                        }else if(mode == 4){
                            x = (int)fmin(fmax(x,0),input.width-1);
                            y = (int)fmin(fmax(y,0),input.width-1);
                        }
                    }

                    size_t i = (x+y*input.width)*input.channels;
                    for(size_t j = 0; j < input.channels; j++){
                        float sample = input.memory[i+j];
                        accumulated[j] += sample * w;
                    }

                    totalWeight += w;
                }
            }

            size_t i = (ix+iy*input.width)*input.channels;
            for(size_t j = 0; j < input.channels; j++){
                if(normalize == 1) accumulated[j] /= totalWeight;
                out.memory[i+j] = accumulated[j];
            }
        }
    }

    return out;
}

img_t img_program_minmax(img_t input, int argc, char** argv){
    uint32_t mode = 0;
    if(argc > 1){
        if(strcmp(argv[1],"erode")) mode = 0;
        else if(strcmp(argv[1],"dilate")) mode = 1;
    }

    uint32_t kernel_width = argc > 2 ? atoi(argv[2]) : 3;
    uint32_t kernel_height = argc > 3 ? atoi(argv[3]) : 3;

    float kernel_3x3[] = {
        0,1,0,
        1,1,1,
        0,1,0
    };

    for(size_t i = 4; i < argc; i++){
        if(argc > i) kernel_3x3[i] = atof(argv[i]);
    }

    const float coeff[] = {.2126,.7152,.0722,0.}; // rgba
    const size_t coeff_n = sizeof(coeff)/sizeof(coeff[0]);

    img_t out = img_create_zero(input.width,input.height,input.depth,input.channels);
    for(int iy = 0; iy < input.height; iy++){
        for(int ix = 0; ix < input.width; ix++){
            float minmaxValue = (float)mode;
            float minmaxColour[4] = {0};

            for(int kx = 0; kx < kernel_width; kx++){
                for(int ky = 0; ky < kernel_height; ky++){
                    int kxo = kx-kernel_width/2;
                    int kyo = ky-kernel_height/2;

                    int x = ix+kxo;
                    int y = iy+kyo;

                    size_t fi = kx+ky*kernel_width;
                    if(kernel_3x3[fi] == 0.) continue;
                    if(x < 0 || x >= input.width || y < 0 || y >= input.height)continue;

                    float grey = 0.;

                    size_t i = (x+y*input.width)*input.channels;

                    for(size_t k = 0; k < coeff_n; k++){
                        uint32_t ch = (k < input.channels) ? k : input.channels-1;
                        grey += input.memory[i+ch] * coeff[k];
                    }

                    if(mode == 0 && grey >= minmaxValue){
                        minmaxValue = grey;
                        for(size_t j = 0; j < input.channels; j++) minmaxColour[j] = input.memory[i+j];
                    }
                    if(mode == 1 && grey <= minmaxValue){
                        minmaxValue = grey;
                        for(size_t j = 0; j < input.channels; j++) minmaxColour[j] = input.memory[i+j];
                    }
                }
            }

            size_t i = (ix+iy*input.width)*input.channels;
            for(size_t j = 0; j < input.channels; j++){
                out.memory[i+j] = minmaxColour[j];
            }
        }
    }

    return out;
}

img_t img_program_compound(img_t input, int argc, char** argv){
    uint32_t mode = 0;
    if(argc > 1){
        if(strcmp(argv[1],"open")) mode = 0;
        else if(strcmp(argv[1],"close")) mode = 1;
    }

    uint32_t kernel_width = argc > 2 ? atoi(argv[2]) : 3;
    uint32_t kernel_height = argc > 3 ? atoi(argv[3]) : 3;

    float kernel_3x3[] = {
        0,1,0,
        1,1,1,
        0,1,0
    };

    for(size_t i = 4; i < argc; i++){
        if(argc > i) kernel_3x3[i] = atof(argv[i]);
    }

    char* modes[] = {"dilate","erode"};
    char* args1[] = {"minmax",modes[mode]};
    img_t first = img_program_minmax(input,2,args1);
    char* args2[] = {"minmax",modes[1-mode]};
    img_t second = img_program_minmax(first,2,args2);
    img_destroy(&first);


    return second;
}*/

// gpu programs

/*img_t img_program_gpu_greyscale(img_t input, int argc, char** argv){
    img_t image_output = img_create_zero(input.width,input.height,input.depth,input.channels);

    img_gpu_t gpu = img_gpu_init();
    size_t gpu_image_input = img_gpu_allocate_image(&gpu, 0, input.width,input.height,input.depth,input.channels);
    size_t gpu_image_output = img_gpu_allocate_image(&gpu, 1, input.width,input.height,input.depth,input.channels);
    size_t upload = img_gpu_upload(&gpu,gpu_image_input,input.memory,img_get_size(&input));
    size_t download = img_gpu_download(&gpu,gpu_image_output,image_output.memory,img_get_size(&input)); // same size, same buffer
    img_gpu_program_t greyscale = img_gpu_load_program_glsl(&gpu,"glsl/greyscale.comp",8,8,1);

    uint32_t uniform_input_image = 0; // push constant
    uint32_t uniform_output_image = 1;

    size_t stage_1 = img_gpu_add_stage(&gpu,&greyscale,input.width,input.height,input.depth);
    img_gpu_add_stage_data(&gpu,stage_1,&uniform_input_image,sizeof(uniform_input_image));
    img_gpu_add_stage_data(&gpu,stage_1, &uniform_output_image, sizeof(uniform_output_image));

    img_gpu_dispatch(&gpu);

    return image_output;
}

img_t img_program_gpu_brightness(img_t input, int argc, char** argv){
    img_t image_output = img_create_zero(input.width,input.height,input.depth,input.channels);

    img_gpu_t gpu = img_gpu_init();
    size_t gpu_image_input = img_gpu_allocate_image(&gpu, 0, input.width,input.height,input.depth,input.channels);
    size_t gpu_image_output = img_gpu_allocate_image(&gpu, 1, input.width,input.height,input.depth,input.channels);
    size_t upload = img_gpu_upload(&gpu,gpu_image_input,input.memory,img_get_size(&input));
    size_t download = img_gpu_download(&gpu,gpu_image_output,image_output.memory,img_get_size(&input)); // same size, same buffer
    img_gpu_program_t greyscale = img_gpu_load_program_glsl(&gpu,"glsl/brightness.comp",8,8,1);

    uint32_t uniform_input_image = 0; // push constant
    uint32_t uniform_output_image = 1;
    float uniform_brightness = argc > 0 ? atof(argv[1]): 0.;

    size_t stage_1 = img_gpu_add_stage(&gpu,&greyscale,input.width,input.height,input.depth);
    img_gpu_add_stage_data(&gpu,stage_1,&uniform_input_image,sizeof(uniform_input_image));
    img_gpu_add_stage_data(&gpu,stage_1, &uniform_output_image, sizeof(uniform_output_image));
    img_gpu_add_stage_data(&gpu,stage_1, &uniform_brightness, sizeof(uniform_brightness));

    img_gpu_dispatch(&gpu);

    return image_output;
}

img_t img_program_gpu_downscale(img_t input, int argc, char** argv){
    float scale = argc > 0 ? atof(argv[1]) : .5;
    img_t image_output = img_create_zero(ceil(input.width*scale),ceil(input.height*scale),input.depth,input.channels);

    img_gpu_t gpu = img_gpu_init();
    size_t gpu_image_input = img_gpu_allocate_image(&gpu, 0, input.width,input.height,input.depth,input.channels);
    size_t gpu_image_output = img_gpu_allocate_image(&gpu, 1, image_output.width,image_output.height,image_output.depth,image_output.channels);
    size_t upload = img_gpu_upload(&gpu,gpu_image_input,input.memory,img_get_size(&input));
    size_t download = img_gpu_download(&gpu,gpu_image_output,image_output.memory,img_get_size(&image_output)); // same size, same buffer
    img_gpu_program_t greyscale = img_gpu_load_program_glsl(&gpu,"glsl/downscale.comp",8,8,1);

    uint32_t uniform_input_image = 0; // push constant
    uint32_t uniform_output_image = 1;
    float uniform_scale = scale;

    size_t stage_1 = img_gpu_add_stage(&gpu,&greyscale,image_output.width,image_output.height,image_output.depth);
    img_gpu_add_stage_data(&gpu,stage_1,&uniform_input_image,sizeof(uniform_input_image));
    img_gpu_add_stage_data(&gpu,stage_1, &uniform_output_image, sizeof(uniform_output_image));
    img_gpu_add_stage_data(&gpu,stage_1, &uniform_scale, sizeof(uniform_scale));

    img_gpu_dispatch(&gpu);

    return image_output;
}

img_t img_program_gpu_convolve(img_t input, int argc, char** argv){
    img_t image_output = img_create_zero(input.width,input.height,input.depth,input.channels);

    img_gpu_t gpu = img_gpu_init();
    size_t gpu_image_input = img_gpu_allocate_image(&gpu, 0, input.width,input.height,input.depth,input.channels);
    size_t gpu_image_output = img_gpu_allocate_image(&gpu, 1, input.width,input.height,input.depth,input.channels);
    size_t upload = img_gpu_upload(&gpu,gpu_image_input,input.memory,img_get_size(&input));
    size_t download = img_gpu_download(&gpu,gpu_image_output,image_output.memory,img_get_size(&input)); // same size, same buffer
    img_gpu_program_t convolve = img_gpu_load_program_glsl(&gpu,"glsl/kernel.comp",8,8,1);

    uint32_t uniform_input_image = 0; // push constant
    uint32_t uniform_output_image = 1;
    uint32_t kernel_width = argc > 1 ? atoi(argv[1]) : 5;
    uint32_t kernel_height = argc > 2 ? atoi(argv[2]) : 5;

    float kernel_5x5[] = {
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1
    };

    for(size_t i = 3; i < argc; i++){
        if(argc > i) kernel_5x5[i] = atof(argv[i]);
    }

    size_t stage_1 = img_gpu_add_stage(&gpu,&convolve,input.width,input.height,input.depth);
    img_gpu_add_stage_data(&gpu,stage_1,&uniform_input_image,sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &uniform_output_image, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &kernel_width, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &kernel_height, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, kernel_5x5, sizeof(kernel_5x5));

    img_gpu_dispatch(&gpu);

    return image_output;
}

img_t img_program_gpu_minmax(img_t input, int argc, char** argv){
    img_t image_output = img_create_zero(input.width,input.height,input.depth,input.channels);

    img_gpu_t gpu = img_gpu_init();
    size_t gpu_image_input = img_gpu_allocate_image(&gpu, 0, input.width,input.height,input.depth,input.channels);
    size_t gpu_image_output = img_gpu_allocate_image(&gpu, 1, input.width,input.height,input.depth,input.channels);
    size_t upload = img_gpu_upload(&gpu,gpu_image_input,input.memory,img_get_size(&input));
    size_t download = img_gpu_download(&gpu,gpu_image_output,image_output.memory,img_get_size(&input)); // same size, same buffer
    img_gpu_program_t convolve = img_gpu_load_program_glsl(&gpu,"glsl/minmax.comp",8,8,1);

    uint32_t mode = 0;
    if(argc > 1){
        if(strcmp(argv[1],"erode")) mode = 0;
        else if(strcmp(argv[1],"dilate")) mode = 1;
    }

    uint32_t uniform_input_image = 0; // push constant
    uint32_t uniform_output_image = 1;
    uint32_t kernel_width = argc > 2 ? atoi(argv[2]) : 5;
    uint32_t kernel_height = argc > 3 ? atoi(argv[3]) : 5;

    float kernel_5x5[] = {
        0,0,0,0,0,
        0,1,1,1,0,
        0,1,1,1,0,
        0,1,1,1,0,
        0,0,0,0,0
    };

    for(size_t i = 3; i < argc; i++){
        if(argc > i) kernel_5x5[i] = atof(argv[i]);
    }

    size_t stage_1 = img_gpu_add_stage(&gpu,&convolve,input.width,input.height,input.depth);
    img_gpu_add_stage_data(&gpu,stage_1,&uniform_input_image,sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &uniform_output_image, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &mode, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &kernel_width, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &kernel_height, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, kernel_5x5, sizeof(kernel_5x5));

    img_gpu_dispatch(&gpu);

    return image_output;
}

img_t img_program_gpu_compound(img_t input, int argc, char** argv){
    img_t image_output = img_create_zero(input.width,input.height,input.depth,input.channels);

    img_gpu_t gpu = img_gpu_init();
    size_t gpu_image_input = img_gpu_allocate_image(&gpu, 0, input.width,input.height,input.depth,input.channels);
    size_t gpu_image_output = img_gpu_allocate_image(&gpu, 1, input.width,input.height,input.depth,input.channels);
    size_t upload = img_gpu_upload(&gpu,gpu_image_input,input.memory,img_get_size(&input));
    size_t download = img_gpu_download(&gpu,gpu_image_input,image_output.memory,img_get_size(&input)); // same size, same buffer
    img_gpu_program_t convolve = img_gpu_load_program_glsl(&gpu,"glsl/minmax.comp",8,8,1);

    uint32_t mode = 0;
    if(argc > 1){
        if(strcmp(argv[1],"open")) mode = 0;
        else if(strcmp(argv[1],"close")) mode = 1;
    }

    uint32_t altmode = 1-mode;

    uint32_t uniform_input_image = 0; // push constant
    uint32_t uniform_output_image = 1;
    uint32_t kernel_width = argc > 2 ? atoi(argv[2]) : 5;
    uint32_t kernel_height = argc > 3 ? atoi(argv[3]) : 5;

    float kernel_5x5[] = {
        0,0,0,0,0,
        0,1,1,1,0,
        0,1,1,1,0,
        0,1,1,1,0,
        0,0,0,0,0
    };

    for(size_t i = 3; i < argc; i++){
        if(argc > i) kernel_5x5[i] = atof(argv[i]);
    }

    size_t stage_1 = img_gpu_add_stage(&gpu,&convolve,input.width,input.height,input.depth);
    img_gpu_add_stage_data(&gpu,stage_1,&uniform_input_image,sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &uniform_output_image, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &mode, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &kernel_width, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, &kernel_height, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_1, kernel_5x5, sizeof(kernel_5x5));

    size_t stage_2 = img_gpu_add_stage(&gpu,&convolve,input.width,input.height,input.depth);
    img_gpu_add_stage_data(&gpu,stage_2,&uniform_output_image,sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_2, &uniform_input_image, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_2, &altmode, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_2, &kernel_width, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_2, &kernel_height, sizeof(uint32_t));
    img_gpu_add_stage_data(&gpu,stage_2, kernel_5x5, sizeof(kernel_5x5)); // ping pong buffer

    img_gpu_dispatch(&gpu);

    return image_output;
}

img_t img_program_gpu_hog(img_t input, int argc, char** argv){

    uint32_t window_w = 64;
    uint32_t window_h = 128;
    uint32_t workgroup = 8;
    uint32_t block = 2;
    uint32_t stride = 1;


    img_t image_output = img_create_zero(input.width,input.height,input.depth,1);

    img_gpu_t gpu = img_gpu_init();
    uint32_t uniform_input_image = 0;
    uint32_t uniform_output_image = 1;
    size_t gpu_image_input = img_gpu_allocate_image(&gpu, uniform_input_image, input.width,input.height,input.depth,input.channels);
    size_t gpu_image_output = img_gpu_allocate_image(&gpu, uniform_output_image, input.width,input.height,input.depth,input.channels);
    size_t upload = img_gpu_upload(&gpu,gpu_image_input,input.memory,img_get_size(&input));

    // greyscale step

    img_gpu_program_t greyscale = img_gpu_load_program_glsl(&gpu,"glsl/greyscale.comp",workgroup,workgroup,1);
    size_t stage_greyscale = img_gpu_add_stage(&gpu,&greyscale,input.width,input.height,input.depth);
    img_gpu_add_stage_data(&gpu,stage_greyscale,&uniform_input_image,sizeof(uniform_input_image));
    img_gpu_add_stage_data(&gpu,stage_greyscale, &uniform_output_image, sizeof(uniform_output_image));

    // sobel filter step

    img_gpu_program_t sobel = img_gpu_load_program_glsl(&gpu,"glsl/sobel.comp",workgroup,workgroup,1);
    float kernel_sobel_x[] = {
        -1,0,1,
        -2,0,2,
        -1,0,1
    };
    float kernel_sobel_y[] = {
        -1,-2,-1,
        0,0,0,
        1,2,1
    };

    size_t stage_sobel = img_gpu_add_stage(&gpu,&sobel,input.width,input.height,input.depth);
    img_gpu_add_stage_data(&gpu,stage_sobel,&uniform_output_image,sizeof(uniform_output_image));
    img_gpu_add_stage_data(&gpu,stage_sobel,&uniform_input_image,sizeof(uniform_input_image));
    img_gpu_add_stage_data(&gpu,stage_sobel, kernel_sobel_x, sizeof(kernel_sobel_x));
    img_gpu_add_stage_data(&gpu,stage_sobel, kernel_sobel_y, sizeof(kernel_sobel_y));

    img_gpu_program_t hog = img_gpu_load_program_glsl(&gpu,"glsl/hog.comp",1,1,1);
    uint32_t cellsize = 8;
    uint32_t bins = 9;
    uint32_t cell_width = input.width / cellsize;
    uint32_t cell_height = input.height / cellsize;
    uint32_t uniform_histogram_image = 2;
    size_t gpu_image_histogram = img_gpu_allocate_image(&gpu, uniform_histogram_image, cell_width, cell_height, 9 ,1);
    size_t stage_hog = img_gpu_add_stage(&gpu,&hog,cell_width, cell_height ,1);
    img_gpu_add_stage_data(&gpu,stage_hog,&uniform_input_image,sizeof(uniform_input_image));
    img_gpu_add_stage_data(&gpu,stage_hog,&uniform_histogram_image,sizeof(uniform_histogram_image));
    img_gpu_add_stage_data(&gpu,stage_hog,&cellsize,sizeof(cellsize));

    uint32_t uniform_block_image = 3;
    uint32_t block_width = cell_width - 1;
    uint32_t block_height = cell_height - 1;
    img_gpu_program_t hog_normalize = img_gpu_load_program_glsl(&gpu,"glsl/normalize.comp",1,1,1);
    size_t gpu_image_blocks = img_gpu_allocate_image(&gpu, uniform_block_image, block_width, block_height, 36, 1);
    size_t stage_blocks = img_gpu_add_stage(&gpu,&hog_normalize,block_width, block_height,1);
    img_gpu_add_stage_data(&gpu,stage_blocks,&uniform_histogram_image,sizeof(uniform_histogram_image));
    img_gpu_add_stage_data(&gpu,stage_blocks,&uniform_block_image,sizeof(uniform_block_image));

    uint32_t uniform_vector_image = 4;
    uint32_t window_width = window_w / cellsize - 1;
    uint32_t window_height = window_h / cellsize - 1;
    uint32_t feature_length = window_width * window_height * 36;
    uint32_t dispatch_width = block_width - window_width;
    uint32_t dispatch_height = block_height - window_height; 
    img_gpu_program_t hog_window = img_gpu_load_program_glsl(&gpu,"glsl/window.comp",1,1,1);
    size_t gpu_image_vectors = img_gpu_allocate_image(&gpu,uniform_vector_image,dispatch_width,dispatch_height,feature_length,1);
    //size_t gpu_buffer_vectors = img_gpu_allocate_buffer(&gpu,0,dispatch_width * dispatch_height * feature_length

    size_t download = img_gpu_download(&gpu,gpu_image_histogram,image_output.memory,img_get_size(&image_output)); // same size, same buffer



    img_gpu_dispatch(&gpu);

    return image_output;
}*/