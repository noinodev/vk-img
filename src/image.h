#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include <shaderc/shaderc.h>

#include "vkr.h"

// data

enum{
    IMG_TYPE_BYTE,
    IMG_TYPE_SHORT,
    IMG_TYPE_INT,
    IMG_TYPE_FLOAT,
    IMG_TYPE_DOUBLE
};

typedef struct {
    unsigned char* memory;
    uint32_t width, height, depth; // fuckin 24 bytes
    uint8_t channels;
    uint8_t type;
    uint8_t meta[2];
} img_t;

enum {
    ATLAS_SERIAL_COUNT,

    ATLAS_SERIAL_STRIDE_META,
    ATLAS_SERIAL_STRIDE_IMAGE,
    ATLAS_SERIAL_STRIDE_TOTAL,

    ATLAS_SERIAL_IMAGE_WIDTH,
    ATLAS_SERIAL_IMAGE_HEIGHT,
    ATLAS_SERIAL_IMAGE_DEPTH,
    ATLAS_SERIAL_IMAGE_CHANNELS,

    ATLAS_SERIAL_HEADER_SIZE = 16
};

typedef struct {
    unsigned char* memory; // atlas memory
    unsigned char* labels;

    size_t start;
    size_t size;

    uint32_t count,label_stride,image_stride,total_stride;
    uint32_t width, height, depth, channels;
} atlas_t;

void img_alloc(img_t* img);
img_t img_create(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels, uint32_t type);
void img_fill(img_t* img, double r, double g, double b, double a);
void img_destroy(img_t* img);
int img_validate(img_t* img);
size_t img_get_size(img_t* img);
size_t img_get_stride(img_t* img);
void img_remap(img_t* src, img_t* dst);

img_t img_create_from_image(const char* file, uint32_t channels);
void img_write_as_image(img_t* img, const char* file);
img_t img_create_from_binary(const char* file);
img_t img_create_from_binary_raw(const char* file);
void img_write_as_binary(img_t* img, const char* file);
void img_write_as_binary_raw(img_t* img, const char* file, const char* mode);
atlas_t img_create_atlas_from_binary(const char* file, size_t bytes, size_t block);
void img_destroy_atlas(atlas_t* atlas);
img_t img_create_from_atlas(atlas_t* atlas, uint32_t index);
void img_copy_atlas_batch(atlas_t* atlas, img_t* image, uint32_t index, uint32_t count);

// gpu

#define IMG_GPU_PUSH_ALIGNMENT sizeof(uint32_t)
#define IMG_GPU_PUSH_MAX 128
#define IMG_GPU_MAX_PASSES 32
#define IMG_GPU_MAX_BUFFERS 256

typedef enum {
    IMG_GPU_TYPE_NONE,
    IMG_GPU_TYPE_IMAGE,
    IMG_GPU_TYPE_BUFFER,
    IMG_GPU_TYPE_STAGING,

    IMG_GPU_TYPE_COUNT
} img_gpu_type_t;

typedef struct {
    img_gpu_type_t type; // finish thius??

    union {
        vkr_texture image;
        vkr_buffer buffer;
    };
} img_gpu_buffer_t;

typedef struct {
    VkPipeline pipeline;
    VkExtent3D workgroup;
} img_gpu_program_t;

typedef struct {
    img_gpu_program_t* program;
    uint32_t width, height, depth;
    uint8_t push_data[IMG_GPU_PUSH_MAX];
    uint32_t push_size;
} img_gpu_pass_t;

typedef struct {
    vkr_state vkr;
    VkFence fence;
    VkCommandBuffer cmd;
    uint32_t cmd_sealed;

    struct {
        img_gpu_buffer_t buffer[IMG_GPU_MAX_BUFFERS];
        size_t count;
    } device;

    struct {
        img_gpu_buffer_t buffer[IMG_GPU_MAX_BUFFERS];
        size_t device_ptr[IMG_GPU_MAX_BUFFERS];

        void* host_ptr[IMG_GPU_MAX_BUFFERS];
        void* mapped_ptr[IMG_GPU_MAX_BUFFERS];

        size_t count;
    } host;

    struct {
        img_gpu_pass_t pass[IMG_GPU_MAX_BUFFERS];
        size_t count;
    } stages;
} img_gpu_t;

img_gpu_t img_gpu_init();
img_gpu_program_t img_gpu_load_program_glsl(img_gpu_t* gpu, const char* shader_file, uint32_t width, uint32_t height, uint32_t depth); // good candidate for reflection
size_t img_gpu_allocate_image(img_gpu_t* gpu, uint32_t binding, uint32_t width, uint32_t height, uint32_t depth, uint32_t channels);
size_t img_gpu_allocate_buffer(img_gpu_t* gpu, uint32_t binding, size_t size);
//void img_gpu_bind_buffer(img_gpu_t* gpu, size_t index);
size_t img_gpu_upload_now(img_gpu_t* gpu, size_t dest, void* src, size_t size);
size_t img_gpu_upload(img_gpu_t* gpu, size_t dest, void* src, size_t size);
void img_gpu_map_host_buffer(img_gpu_t* gpu, size_t index, void* src); // reupload
size_t img_gpu_download_now(img_gpu_t* gpu, size_t src, void* dest, size_t size);
size_t img_gpu_download(img_gpu_t* gpu, size_t src, void* dest, size_t size);
void img_gpu_map_device_buffer(img_gpu_t* gpu, size_t index, void* dest); // redownload
void img_gpu_free(img_gpu_t* gpu, img_gpu_buffer_t* buffer);

size_t img_gpu_add_stage(img_gpu_t* gpu, img_gpu_program_t* program, uint32_t width, uint32_t height, uint32_t depth);
void img_gpu_add_stage_data(img_gpu_t* gpu, size_t pass, void* data, size_t size);
void img_gpu_set_stage_data(img_gpu_t* gpu, size_t pass, void* data, size_t size, size_t offset);
float img_gpu_dispatch(img_gpu_t* gpu);
void img_gpu_reset(img_gpu_t* gpu);

#endif // image_h