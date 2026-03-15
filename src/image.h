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

typedef struct {
    float* memory;
    uint32_t width, height, depth, channels; // fuckin 24 bytes
} img_t;

void img_alloc(img_t* img);
img_t img_create_fill(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels, float* fill);
img_t img_create_zero(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels);
void img_destroy(img_t* img);
int img_validate(img_t* img);
size_t img_get_size(img_t* img);

img_t img_create_from_image(const char* file, uint32_t channels);
void img_write_as_image(img_t* img, const char* file);
img_t img_create_from_binary(const char* file);
void img_write_as_binary(img_t* img, const char* file);

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

    struct {
        img_gpu_buffer_t buffer[IMG_GPU_MAX_BUFFERS];
        size_t count;
    } device;

    struct {
        img_gpu_buffer_t buffer[IMG_GPU_MAX_BUFFERS];
        size_t device_ptr[IMG_GPU_MAX_BUFFERS];

        void* host_ptr[IMG_GPU_MAX_BUFFERS];
        //size_t copy_size[IMG_GPU_MAX_BUFFERS];

        size_t count;
    } host;

    struct {
        img_gpu_pass_t pass[IMG_GPU_MAX_BUFFERS];
        size_t count;
    } stages;
} img_gpu_t;

static const vkr_descriptor_info img_gpu_descriptors[] = {
    {IMG_GPU_TYPE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, MAX_TEXTURES},
    {IMG_GPU_TYPE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, MAX_TEXTURES}
};

img_gpu_t img_gpu_init();
img_gpu_program_t img_gpu_load_program_glsl(img_gpu_t* gpu, const char* shader_file, uint32_t width, uint32_t height, uint32_t depth); // good candidate for reflection
size_t img_gpu_allocate_image(img_gpu_t* gpu, uint32_t binding, uint32_t width, uint32_t height, uint32_t depth, uint32_t channels);
size_t img_gpu_allocate_buffer(img_gpu_t* gpu, uint32_t binding, size_t size);
size_t img_gpu_upload(img_gpu_t* gpu, size_t dest, void* src, size_t size);
size_t img_gpu_download(img_gpu_t* gpu, size_t src, void* dest, size_t size);
void img_gpu_free(img_gpu_t* gpu, img_gpu_buffer_t* buffer);

size_t img_gpu_add_stage(img_gpu_t* gpu, img_gpu_program_t* program, uint32_t width, uint32_t height, uint32_t depth);
void img_gpu_add_stage_data(img_gpu_t* gpu, size_t pass, void* data, size_t size);
void img_gpu_dispatch(img_gpu_t* gpu);

// programs

typedef struct {
    char* name;
    img_t (*program)(img_t input, int argc, char** argv);
    uint32_t channel_override;
} img_program_t;

img_t img_program_greyscale(img_t input, int argc, char** argv);
img_t img_program_brightness(img_t input, int argc, char** argv);
img_t img_program_clamp(img_t input, int argc, char** argv);
img_t img_program_window(img_t input, int argc, char** argv);
img_t img_program_histogram(img_t input, int argc, char** argv);
img_t img_program_histogram_rgb(img_t input, int argc, char** argv);
img_t img_program_otsu(img_t input, int argc, char** argv);

img_t img_program_gpu_greyscale(img_t input, int argc, char** argv);
img_t img_program_gpu_brightness(img_t input, int argc, char** argv);
img_t img_program_gpu_downscale(img_t input, int argc, char** argv);


static const img_program_t img_program_table[] = {
    {"greyscale", img_program_greyscale,0},
    {"brightness", img_program_brightness,0},
    {"clamp", img_program_clamp,0},
    {"window", img_program_window,0},
    {"histogram", img_program_histogram,0},
    {"histogram_rgb", img_program_histogram_rgb,0},
    {"otsu", img_program_otsu,0},

    {"greyscale_gpu",img_program_gpu_greyscale,4},
    {"brightness_gpu",img_program_gpu_brightness,4},
    {"downscale",img_program_gpu_downscale,4}
};

#endif // image_h