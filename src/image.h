#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdarg.h>

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

img_t img_create_from_image(const char* file, uint32_t channels);
void img_write_as_image(img_t* img, const char* file);
img_t img_create_from_binary(const char* file);
void img_write_as_binary(img_t* img, const char* file);

// programs

typedef struct {
    char* name;
    img_t (*program)(img_t input, int argc, char** argv);
} img_program_t;

img_t img_program_greyscale(img_t input, int argc, char** argv);
img_t img_program_brightness(img_t input, int argc, char** argv);
img_t img_program_clamp(img_t input, int argc, char** argv);
img_t img_program_window(img_t input, int argc, char** argv);
img_t img_program_histogram(img_t input, int argc, char** argv);

static const img_program_t img_program_table[] = {
    {"greyscale", img_program_greyscale},
    {"brightness", img_program_brightness},
    {"clamp", img_program_clamp},
    {"window", img_program_window},
    {"histogram", img_program_histogram}
};

#endif // image_h