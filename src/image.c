#include "image.h"

void img_alloc(img_t* img){
    assert(img);
    assert(img->width > 0);
    assert(img->height > 0);
    assert(img->depth > 0);
    assert(img->channels > 0);

    img->memory = (float*)calloc(
        img->width * img->height * img->depth * img->channels,
        sizeof(float)
    );

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

img_t img_create_zero(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels){
    img_t out = {0};
    out.width = width;
    out.height = height;
    out.depth = depth;
    out.channels = channels;

    img_alloc(&out);
    printf("img_create_zero\n");
    return out;
}

img_t img_create_fill(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels, float* fill){
    img_t out = img_create_zero(width,height,depth,channels);

    for(uint32_t i = 0; i < width*height*depth; i++){
        memcpy(&out.memory[i*channels],fill,sizeof(float)*channels); // not ideal lol
        //if(i%10==0)printf("%d ",out.memory[4*i]);
    }
    printf("img_create_fill\n");
    return out;
}

img_t img_create_from_image(const char* file, uint32_t channels){
    int w,h,c;
    stbi_uc* pixels = stbi_load(file, &w, &h, &c, channels);
    if(channels != 0) c = channels;

    img_t out = img_create_zero(w,h,1,c);
    for(uint32_t i = 0; i < w*h*c; i++){
        out.memory[i] = ((float)pixels[i])/255.; // NORMALIZE !
        //if(i%10==0)printf("%f ",out.memory[i]);
    }

    stbi_image_free(pixels);
    printf("img_create_from_image\n");
    return out;
}

void img_write_as_image(img_t* img, const char* file){
    printf("img_write_as_image '%s' \n",file);

    if(img->depth != 1){
        printf("stb only supports 2d image writes\n");
        exit(-1);
    }
    unsigned char* pixels = malloc(img->width*img->height*img->depth*img->channels*sizeof(unsigned char));
    for(uint32_t i = 0; i < img->width*img->height*img->depth*img->channels; i++){
        pixels[i] = fmin(fmax(img->memory[i],0.),1.)*255;
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

void img_write_as_binary(img_t* img, const char* file){
    FILE* f = fopen(file,"w");
    fwrite(img,sizeof(img),1,f);
    fwrite(img->memory, sizeof(float), img->width*img->height*img->depth*img->channels, f);
    fclose(f);
}

// processors

img_t img_program_greyscale(img_t input, int argc, char** argv){
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

img_t img_program_histogram(img_t input, int argc, char** argv){
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
    printf("]");

    free(histogram);

    return out;
}