#include "image-lua.h"
#include "image.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include <math.h>
#include <stdlib.h>

int lua_img_fill(lua_State* L){
    img_t* img = lua_touserdata(L,1);
    size_t channels = img->channels;

    double col[4] = {0};
    for(int i = 0; i < channels; i++) col[i] = (double)luaL_checknumber(L,2+i);

    img_fill(img,col[0],col[1],col[2],col[3]);
    return 1;
}

int lua_img_create(lua_State* L){
    int w = luaL_checkinteger(L,1);
    int h = luaL_checkinteger(L,2);
    int d = luaL_checkinteger(L,3);
    int c = luaL_checkinteger(L,4);
    int t = luaL_checkinteger(L,5);

    img_t* image = lua_newuserdata(L,sizeof(img_t));
    *image = img_create(w,h,d,c,t);
    return 1;
}

int lua_img_destroy(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    img_destroy(image);
    return 0;
}

/*int lua_img_validate(lua_State* L){

}*/

/*int lua_img_get_size(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    lua_pushinteger(L,img_get_size(image));
    return 1;
}

int lua_img_width(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    lua_pushinteger(L, image->width);
    return 1;
}
int lua_img_height(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    lua_pushinteger(L, image->height);
    return 1;
}
int lua_img_depth(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    lua_pushinteger(L, image->depth);
    return 1;
}
int lua_img_channels(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    lua_pushinteger(L, image->channels);
    return 1;
}*/

int lua_img_dim(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    lua_pushinteger(L,image->width);
    lua_pushinteger(L,image->height);
    lua_pushinteger(L,image->depth);
    lua_pushinteger(L,image->channels);
    printf("dim: %zu %zu %zu %zu\n",image->width,image->height,image->depth,image->channels);
    return 4;
}

int lua_img_meta(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    lua_pushinteger(L,image->meta[0]);
    lua_pushinteger(L,image->meta[1]);
    return 2;
}

int lua_img_remap(lua_State* L){
    img_t* src = lua_touserdata(L,1);
    img_t* dst = lua_touserdata(L,2);
    img_remap(src,dst);
    return 0;
}

int lua_img_find_max_1d(lua_State* L){
    img_t* img = lua_touserdata(L, 1);
    size_t true_label = luaL_checkinteger(L, 2); // Pass the true label from Lua
    
    float max = -1;
    size_t predicted_x = 0;
    float true_label_prob = 0.0f;

    for(size_t i = 0; i < img->width; i++){
        float val = ((float*)img->memory)[i];
        
        // Track the network's highest prediction
        if(val > max){
            predicted_x = i;
            max = val;
        }
        
        // Grab the probability of the actual correct target
        if(i == true_label) {
            true_label_prob = val;
        }
    }
    
    lua_pushinteger(L, predicted_x);
    lua_pushnumber(L, max);
    lua_pushnumber(L, true_label_prob); // Now returning 3 values
    return 3;
}

/*int lua_img_get_float(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    int x = luaL_checkinteger(L,2);

    lua_pushnumber(L,((float)((uint8_t*)image->memory)[x])/255.f);
    return 1;
}*/

int lua_img_create_from_image(lua_State* L){
    char* file = luaL_checkstring(L,1);

    img_t* image = lua_newuserdata(L,sizeof(img_t));
    *image = img_create_from_image(file,0);
    return 1;
}

int lua_img_write_as_image(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    char* file = luaL_checkstring(L,2);

    img_write_as_image(image,file);
    return 0;
}

int lua_img_create_from_binary(lua_State* L){
    char* file = luaL_checkstring(L,1);

    img_t* image = lua_newuserdata(L,sizeof(img_t));
    *image = img_create_from_binary(file);
    return 1;
}

int lua_img_print(lua_State* L){
    img_t* img = lua_touserdata(L,1);
    for(size_t i = 0; i < img->width; i++){
        printf("%d ",img->memory[i]);
    }
    return 0;
}

int lua_img_create_from_binary_raw(lua_State* L){
    char* file = luaL_checkstring(L,1);

    img_t* image = lua_newuserdata(L,sizeof(img_t));
    *image = img_create_from_binary_raw(file);
    return 1;
}

int lua_img_write_as_binary(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    char* file = luaL_checkstring(L,2);

    img_write_as_binary(image,file);
    return 0;
}

int lua_img_write_as_binary_raw(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    char* file = luaL_checkstring(L,2);
    char* mode = luaL_checkstring(L,3);

    img_write_as_binary_raw(image,file,mode);
    return 0;
}

int lua_img_read_from_binary_raw(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    char* file = luaL_checkstring(L,2);

    //img_write_as_binary_raw(image,file,mode);
    FILE* f = fopen(file,"r");
    fread(image->memory,sizeof(float),img_get_size(image),f);
    fclose(f);
    return 0;
}

//int lua_img_create_

int lua_img_create_atlas(lua_State* L){
    char* file = luaL_checkstring(L,1);
    size_t bytes = luaL_checkinteger(L,2);
    int index = luaL_checkinteger(L,3);

    atlas_t temp = img_create_atlas_from_binary(file,bytes,index);
    if(temp.memory == NULL){
        // EOF, or allocation failed, or some err
        lua_pushnil(L);
    }else{
        atlas_t* atlas = lua_newuserdata(L,sizeof(atlas_t));
        *atlas = temp;
    }

    return 1;
}

int lua_img_destroy_atlas(lua_State* L){
    atlas_t* atlas = lua_touserdata(L,1);
    img_destroy_atlas(atlas);
    return 0;
}

int lua_img_atlas_count(lua_State* L){
    atlas_t* atlas = lua_touserdata(L,1);
    lua_pushinteger(L,atlas->count);
    return 1;
}

int lua_img_atlas_size(lua_State* L){
    atlas_t* atlas = lua_touserdata(L,1);
    lua_pushinteger(L,atlas->size);
    return 1;
}

int lua_img_create_from_atlas(lua_State* L){
    atlas_t* atlas = lua_touserdata(L,1);
    int index = luaL_checkinteger(L,2);

    img_t* img = lua_newuserdata(L,sizeof(img_t));
    *img = img_create_from_atlas(atlas,index);
    return 1;
}

int lua_img_create_batch_from_atlas(lua_State* L){
    atlas_t* atlas = lua_touserdata(L,1);
    img_t* image = lua_touserdata(L,2);
    int index = luaL_checkinteger(L,3);
    int count = luaL_checkinteger(L,4);

    img_copy_atlas_batch(atlas,image,index,count);
    return 0;
}

int lua_gpu_init(lua_State* L){
    img_gpu_t* gpu = lua_newuserdata(L,sizeof(img_gpu_t));
    *gpu = img_gpu_init();
    return 1;
}

int lua_gpu_load_program_glsl(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    char* file = luaL_checkstring(L,2);
    int x = luaL_checkinteger(L,3);
    int y = luaL_checkinteger(L,4);
    int z = luaL_checkinteger(L,5);

    img_gpu_program_t* program = lua_newuserdata(L,sizeof(img_gpu_program_t));
    *program = img_gpu_load_program_glsl(gpu,file,x,y,z);
    return 1;
}

int lua_gpu_allocate_image(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int binding = luaL_checkinteger(L,2);
    int w = luaL_checkinteger(L,3);
    int h = luaL_checkinteger(L,4);
    int d = luaL_checkinteger(L,5);
    int c = luaL_checkinteger(L,6);

    lua_pushinteger(L,img_gpu_allocate_image(gpu,binding,w,h,d,c));
    return 1;
}

int lua_gpu_allocate_buffer(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int binding = luaL_checkinteger(L,2);
    int size = luaL_checkinteger(L,3);

    lua_pushinteger(L,img_gpu_allocate_buffer(gpu,binding,size));
    return 1;
}

int lua_gpu_upload_now(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int input = luaL_checkinteger(L,2);
    img_t* image = lua_touserdata(L,3);

    lua_pushinteger(L,img_gpu_upload_now(gpu,input,image->memory,img_get_size(image)*img_get_stride(image)));
    return 1;
}


int lua_gpu_upload(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int input = luaL_checkinteger(L,2);
    img_t* image = lua_touserdata(L,3);

    lua_pushinteger(L,img_gpu_upload(gpu,input,image->memory,img_get_size(image)*img_get_stride(image)));
    return 1;
}


int lua_gpu_map_host(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int index = luaL_checkinteger(L,2);
    img_t* image = lua_touserdata(L,3);

    img_gpu_map_host_buffer(gpu, index,image->memory);
}

int lua_gpu_download_now(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int input = luaL_checkinteger(L,2);
    img_t* image = lua_touserdata(L,3);

    lua_pushinteger(L,img_gpu_download_now(gpu,input,image->memory,img_get_size(image)*img_get_stride(image)));
    return 1;
}

int lua_gpu_download(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int input = luaL_checkinteger(L,2);
    img_t* image = lua_touserdata(L,3);

    lua_pushinteger(L,img_gpu_download(gpu,input,image->memory,img_get_size(image)*img_get_stride(image)));
    return 1;
}

int lua_gpu_map_device(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int index = luaL_checkinteger(L,2);
    img_t* image = lua_touserdata(L,3);

    img_gpu_map_device_buffer(gpu, index,image->memory);
}

/*int lua_gpu_bind_buffer(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int index = luaL_checkinteger(L,2);

    img_gpu_bind_buffer(gpu,index);
    return 0;
}*/

int lua_gpu_add_stage(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    img_gpu_program_t* program = lua_touserdata(L,2);
    int w = luaL_checkinteger(L,3);
    int h = luaL_checkinteger(L,4);
    int d = luaL_checkinteger(L,5);

    lua_pushinteger(L,img_gpu_add_stage(gpu,program,w,h,d));
    return 1;
}

int lua_gpu_add_stage_data_uint(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int stage = luaL_checkinteger(L,2);

    uint32_t val = lua_tointeger(L, 3);
    img_gpu_add_stage_data(gpu, stage, &val, sizeof(val));


    /*int type = lua_type(L, 3);
    if(type == LUA_TNUMBER){
        if(lua_isinteger(L, 3)){
            uint32_t val = lua_tointeger(L, 3);
            img_gpu_add_stage_data(gpu, stage, &val, sizeof(val));
        } else {
            float val = lua_tonumber(L, 3);
            img_gpu_add_stage_data(gpu, stage, &val, sizeof(val));
        }
    } else if(type == LUA_TTABLE){
        // array of floats, kernel
        int n = lua_rawlen(L, 3);
        float* buf = alloca(n * sizeof(float));
        for(int i = 0; i < n; i++){
            lua_rawgeti(L, 3, i + 1);
            buf[i] = lua_tonumber(L, -1);
            lua_pop(L, 1);
            printf("%.0f ",buf[i]);
        }
        printf("\n");
        img_gpu_add_stage_data(gpu, stage, buf, n * sizeof(float));
    }else{
        uint32_t val = 123456;
        printf("invalid stage data: %f\n",lua_tonumber(L, 3));
        abort();
        img_gpu_add_stage_data(gpu, stage, &val, sizeof(val));
    }*/

    return 0;
}

int lua_gpu_add_stage_data_float(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int stage = luaL_checkinteger(L,2);

    float val = lua_tonumber(L, 3);
    img_gpu_add_stage_data(gpu, stage, &val, sizeof(val));
    return 0;
}

int lua_gpu_set_stage_data(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int stage = luaL_checkinteger(L,2);

    int offset = luaL_checkinteger(L,4);

    int type = lua_type(L, 3);
    if(type == LUA_TNUMBER){
        if(lua_isinteger(L, 3)){
            uint32_t val = lua_tointeger(L, 3);
            img_gpu_set_stage_data(gpu, stage, &val, sizeof(val),offset);
        } else {
            float val = lua_tonumber(L, 3);
            img_gpu_set_stage_data(gpu, stage, &val, sizeof(val),offset);
        }
        float val = lua_tonumber(L, 3);
        //printf("modified staging data: %d -> %f\n",offset,val);
    }else{
        printf("invalid stage data for a SET can you not??\n");
        uint32_t val = 123456;
        img_gpu_set_stage_data(gpu, stage, &val, sizeof(val),offset);
    }

    return 0;
}

int lua_gpu_dispatch(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    float t = img_gpu_dispatch(gpu);
    lua_pushnumber(L,t);
    return 1;
}

int lua_gpu_reset(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    img_gpu_reset(gpu);
    return 0;
}

// special processors

int lua_img_randomize_kaiming_normal(lua_State* L){
    img_t* img = lua_touserdata(L,1);
    size_t fan_in = luaL_checkinteger(L,2);
    size_t size = luaL_checkinteger(L,3);

    float std = sqrt(2./fan_in);
    float* buf = (float*)img->memory;
    for(size_t i = 0; i < size; i++){
        float u1 = (rand()+1) / (float)(RAND_MAX+2.);
        float u2 = (rand()+1) / (float)(RAND_MAX+2.);
        float z = sqrt(-2 * log(u1)) * cos(2 * 3.141592654 * u2);
        buf[i] = z * std;
    }
    printf("kaiming fan_in: %zu, size %zu, std %f\n",fan_in,size,std);
    return 0;
}


// table

static const luaL_Reg img_lib[] = {
    {"create",lua_img_create},
    {"fill",lua_img_fill},
    {"destroy",lua_img_destroy},
    /*{"size",lua_img_get_size},
    {"width",lua_img_width},
    {"height",lua_img_height},
    {"depth",lua_img_depth},
    {"channels",lua_img_channels},
    {"get_float",lua_img_get_float},*/
    {"dim",lua_img_dim},
    {"meta",lua_img_meta},
    {"remap",lua_img_remap},
    {"create_from_image",lua_img_create_from_image},
    {"create_from_binary",lua_img_create_from_binary},
    {"create_raw",lua_img_create_from_binary_raw},
    {"write_as_image",lua_img_write_as_image},
    {"write_as_binary",lua_img_write_as_binary},
    {"write_raw",lua_img_write_as_binary_raw},
    {"read_raw",lua_img_read_from_binary_raw},
    {"create_atlas",lua_img_create_atlas},
    {"destroy_atlas",lua_img_destroy_atlas},
    {"create_from_atlas",lua_img_create_from_atlas},
    {"create_batch_from_atlas",lua_img_create_batch_from_atlas},
    {"atlas_count",lua_img_atlas_count},
    {"atlas_size",lua_img_atlas_size},

    {"print",lua_img_print},

    {"randomize_kaiming_normal",lua_img_randomize_kaiming_normal},

    {"gpu_init",            lua_gpu_init},
    {"gpu_load_program",    lua_gpu_load_program_glsl},
    {"gpu_allocate_image",  lua_gpu_allocate_image},
    {"gpu_allocate_buffer", lua_gpu_allocate_buffer},
    {"gpu_upload_now",      lua_gpu_upload_now},
    {"gpu_upload",          lua_gpu_upload},
    {"gpu_map_host",        lua_gpu_map_host},
    {"gpu_download_now",        lua_gpu_download_now},
    {"gpu_download",        lua_gpu_download},
    {"gpu_map_device",      lua_gpu_map_device},
    {"gpu_add_stage",       lua_gpu_add_stage},
    {"gpu_push_uint",       lua_gpu_add_stage_data_uint},
    {"gpu_push_float",      lua_gpu_add_stage_data_float},
    {"gpu_set_stage_data",  lua_gpu_set_stage_data},
    {"gpu_dispatch",        lua_gpu_dispatch},
    {"gpu_reset",lua_gpu_reset},

    {"class_1d",lua_img_find_max_1d},
    {NULL, NULL}
};

int luaopen_img(lua_State* L){
    luaL_newlib(L, img_lib);
    return 1;
}