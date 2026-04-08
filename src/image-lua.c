#include "image.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

/*int lua_img_alloc(lua_State* L){
    img_t* 
}*/

int lua_img_create_fill(lua_State* L){
    int w = luaL_checkinteger(L,1);
    int h = luaL_checkinteger(L,2);
    int d = luaL_checkinteger(L,3);
    int c = luaL_checkinteger(L,4);

    float col[4] = {0};
    for(int i = 0; i < c; i++) col[i] = luaL_checknumber(L,5+i);

    img_t* image = lua_newuserdata(L,sizeof(img_t));
    *image = img_create_fill(w,h,d,c,col);
    return 1;
}

int lua_img_create_zero(lua_State* L){
    int w = luaL_checkinteger(L,1);
    int h = luaL_checkinteger(L,2);
    int d = luaL_checkinteger(L,3);
    int c = luaL_checkinteger(L,4);

    img_t* image = lua_newuserdata(L,sizeof(img_t));
    *image = img_create_zero(w,h,d,c);
    return 1;
}

int lua_img_destroy(lua_State* L){
    img_t* image = lua_touserdata(L,1);
    img_destroy(image);
    return 0;
}

/*int lua_img_validate(lua_State* L){

}*/

int lua_img_get_size(lua_State* L){

}


int lua_img_create_from_image(lua_State* L){

}

int lua_img_write_as_image(lua_State* L){

}

int lua_img_create_from_binary(lua_State* L){

}

int lua_img_write_as_binary(lua_State* L){

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

int lua_gpu_upload(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int input = luaL_checkinteger(L,2);
    img_t* image = lua_touserdata(L,3);

    lua_pushinteger(L,img_gpu_upload(gpu,input,image->memory,img_get_size(image)));
    return 1;
}

int lua_gpu_download(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int input = luaL_checkinteger(L,2);
    img_t* image = lua_touserdata(L,3);

    lua_pushinteger(L,img_gpu_download(gpu,input,image->memory,img_get_size(image)));
    return 1;
}


int lua_gpu_add_stage(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    img_gpu_program_t* program = lua_touserdata(L,2);
    int w = luaL_checkinteger(L,3);
    int h = luaL_checkinteger(L,4);
    int d = luaL_checkinteger(L,5);

    lua_pushinteger(L,img_gpu_add_stage(gpu,program,w,h,d));
    return 1;
}

int lua_gpu_add_stage_data(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    int stage = luaL_checkinteger(L,2);

    int type = lua_type(L, 3);
    if(type == LUA_TNUMBER){
        if(lua_isinteger(L, 3)){
            uint32_t val = lua_tointeger(L, 3);
            img_gpu_add_stage_data(gpu, stage, &val, sizeof(val));
        } else {
            float val = lua_tonumber(L, 3);
            img_gpu_add_stage_data(gpu, stage, &val, sizeof(val));
        }
    } else if(type == LUA_TTABLE){
        // array of floats e.g. a kernel
        int n = lua_rawlen(L, 3);
        float* buf = alloca(n * sizeof(float));
        for(int i = 0; i < n; i++){
            lua_rawgeti(L, 3, i + 1);
            buf[i] = lua_tonumber(L, -1);
            lua_pop(L, 1);
        }
        img_gpu_add_stage_data(gpu, stage, buf, n * sizeof(float));
    }

    return 0;
}

int lua_gpu_dispatch(lua_State* L){
    img_gpu_t* gpu = lua_touserdata(L,1);
    img_gpu_dispatch(gpu);
    return 0;
}

static const luaL_Reg img_lib[] = {
    {"gpu_init",            lua_gpu_init},
    {"gpu_load_program",    lua_gpu_load_program_glsl},
    {"gpu_allocate_image",  lua_gpu_allocate_image},
    {"gpu_allocate_buffer", lua_gpu_allocate_buffer},
    {"gpu_upload",          lua_gpu_upload},
    {"gpu_download",        lua_gpu_download},
    {"gpu_add_stage",       lua_gpu_add_stage},
    {"gpu_add_stage_data",  lua_gpu_add_stage_data},
    {"gpu_dispatch",        lua_gpu_dispatch},
    {NULL, NULL}
};

int luaopen_img(lua_State* L){
    luaL_newlib(L, img_lib);
    return 1;
}