#ifndef IMAGE_LUA_H
#define IMAGE_LUA_H

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

//int lua_img_alloc(lua_State* L);
int lua_img_create_fill(lua_State* L);
int lua_img_create_zero(lua_State* L);
int lua_img_destroy(lua_State* L);
//int lua_img_validate(lua_State* L);
int lua_img_get_size(lua_State* L);

int lua_img_width(lua_State* L);
int lua_img_height(lua_State* L);
int lua_img_depth(lua_State* L);
int lua_img_channels(lua_State* L);

int lua_img_create_from_image(lua_State* L);
int lua_img_write_as_image(lua_State* L);
int lua_img_create_from_binary(lua_State* L);
int lua_img_write_as_binary(lua_State* L);

int lua_gpu_init(lua_State* L);
int lua_gpu_load_program_glsl(lua_State* L);
int lua_gpu_allocate_image(lua_State* L);
int lua_gpu_allocate_buffer(lua_State* L);
int lua_gpu_upload(lua_State* L);
int lua_gpu_download(lua_State* L);

int lua_gpu_add_stage(lua_State* L);
int lua_gpu_add_stage_data(lua_State* L);
int lua_gpu_dispatch(lua_State* L);

int luaopen_img(lua_State* L);

#endif