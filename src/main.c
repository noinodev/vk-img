#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "stb_image.h"
#include <shaderc/shaderc.h>
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include "vkr.h"
#define CGLM_SIMD_ENABLE
#define CGLM_DEFINE_PRINTS
#include <cglm/cglm.h>

#define IMAGE_IMPL
#include "image.h"

int main(int argc, char** argv){

	int status;
	lua_State* L = luaL_newstate();
	luaL_openlibs(L);

	if(argc < 2){
		printf("usage: %s /path/to/script.lua\n",argv[0]);
		return 1;
	}

	status = luaL_dofile(L,argv[1]);
	if(status){
		printf("err: %s\n",lua_tostring(L,-1));
		return 1;
	}

	return 0;
}

/*int main(int argc, char** argv){

	uint32_t input;
	const char* file_output_default = "output.bmp";

	char* file_input = NULL;
	char* file_output = file_output_default;
	char* program_name = NULL;
	char* pargs[16] = {0};

	uint32_t cpu_program = UINT32_MAX;

	
		-i input.xyz
		-o output
		-p program, including gpu: gpu_program.comp / .glsl / .spv (shell call to auto compile glsl?)
	

	uint8_t arg_gather = 0;
	uint8_t arg_count = 0;
	for(uint32_t i = 1; i < argc-1; i++){
		uint8_t arg_stop = 1;
		if(strncmp(argv[i],"-i",2) == 0) file_input = argv[i+1];
		else if(strncmp(argv[i],"-o",2) == 0) file_output = argv[i+1];
		else if(strncmp(argv[i],"-p",2) == 0) {
			arg_gather = 1;
			arg_stop = 0;
		}else if(arg_gather == 1){
			arg_stop = 0;
		}

		if(arg_gather == 1){
			if(arg_stop == 1) arg_gather = 0;
			else pargs[arg_count++] = argv[i+1];
		}
	}

	program_name = pargs[0];

	if(program_name != NULL){

		// check cpu programs
		for(uint32_t i = 0; i < sizeof(img_program_table)/sizeof(img_program_table[0]); i++){
			if(strcmp(program_name,img_program_table[i].name) == 0 && s#include <lauxlib.h>
#include <lualib.h>trlen(program_name) == strlen(img_program_table[i].name)){
				cpu_program = i;
				break;
			}
		}

		img_t image_input = {0};
		img_t image_output = {0};

		time_t time_start = 0;
		time_t time_end = 0;

		if(cpu_program != UINT32_MAX){
			// cpu program
			time_start = clock();
			uint32_t channels = img_program_table[cpu_program].channel_override;
			image_input = img_create_from_image(file_input,channels);
			time_end = clock();
			printf("image load: %.2fms\n",(((float)(time_end-time_start))/CLOCKS_PER_SEC)*1000);
			printf("starting program '%s' \n",img_program_table[cpu_program].name);

			time_start = clock();
			image_output = img_program_table[cpu_program].program(image_input, arg_count, pargs);
			time_end = clock();
			printf("program exec: %.2fms\n",(((float)(time_end-time_start))/CLOCKS_PER_SEC)*1000);
		}else{
			// no program
			printf("not a valid program\n");
			exit(-1);
		}

		img_write_as_image(&image_output, file_output);

		img_destroy(&image_input);
		img_destroy(&image_output);
	}
	return 0;
}*/