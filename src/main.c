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

#include "vkr.h"
#define CGLM_SIMD_ENABLE
#define CGLM_DEFINE_PRINTS
#include <cglm/cglm.h>

#define IMAGE_IMPL
#include "image.h"

int main(int argc, char** argv){

	uint32_t input;
	const char* file_output_default = "output.bmp";

	char* file_input = NULL;
	char* file_output = file_output_default;
	char* program_name = NULL;
	char* pargs[16] = {0};

	uint32_t cpu_program = UINT32_MAX;

	/*
		-i input.xyz
		-o output
		-p program, including gpu: gpu_program.comp / .glsl / .spv (shell call to auto compile glsl?)
	*/

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
			if(strcmp(program_name,img_program_table[i].name) == 0 && strlen(program_name) == strlen(img_program_table[i].name)){
				cpu_program = i;
				break;
			}
		}

		img_t image_input;
		img_t image_output;

		time_t time_start = 0;
		time_t time_end = 0;

		if(cpu_program != UINT32_MAX){
			// cpu program
			time_start = clock();
			image_input = img_create_from_image(file_input,0);
			time_end = clock();
			printf("image load: %.2fms\n",(((float)(time_end-time_start))/CLOCKS_PER_SEC)*1000);
			printf("starting program '%s' \n",img_program_table[cpu_program].name);

			time_start = clock();
			image_output = img_program_table[cpu_program].program(image_input, arg_count, pargs);
			time_end = clock();
			printf("program exec: %.2fms\n",(((float)(time_end-time_start))/CLOCKS_PER_SEC)*1000);
		}else{
			// gpu program or nothing at all, who knows
			printf("starting gpu program '%s'\n",program_name);
			time_start = clock();
			image_input = img_create_from_image(file_input,4);
			FILE* f = fopen(program_name,"r");
			if(!f){
				printf("program not in table, and no gpu shader found: %s\n",program_name);
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
				program_name,
				"main",
				options
			);

			if(shaderc_result_get_compilation_status(module) != shaderc_compilation_status_success) {
				printf("glslc compile fail %s\n", shaderc_result_get_error_message(module));
				exit(-1);
			}

			size_t size = shaderc_result_get_length(module);
			const uint32_t* spv_data = shaderc_result_get_bytes(module);

			vkr_state vkr = {0};
			int res = 0;
			res = vkr_init(&vkr);

			VkFenceCreateInfo info_fence = {
				.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
				.flags = VK_FENCE_CREATE_SIGNALED_BIT
			};
			VkFence fence;
			vkCreateFence(vkr.device,&info_fence,NULL,&fence);

			// descriptors etc

			vkr_descriptor_info descriptors[] = {
				{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, MAX_TEXTURES}
			};
			vkr_generate_descriptor_state(&vkr,sizeof(descriptors)/sizeof(descriptors[0]),descriptors);
			vkr_generate_pipeline_layout(&vkr);

			printf("init with %d\n",res);

			vkr_texture device_input = vkr_create_texture(
				&vkr, image_input.width,image_input.height,image_input.depth,
				VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,VK_IMAGE_ASPECT_COLOR_BIT
			);
			vkr_texture device_output = vkr_create_texture(
				&vkr, image_input.width,image_input.height,image_input.depth,
				VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,VK_IMAGE_ASPECT_COLOR_BIT
			);

			vkr_bind_view_compute(&vkr,0,device_input.view, 0 );
			vkr_bind_view_compute(&vkr,0,device_output.view, 1 );

			// staging
			
			VkBuffer hostBuffer;
			VkDeviceMemory hostMemory;
			VkDeviceSize hostSize = image_input.width*image_input.height*image_input.depth*image_input.channels*sizeof(float);

			vkr_alloc(vkr.device,vkr.physical_device,hostSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &hostBuffer, &hostMemory);

			void* data;
			vkMapMemory(vkr.device, hostMemory, 0, hostSize, 0, &data);
				memcpy(data, image_input.memory, (size_t)hostSize);
			vkUnmapMemory(vkr.device, hostMemory);

			// pso

			VkShaderModule shader = vkr_shader_module_create(vkr.device, spv_data, size);
			VkPipeline pipeline = vkr_generate_pipeline_compute(&vkr,shader);


			// process image

			VkCommandBuffer cmd = vkr_stc_begin(vkr.device,vkr.command_pool);

			vkr_texture_transition_many(cmd,1,&device_input.image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,vkr_texture_subresource_default());
			vkr_copy_buffer_to_image(&vkr,cmd,hostBuffer,device_input.image, image_input.width, image_input.height, image_input.depth);
			vkr_texture_transition_many(cmd,1,&device_input.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,vkr_texture_subresource_default());
			vkr_texture_transition_many(cmd,1,&device_output.image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,vkr_texture_subresource_default());

			vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
			vkCmdBindDescriptorSets(
				cmd,
				VK_PIPELINE_BIND_POINT_COMPUTE,
				vkr.pipeline_state.layout,
				0,
				1, &vkr.descriptor_state.sets[0],
				0, NULL
			);

			int width = image_input.width;
			int height = image_input.height;
			if(width <= 0 || height <= 0) exit(-2);

			struct pushconstant {
				uint32_t lol;
			};
			struct pushconstant pc = {
				.lol = 5.
			};
			
			vkCmdPushConstants(cmd, vkr.pipeline_state.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
			vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
			

			vkr_texture_transition_many(cmd,1,&device_output.image,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,vkr_texture_subresource_default());
			vkr_copy_image_to_buffer(&vkr,cmd,hostBuffer,device_output.image,image_input.width,image_input.height,image_input.depth);

			vkr_stc_end(vkr.device,vkr.command_pool,vkr.queue[VKR_QUEUE_GRAPHICS],cmd);

			VkSubmitInfo info_submit = {
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.commandBufferCount = 1,
				.pCommandBuffers = &cmd
			};

			time_end = clock();
			printf("vulkan instance and PSOs setup: %.2fms\n",(((float)(time_end-time_start))/CLOCKS_PER_SEC)*1000);
			time_start = clock();

			vkQueueSubmit(vkr.queue[0],1,&info_submit,fence);
			vkWaitForFences(vkr.device,1,&fence,VK_TRUE,UINT64_MAX);
			vkResetFences(vkr.device,1,&fence);

			time_end = clock();
			printf("gpu compute dispatch: %.2fms\n",(((float)(time_end-time_start))/CLOCKS_PER_SEC)*1000);

			image_output = img_create_zero(image_input.width,image_input.height,image_input.depth,image_input.channels);

			vkMapMemory(vkr.device, hostMemory, 0, hostSize, 0, &data);
				memcpy(image_output.memory, data, (size_t)hostSize);
			vkUnmapMemory(vkr.device, hostMemory);
			
			//exit(-2);
			vkr_destroy(&vkr);
		}

		img_write_as_image(&image_output, file_output);

		img_destroy(&image_input);
		img_destroy(&image_output);
	}
	return 0;
}