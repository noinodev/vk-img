#include "vkr.h"
#include "vulkan/vulkan_core.h"
#include <stdint.h>
#include <cglm/cglm.h>
#include <time.h>

#define DEBUG

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* user_data) {

    fprintf(stderr, "[VALIDATION: %s]: %s\n",data->pMessageIdName, data->pMessage);
    return VK_FALSE;
}

VkResult vkr_create_debug(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    PFN_vkCreateDebugUtilsMessengerEXT func = 
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != NULL) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

int vkr_init(vkr_state* vkr){


	// VK INIT

	VkResult vkres;
	VkInstance instance;
	VkApplicationInfo info_application = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "image processor",
		.applicationVersion = VK_MAKE_VERSION(0,0,1),
		.pEngineName = "noengine",
		.engineVersion = VK_MAKE_VERSION(0,0,1),
		.apiVersion = VK_API_VERSION_1_3
	};

	const char* validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
	const char* debug_ext[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
	const char* required_extensions[] = {
		VK_EXT_DEBUG_UTILS_EXTENSION_NAME
	};

	VkInstanceCreateInfo info_create = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &info_application,
		.enabledExtensionCount = 1,
		.ppEnabledExtensionNames = required_extensions
	};

	#ifdef DEBUG
	VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {
	    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
	    .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
	                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
	    .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
	                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
	                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
	    .pfnUserCallback = debug_callback
	};

	info_create.enabledLayerCount = 1;
	info_create.ppEnabledLayerNames = validation_layers;
	info_create.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debug_create_info;
	#else
	info_create.enabledLayerCount = 0;
	info_create.ppEnabledLayerNames = NULL;
	info_create.pNext = NULL;
	#endif

	vkres = vkCreateInstance(&info_create, NULL, &instance);
	if (vkres != VK_SUCCESS) {
	    printf("vkCreateInstance failed [%d]\n", vkres);
	    return -1;
	}


	#ifdef DEBUG
	if (vkr_create_debug(instance, &debug_create_info, NULL, &vkr->debug_messenger) != VK_SUCCESS) {
	    fprintf(stderr, "failed to set up debug messenger!\n");
	}
	#endif

	// VK PHYSICAL DEVICES

	uint32_t physical_device_count = 0;
	vkres = vkEnumeratePhysicalDevices(instance,&physical_device_count,NULL);
	if(vkres != VK_SUCCESS){
		printf("vkEnumeratePhysicalDevices failed [%d]\n",vkres);
		return -1;
	}

	VkPhysicalDevice* physical_device_list = malloc(physical_device_count*sizeof(VkPhysicalDevice));
	vkEnumeratePhysicalDevices(instance,&physical_device_count,physical_device_list);
	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	int physical_device_best_score = -1;

	for (uint32_t i = 0; i < physical_device_count; i++) {
	    VkPhysicalDeviceProperties props;
	    vkGetPhysicalDeviceProperties(physical_device_list[i], &props);
	    VkPhysicalDeviceFeatures features;
	    vkGetPhysicalDeviceFeatures(physical_device_list[i], &features);

	    int score = 0;
	    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
	    score += props.limits.maxImageDimension2D;
	    if (!features.geometryShader) continue;

	    if (score > physical_device_best_score) {
	        physical_device_best_score = score;
	        physical_device = physical_device_list[i];
	    }
	}

	if(physical_device == VK_NULL_HANDLE){
	    printf("no suitable physical device found\n");
	    return -1;
	}

	free(physical_device_list);

	// QUEUE FAMILIES AND PHYSICAL DEVICE

	VkDevice device;

	float priority = 1.;
	vkr->indices = vkr_find_queue_families(physical_device);

	if(vkr_validate_queue_families(vkr->indices) == 0) {
	    printf(" queue families not found\n");
	    return -1;
	}

	uint32_t used_families[VKR_QUEUE_COUNT];
	size_t unique_count = 0;

	for (size_t i = 0; i < VKR_QUEUE_COUNT; i++) {
	    int already_added = 0;
	    for(size_t j = 0; j < unique_count; j++) {
	        if(vkr->indices.queue[i] == used_families[j]) {
	            already_added = 1;
	            break;
	        }
	    }

	    if(!already_added) used_families[unique_count++] = vkr->indices.queue[i];
	}

	VkDeviceQueueCreateInfo unique_families[VKR_QUEUE_COUNT] = {0};
	for(size_t i = 0; i < unique_count; i++){
		unique_families[i] = (VkDeviceQueueCreateInfo){
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = used_families[i],
			.queueCount = 1,
			.pQueuePriorities = &priority
		};
	}

	VkPhysicalDeviceFeatures physical_device_features = {0};
	const char* physical_device_extensions[] = {
	};

	VkPhysicalDeviceDescriptorIndexingFeatures indexing = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
		.descriptorBindingPartiallyBound = VK_TRUE,
		.runtimeDescriptorArray = VK_TRUE,
		.shaderSampledImageArrayNonUniformIndexing = VK_TRUE
	};

	VkPhysicalDeviceDynamicRenderingFeatures dynamic = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
		.dynamicRendering = VK_TRUE,
		.pNext = &indexing
	};

	VkPhysicalDeviceFeatures2 features2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.features = physical_device_features,
		.pNext = &dynamic
	};

	VkDeviceCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &features2,
		.pQueueCreateInfos = unique_families,
		.queueCreateInfoCount = unique_count
	};


	vkres = vkCreateDevice(physical_device,&info,NULL,&device);
	if(vkres != VK_SUCCESS){
		printf("vkCreateDevice failed [%d]\n",vkres);
		return -1;
	}

	for(size_t i = 0; i < VKR_QUEUE_COUNT; i++) vkGetDeviceQueue(device,vkr->indices.queue[i],0,&vkr->queue[i]);


	// RETURN
	vkr->instance = instance;
	vkr->physical_device = physical_device;
	vkr->device = device;

	vkGetPhysicalDeviceProperties(vkr->physical_device, &vkr->properties); // DEVICE PROPS!!


	int res;
	res = vkr_create_command_pool(vkr);
	if(res == -1){
		printf("vkr_create_command_pool failed\n");
		return -1;
	}
	res = vkr_create_command_buffer(vkr);
	if(res == -1){
		printf("vkr_create_command_buffer failed\n");
		return -1;
	}
	

	{
		VkQueryPoolCreateInfo queryPoolInfo = {};
		queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		queryPoolInfo.queryCount = 2*3; // start and end

		vkCreateQueryPool(vkr->device, &queryPoolInfo, NULL, &vkr->query_pool);
	}

	return 1;
}

vkr_qfi vkr_find_queue_families(VkPhysicalDevice device){
    vkr_qfi indices = {0};
    for(size_t i = 0; i < VKR_QUEUE_COUNT; i++) indices.queue[i] = -1;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device,&count,NULL);
    VkQueueFamilyProperties* props = malloc(count*sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(device,&count,props);

    for(uint32_t i = 0; i < count; i++) {
        // graphics
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.queue[VKR_QUEUE_GRAPHICS] = i;

        // validate and exit
        if(vkr_validate_queue_families(indices) == 1) break;
    }

    free(props);
    return indices;
}

uint8_t vkr_validate_queue_families(vkr_qfi indices){
	for(size_t i = 0; i < VKR_QUEUE_COUNT; i++){
		if(indices.queue[i] < 0) return 0;
	}
	return 1;
}

int vkr_destroy(vkr_state *vkr) {
    vkDeviceWaitIdle(vkr->device); // Ensure no operations are pending

    // Rest of Vulkan cleanup
    vkDestroyCommandPool(vkr->device, vkr->command_pool, NULL);

    // Destroy device
    vkDestroyDevice(vkr->device, NULL);

    // Destroy debug messenger if you created one
    if (vkr->debug_messenger != VK_NULL_HANDLE) {
        PFN_vkDestroyDebugUtilsMessengerEXT destroyFunc =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vkr->instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroyFunc != NULL) {
            destroyFunc(vkr->instance, vkr->debug_messenger, NULL);
        }
    }

    // Destroy instance
    vkDestroyInstance(vkr->instance, NULL);

    return 1;
}

int vkr_generate_shader_family(vkr_state* vkr, vkr_shader_bundle* bundle, vkr_shader_family* family){
	uint32_t count = 0;
	family->stage_mask = bundle->stage_mask;
	for(uint32_t i = 0; i < VKR_SHADER_COUNT; i++){
		family->stage_shader[i] = (vkr_shader){ 0 };
		if(!(bundle->stage_mask & (1 << i))) continue;

		size_t size;
		uint32_t* source = vkr_shader_load(bundle->path[count++],&size);

		family->stage_shader[i].module = vkr_shader_module_create(vkr->device,source,size);
		family->stage_shader[i].stage = vkr_stage_conversion[i];

		free(source);
	}
	return 1;
}

/*int vkr_generate_shader_state(vkr_state* vkr){
	vkr_shader_bundle bundle = { 0 };

	bundle = (vkr_shader_bundle){
		.stage_mask=VKR_STAGEF_VS|VKR_STAGEF_FS,
		.path={
			"spv/default.vert.spv",
			"spv/default.frag.spv"
		},
	};
	vkr_generate_shader_family(vkr,&bundle,&vkr->shader_state.family[VKR_SHADER_FAMILY_FORWARD]);

	bundle = (vkr_shader_bundle){
		.stage_mask=VKR_STAGEF_VS|VKR_STAGEF_FS,
		.path={
			"spv/deferred.vert.spv",
			"spv/deferred.frag.spv"
		},
	};
	vkr_generate_shader_family(vkr,&bundle,&vkr->shader_state.family[VKR_SHADER_FAMILY_GBUFFER]);

	bundle = (vkr_shader_bundle){
		.stage_mask=VKR_STAGEF_VS|VKR_STAGEF_FS,
		.path={
			"spv/composite.vert.spv",
			"spv/composite.frag.spv"
		},
	};
	vkr_generate_shader_family(vkr,&bundle,&vkr->shader_state.family[VKR_SHADER_FAMILY_COMPOSITE]);

	bundle = (vkr_shader_bundle){
		.stage_mask=VKR_STAGEF_VS|VKR_STAGEF_FS,
		.path={
			"spv/reflection.vert.spv",
			"spv/reflection.frag.spv"
		},
	};
	vkr_generate_shader_family(vkr,&bundle,&vkr->shader_state.family[VKR_SHADER_FAMILY_REFLECTION]);

	bundle = (vkr_shader_bundle){
		.stage_mask=VKR_STAGEF_VS|VKR_STAGEF_FS,
		.path={
			"spv/shadow.vert.spv",
			"spv/shadow.frag.spv"
		},
	};
	vkr_generate_shader_family(vkr,&bundle,&vkr->shader_state.family[VKR_SHADER_FAMILY_SHADOW]);

	bundle = (vkr_shader_bundle){
		.stage_mask=VKR_STAGEF_VS|VKR_STAGEF_FS,
		.path={
			"spv/composite.vert.spv",
			"spv/radiance.frag.spv"
		},
	};
	vkr_generate_shader_family(vkr,&bundle,&vkr->shader_state.family[VKR_SHADER_FAMILY_RADIANCE]);

	bundle = (vkr_shader_bundle){
		.stage_mask=VKR_STAGEF_COMP,
		.path={
			"spv/minmax.comp.spv"
		},
	};
	vkr_generate_shader_family(vkr,&bundle,&vkr->shader_state.family[VKR_SHADER_FAMILY_MINMAX]);

	
	return 1;
}*/

int vkr_fill_shader_stages(vkr_shader_family* family, uint32_t stage_mask, VkPipelineShaderStageCreateInfo* out){
    uint32_t idx = 0;
    for(uint32_t i = 0; i < VKR_STAGE_COUNT; i++){
        if(!(stage_mask & (1 << i))) continue;

        out[idx].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        out[idx].stage  = family->stage_shader[i].stage;
        out[idx].module = family->stage_shader[i].module;
        out[idx].pName  = "main";
        idx++;
    }

	return 1;
}

VkPipeline vkr_generate_pipeline_compute(vkr_state* vkr, VkShaderModule shader){
	VkPipelineShaderStageCreateInfo stage = {0};
	stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = shader;
	stage.pName = "main";
	//vkr_fill_shader_stages(shaders, shaders->stage_mask & VKR_STAGEF_COMP, &stage);

    VkComputePipelineCreateInfo info_pipeline = {
    	.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage,
        .layout = vkr->pipeline_state.layout,
        .flags = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1
    };

	VkPipeline out = {0};
    VkResult vkres = vkCreateComputePipelines(vkr->device,VK_NULL_HANDLE,1,&info_pipeline,NULL,&out);
    if(vkres != VK_SUCCESS){
    	printf("vkCreateGraphicsPipelines failed [%d]\n",vkres);
    	exit(-1);
    }
    
    return out;
}

int vkr_generate_pipeline_layout(vkr_state* vkr){
	// generate unified pipeline layout
	VkPushConstantRange push_constant_range = {
    	.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    	.offset = 0,
    	.size = 128 // 128 bytes or 2*sizeof(mat4)
    };

    VkPipelineLayoutCreateInfo pipeline_layout = {
    	.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    	.setLayoutCount = 1,
		.pSetLayouts = &vkr->descriptor_state.layout,
    	.pushConstantRangeCount = 1,
    	.pPushConstantRanges = &push_constant_range
    };

    VkResult vkres;
    vkres = vkCreatePipelineLayout(vkr->device,&pipeline_layout,NULL,&vkr->pipeline_state.layout);
    if(vkres != VK_SUCCESS){
    	printf("vkCreatePipelineLayout failed [%d]\n",vkres);
    	return -1;
    }

	return 1;
}

int vkr_bind_view(vkr_state* vkr, uint32_t binding, VkImageView view, uint32_t index, VkSampler sampler){
	VkDescriptorImageInfo info = {
		.sampler = sampler,           // one sampler for all textures is fine
		.imageView = view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
	};

	VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vkr->descriptor_state.sets[0], // your one bindless set
		.dstBinding = binding,                          // the bindless array binding
		.dstArrayElement = index,         // the slot in the array
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &info
	};

	vkUpdateDescriptorSets(vkr->device, 1, &write, 0, NULL);
}

int vkr_bind_view_compute(vkr_state* vkr, uint32_t binding, VkImageView view, uint32_t index){
	VkDescriptorImageInfo info = {
		.imageView = view,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL
	};

	VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vkr->descriptor_state.sets[0], // your one bindless set
		.dstBinding = binding,                          // the bindless array binding
		.dstArrayElement = index,         // the slot in the array
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &info
	};

	vkUpdateDescriptorSets(vkr->device, 1, &write, 0, NULL);
}

int vkr_bind_buffer_compute(vkr_state* vkr, uint32_t binding, VkBuffer buffer, uint32_t index, size_t size){
	VkDescriptorBufferInfo info = {
		.buffer = buffer,
		.range = size
	};

	VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vkr->descriptor_state.sets[0], // your one bindless set
		.dstBinding = binding,                          // the bindless array binding
		.dstArrayElement = index,         // the slot in the array
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &info
	};

	vkUpdateDescriptorSets(vkr->device, 1, &write, 0, NULL);
}

int vkr_create_command_pool(vkr_state* vkr){
	vkr_qfi queue_family_index = vkr_find_queue_families(vkr->physical_device);
	VkCommandPoolCreateInfo info_pool = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = queue_family_index.queue[VKR_QUEUE_GRAPHICS]
	};

	VkResult vkres;
	vkres = vkCreateCommandPool(vkr->device,&info_pool,NULL,&vkr->command_pool);
	if(vkres != VK_SUCCESS){
		printf("vkCreateCommandPool failed [%d]\n",vkres);
		return -1;
	}
	return 1;
}

int vkr_create_command_buffer(vkr_state* vkr){
	VkCommandBufferAllocateInfo info_alloc = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vkr->command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = MAX_FRAMES_IN_FLIGHT
	};

	VkResult vkres;
	vkres = vkAllocateCommandBuffers(vkr->device,&info_alloc,vkr->command_buffer);
	if(vkres != VK_SUCCESS){
		printf("vkAllocateCommandBuffers failed [%d]\n",vkres);
		return -1;
	}	
	return 1;
}

int vkr_generate_descriptor_layout(VkDevice device, vkr_descriptor_info* bindings, uint32_t binding_count, VkDescriptorSetLayout* layout){
    VkDescriptorSetLayoutBinding* vkBindings = malloc(sizeof(VkDescriptorSetLayoutBinding) * binding_count);
    for (uint32_t i = 0; i < binding_count; i++) {
        vkBindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = bindings[i].binding,
            .descriptorType = bindings[i].type,
            .descriptorCount = bindings[i].count,
            .stageFlags = bindings[i].stage,
            .pImmutableSamplers = NULL
        };
    }

	VkDescriptorBindingFlags bindingFlags = 
    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

	VkDescriptorBindingFlags* bindingFlagsArr = malloc(sizeof(VkDescriptorBindingFlags) * binding_count);
	for (uint32_t i = 0; i < binding_count; i++) bindingFlagsArr[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

	VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		.bindingCount = binding_count,
		.pBindingFlags = bindingFlagsArr
	};

    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = binding_count,
        .pBindings = vkBindings,
		.pNext = &flagsInfo,
    	.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
    };

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, layout) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor set layout\n");
    }

    free(vkBindings);
	free(bindingFlagsArr);
    return 1;
}

int vkr_generate_descriptor_pool(VkDevice device,vkr_descriptor_info* bindings,uint32_t binding_count,uint32_t max_sets, VkDescriptorPool* pool){
    VkDescriptorPoolSize* poolSizes = malloc(sizeof(VkDescriptorPoolSize) * binding_count);
    for (uint32_t i = 0; i < binding_count; i++) {
        poolSizes[i].type = bindings[i].type;
        poolSizes[i].descriptorCount = bindings[i].count * max_sets;
    }

    VkDescriptorPoolCreateInfo poolInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.poolSizeCount = binding_count,
		.pPoolSizes = poolSizes,
		.maxSets = max_sets,
		.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT
	};


    if (vkCreateDescriptorPool(device, &poolInfo, NULL, pool) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor pool\n");
    }

    free(poolSizes);
    return 1;
}

int vkr_generate_descriptor_sets(VkDevice device,VkDescriptorPool pool,VkDescriptorSetLayout layout, VkDescriptorSet* sets){
    VkDescriptorSetLayout layouts[MAX_FRAMES_IN_FLIGHT] = { 0 };
    for(uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) layouts[i] = layout;

    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pSetLayouts = layouts
    };

    if (vkAllocateDescriptorSets(device, &allocInfo, sets) != VK_SUCCESS) {
        fprintf(stderr, "Failed to allocate descriptor sets\n");
		return -1;
    }

    return 1;
}

int vkr_generate_descriptor_state(vkr_state *vkr, uint32_t binding_count, vkr_descriptor_info* descriptors){
	/*vkr_descriptor_info descriptors[] = {
		{VKR_DESCRIPTOR_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, MAX_TEXTURES},
		{VKR_DESCRIPTOR_TARGET, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, MAX_TEXTURES},
		{VKR_DESCRIPTOR_MATERIAL, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, MAX_TEXTURES},
		{VKR_DESCRIPTOR_INSTANCE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_VERTEX_BIT, MAX_TEXTURES},
		{VKR_DESCRIPTOR_CUBES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, MAX_TEXTURES},
		{VKR_DESCRIPTOR_CUBES_DEPTH, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, MAX_TEXTURES},
		{VKR_DESCRIPTOR_TEXTURE_ARRAY, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, MAX_TEXTURES},
		{VKR_DESCRIPTOR_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, MAX_TEXTURES}

		// more
	};*/

	uint32_t set_count = MAX_TEXTURES;
    vkr_generate_descriptor_layout(vkr->device, descriptors, binding_count,&vkr->descriptor_state.layout);
    vkr_generate_descriptor_pool(vkr->device, descriptors, binding_count, set_count, &vkr->descriptor_state.pool );
    vkr_generate_descriptor_sets(vkr->device, vkr->descriptor_state.pool, vkr->descriptor_state.layout, vkr->descriptor_state.sets);

    return 1;
}

uint32_t* vkr_shader_load(const char* filename, size_t* sizeptr){
	FILE* file = fopen(filename, "rb");
    if(!file){
        fprintf(stderr, "Failed to open shader file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long filesize = ftell(file);
    rewind(file);

    uint32_t* buffer = (uint32_t*)malloc(filesize);
    if(!buffer){
        fprintf(stderr, "Failed to allocate memory for shader\n");
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, filesize, file);
    fclose(file);

    *sizeptr = filesize;
    return buffer;
}

VkShaderModule vkr_shader_module_create(VkDevice device, const uint32_t* code, size_t size){
	VkShaderModuleCreateInfo createInfo = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code
	};

    VkShaderModule shaderModule;
    if(vkCreateShaderModule(device, &createInfo, NULL, &shaderModule) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create shader module\n");
        return VK_NULL_HANDLE;
    }

    return shaderModule;
}