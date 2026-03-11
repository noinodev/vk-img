#ifndef VKR_H
#define VKR_H

#include "vulkan/vulkan_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "cglm/cglm.h"

#define MAX_FRAMES_IN_FLIGHT 2
#define MAX_TEXTURES 256

#define VK_CHECK(f,r)                                    \
    do {                                                     \
        VkResult res = (f);                                  \
        if (res != VK_SUCCESS) {                             \
            printf("Fatal : VkResult is \"%d\" in %s at line %d\n", res, __FILE__, __LINE__); \
            return (r);                                       \
        }                                                    \
    } while (0)

enum VKR_QUEUE_FAMILIES {
	VKR_QUEUE_GRAPHICS,

	VKR_QUEUE_COUNT
};

typedef struct vkr_state vkr_state;

typedef struct {
	int queue[VKR_QUEUE_COUNT];
} vkr_qfi; // queue family indices

typedef struct {
	uint32_t width, height, depth, mips;
    VkFormat format;
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
	VkImageViewType type;
	VkImageLayout layout;
} vkr_texture;

enum VKR_DESCRIPTOR_BINDINGS {
	VKR_DESCRIPTOR_TEXTURE,
	VKR_DESCRIPTOR_TARGET,
	VKR_DESCRIPTOR_MATERIAL,
	VKR_DESCRIPTOR_INSTANCE,
	VKR_DESCRIPTOR_CUBES,
	VKR_DESCRIPTOR_CUBES_DEPTH,
	VKR_DESCRIPTOR_TEXTURE_ARRAY,
	VKR_DESCRIPTOR_STORAGE_IMAGE,

	VKR_DESCRIPTOR_COUNT
};

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
} vkr_ssbo; // shader storage buffer object


#define VKR_SBO_ALLOC 200*1024*1024
typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped_ptr;
    VkDeviceSize capacity;
    VkDeviceSize offset;
	size_t alignment;
} vkr_sbo; // staging buffer object

int vkr_generate_staging_state(vkr_state* vkr);

/*enum VKR_SHADER_FAMILY {
	VKR_SHADER_FAMILY_FORWARD,
	VKR_SHADER_FAMILY_GBUFFER,
	VKR_SHADER_FAMILY_COMPOSITE,
	VKR_SHADER_FAMILY_REFLECTION,
	VKR_SHADER_FAMILY_SHADOW,
	VKR_SHADER_FAMILY_RADIANCE,
	VKR_SHADER_FAMILY_MINMAX,
	VKR_SHADER_FAMILY_COUNT
};*/

enum VKR_STAGE_INDEX {
    VKR_STAGE_VS = 0,
    VKR_STAGE_FS,
    VKR_STAGE_GS,
    VKR_STAGE_TES,
    VKR_STAGE_TCS,
	VKR_STAGE_COMP,
    VKR_STAGE_COUNT
};

enum VKR_STAGE_FLAGS {
    VKR_STAGEF_VS  = 1 << VKR_STAGE_VS,
    VKR_STAGEF_FS  = 1 << VKR_STAGE_FS,
    VKR_STAGEF_GS  = 1 << VKR_STAGE_GS,
    VKR_STAGEF_TES = 1 << VKR_STAGE_TES,
    VKR_STAGEF_TCS = 1 << VKR_STAGE_TCS,
	VKR_STAGEF_COMP = 1 << VKR_STAGE_COMP
};

static const VkShaderStageFlagBits vkr_stage_conversion[] = {
	VK_SHADER_STAGE_VERTEX_BIT,
	VK_SHADER_STAGE_FRAGMENT_BIT,
	VK_SHADER_STAGE_GEOMETRY_BIT,
	VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
	VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
	VK_SHADER_STAGE_COMPUTE_BIT
};

typedef struct { // use this to generate families
    const char* path[VKR_STAGE_COUNT];
    uint32_t stage_mask;
} vkr_shader_bundle;

typedef struct { // individual modules
    VkShaderModule module;
    VkShaderStageFlagBits stage;
    const char* debug_name;
} vkr_shader;

typedef struct { // group of modules 'family'
	vkr_shader stage_shader[VKR_STAGE_COUNT];
	uint32_t stage_mask;
} vkr_shader_family;

int vkr_generate_shader_family(vkr_state* vkr, vkr_shader_bundle* bundle, vkr_shader_family* family);
int vkr_generate_shader_state(vkr_state* vkr);

enum VKR_SHADER_TYPE {
    VKR_SHADER_VERT,
    VKR_SHADER_FRAG,
    VKR_SHADER_TESC,
    VKR_SHADER_TESE,
    VKR_SHADER_GEOM,
    VKR_SHADER_COMP,
    VKR_SHADER_COUNT
};

enum VKR_PIPELINE_SET {
	VKR_PIPELINE_FORWARD,
	VKR_PIPELINE_DEFERRED,
	VKR_PIPELINE_COMPOSITE,
	VKR_PIPELINE_COMPUTE,
	VKR_PIPELINE_REFLECTION,
	VKR_PIPELINE_SHADOWS,
	VKR_PIPELINE_RADIANCE,
	VKR_PIPELINE_MINMAX,
	VKR_PIPELINE_COUNT
};

typedef struct {
	VkGraphicsPipelineCreateInfo info[VKR_PIPELINE_COUNT];
	VkPipeline pipeline[VKR_PIPELINE_COUNT]; // pipelines
	VkPipelineLayout layout; // unified layout
	uint32_t attributes[VKR_PIPELINE_COUNT];
} vkr_pipeline_state;

int vkr_generate_pipeline_layout(vkr_state* vkr);
VkPipeline vkr_generate_pipeline_compute(vkr_state* vkr, VkShaderModule shader);
int vkr_bind_view_compute(vkr_state* vkr, uint32_t binding, VkImageView view, uint32_t index);

typedef struct {
	uint32_t binding;
	VkDescriptorType type;
	VkShaderStageFlags stage;
	uint32_t count;
} vkr_descriptor_info;

typedef struct {
	vkr_descriptor_info* descriptor_info;
	VkDescriptorSetLayout layout;
	VkDescriptorPool pool;
	VkDescriptorSet sets[MAX_FRAMES_IN_FLIGHT];
} vkr_descriptor_state;

int vkr_generate_descriptor_state(vkr_state *vkr, uint32_t binding_count, vkr_descriptor_info* descriptors);

struct vkr_state {
	VkInstance instance;
	VkSurfaceKHR surface;
	VkPhysicalDevice physical_device;
	VkDevice device;
	VkQueue queue[VKR_QUEUE_COUNT];
	vkr_qfi indices;

	VkQueryPool query_pool;
	VkPhysicalDeviceProperties properties;

	//vkr_shader_state shader_state;
	vkr_pipeline_state pipeline_state;
	vkr_descriptor_state descriptor_state;

	VkCommandPool command_pool;
	VkCommandBuffer command_buffer[MAX_FRAMES_IN_FLIGHT];

	/*uint32_t sem_idx;
	uint32_t sem_idx_to_frame[MAX_FRAMES_IN_FLIGHT];
	VkSemaphore* sem_image_available;
	VkSemaphore* sem_render_finished;
	VkFence fence_inflight[MAX_FRAMES_IN_FLIGHT];*/

	// textures
	//vkr_texture* textures;
	//uint32_t texture_count, texture_capacity;

	// targets
	//vkr_texture targets[VKR_TARGET_COUNT];

	// staging arenas
	//vkr_sbo staging[MAX_FRAMES_IN_FLIGHT];

	// draw queue

	VkDebugUtilsMessengerEXT debug_messenger;
};


uint32_t vkr_find_memtype(VkPhysicalDevice phys_dev, uint32_t typeFilter, VkMemoryPropertyFlags props);
void vkr_alloc(VkDevice device, VkPhysicalDevice phys_dev, VkDeviceSize size,
                  VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                  VkBuffer* buffer, VkDeviceMemory* bufferMemory);
VkCommandBuffer vkr_stc_begin(VkDevice device, VkCommandPool pool);
void vkr_stc_end(VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer cmd);
VkImageSubresourceRange vkr_texture_subresource_default();
int vkr_texture_transition(vkr_state* vkr, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
int vkr_texture_transition_many(VkCommandBuffer cmd, uint32_t count, VkImage* image, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subresource);
int vkr_texture_copy_many(VkCommandBuffer cmd, uint32_t count, VkBuffer buffer, size_t* arena, VkImage* image, uint32_t offset[][3], uint32_t extent[][3]); 

vkr_sbo vkr_sbo_init(vkr_state* vkr, size_t capacity);
void vkr_sbo_reset(vkr_sbo* sbo);
size_t vkr_sbo_alloc(vkr_sbo* sbo, VkDeviceSize size);
void vkr_sbo_destroy(vkr_sbo* sbo); // ADD THIS FOR REALLOC !?

//void vkr_copy_buffer_to_image(vkr_state* vkr, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth);
void vkr_copy_image_to_buffer(vkr_state* vkr, VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth);
void vkr_copy_buffer_to_image(vkr_state* vkr, VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth);
//size_t vkr_texture_upload(vkr_state* vkr, const char* file);

//vkr_ssbo vkr_create_ssbo(vkr_state* vkr, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);

uint8_t vkr_validate_queue_families(vkr_qfi indices);
vkr_qfi vkr_find_queue_families(VkPhysicalDevice device);

uint32_t* vkr_shader_load(const char* filename, size_t* sizeptr);
VkShaderModule vkr_shader_module_create(VkDevice device, const uint32_t* code, size_t size);

int vkr_init(vkr_state* vkr);
//int vkr_create_compute(vkr_state *vkr, const char *shader, VkPipeline* pipeline, VkPipelineLayout* pipeline_layout, VkDescriptorSetLayout* desc_layout);
//int vkr_create_graphics(vkr_state* vkr, const char *vertex, const char* fragment, VkPipeline* pipeline, VkPipelineLayout* pipeline_layout, VkDescriptorSetLayout* desc_layout);
int vkr_create_command_pool(vkr_state* vkr);
int vkr_create_command_buffer(vkr_state* vkr);
//int vkr_create_sampler(vkr_state* vkr);
VkImageView vkr_create_image_view(vkr_state* vkr, VkImage image, VkFormat format, VkImageAspectFlags flags, VkImageViewType type);
int vkr_create_image(vkr_state* vkr, uint32_t width, uint32_t height, uint32_t depth,
	VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, 
	VkMemoryPropertyFlags properties, VkImage* image, VkDeviceMemory* imageMemory);
vkr_texture vkr_create_texture(vkr_state* vkr, uint32_t width, uint32_t height, uint32_t depth, 
	VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties, VkImageAspectFlags flags);
vkr_texture vkr_create_texture_cube(
	vkr_state* vkr, uint32_t width, uint32_t height,
	VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties, VkImageAspectFlags flags);
vkr_texture vkr_create_texture_mips(
	vkr_state* vkr, uint32_t width, uint32_t height, 
	VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties, VkImageAspectFlags flags);
vkr_texture vkr_create_texture_array(
	vkr_state* vkr, uint32_t width, uint32_t height, uint32_t count,
	VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties, VkImageAspectFlags flags);

int vkr_destroy(vkr_state* vkr);
int vkr_bind_view(vkr_state* vkr, uint32_t binding, VkImageView view, uint32_t index, VkSampler sampler);
int vkr_bind_pipeline(vkr_state* vkr, VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineBindPoint bind_point);

#endif