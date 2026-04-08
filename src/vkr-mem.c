#include "vkr.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "stb_image.h"
#include "vulkan/vulkan_core.h"

uint32_t vkr_find_memtype(VkPhysicalDevice phys_dev, uint32_t typeFilter, VkMemoryPropertyFlags props){
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }

    // Handle failure (production code should use VK_ERROR_EXTENSION_NOT_PRESENT, etc.)
    assert(0 && "Failed to find suitable memory type!");
    return 0;
}

void vkr_alloc(VkDevice device, VkPhysicalDevice phys_dev, VkDeviceSize size,
                  VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                  VkBuffer* buffer, VkDeviceMemory* bufferMemory) {
	VkResult vkres;

    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    vkCreateBuffer(device, &bufferInfo, NULL, buffer);

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, *buffer, &memRequirements);
	assert(memRequirements.size > 0 && "Buffer memory requirements size is zero!");

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = vkr_find_memtype(phys_dev, memRequirements.memoryTypeBits, properties),
    };

    vkres = vkAllocateMemory(device, &allocInfo, NULL, bufferMemory);
	assert(vkres == VK_SUCCESS);
    vkBindBufferMemory(device, *buffer, *bufferMemory, 0);
	assert(vkres == VK_SUCCESS);
}

VkCommandBuffer vkr_stc_begin(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = pool,
        .commandBufferCount = 1,
    };

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

void vkr_stc_end(VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

void vkr_copy_buffer_to_image(vkr_state* vkr, VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth){
    //VkCommandBuffer cmd = vkr_stc_begin(vkr->device,vkr->command_pool);

    VkBufferImageCopy region = {
    	.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,

		.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.imageSubresource.mipLevel = 0,
		.imageSubresource.baseArrayLayer = 0,
		.imageSubresource.layerCount = 1,

		.imageOffset.x = 0,
		.imageOffset.y = 0,
		.imageOffset.z = 0,
		.imageExtent.width = width,
		.imageExtent.height = height,
		.imageExtent.depth = depth
    };

    vkCmdCopyBufferToImage(
	    cmd,
	    buffer,
	    image,
	    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    1,
	    &region
	);

    //vkr_stc_end(vkr->device,vkr->command_pool,vkr->queue[VKR_QUEUE_GRAPHICS],cmd);
}

void vkr_copy_image_to_buffer(vkr_state* vkr, VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth){
    VkBufferImageCopy region = {
    	.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,

		.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.imageSubresource.mipLevel = 0,
		.imageSubresource.baseArrayLayer = 0,
		.imageSubresource.layerCount = 1,

		.imageOffset.x = 0,
		.imageOffset.y = 0,
		.imageOffset.z = 0,
		.imageExtent.width = width,
		.imageExtent.height = height,
		.imageExtent.depth = depth
    };

    vkCmdCopyImageToBuffer(
	    cmd,
	    image,
	    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		buffer,
	    1,
	    &region
	);
}

void vkr_copy_buffer(vkr_state* vkr, VkCommandBuffer cmd, VkBuffer src, VkBuffer dest, VkDeviceSize size){
    VkBufferCopy region = {
		.dstOffset = 0,
		.srcOffset = 0,
		.size = size
    };

	vkCmdCopyBuffer(cmd,src,dest,1,&region);
}

VkImageSubresourceRange vkr_texture_subresource_default(){
	return (VkImageSubresourceRange){
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.baseMipLevel = 0,
		.levelCount = 1,
		.baseArrayLayer = 0,
		.layerCount = 1
	};
}

int vkr_texture_transition(vkr_state* vkr, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout){
	VkCommandBuffer cmd = vkr_stc_begin(vkr->device,vkr->command_pool);

	vkr_texture_transition_many(cmd,1,&image,oldLayout,newLayout,vkr_texture_subresource_default());

	vkr_stc_end(vkr->device,vkr->command_pool,vkr->queue[VKR_QUEUE_GRAPHICS],cmd);
	return 1;
}

int vkr_texture_transition_many(VkCommandBuffer cmd, uint32_t count, VkImage* image, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subresource){
	VkImageMemoryBarrier barrier[16] = {0};

	VkAccessFlags srcAccessMask;
	VkAccessFlags dstAccessMask;
	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
	    srcAccessMask = 0;
	    dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL){
		srcAccessMask = 0;
		dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL){
		srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
		srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
		srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL){
		srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}/*else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
	    srcAccessMask = 0;
	    dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	    destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
	    srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	    dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL){
		srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL){
	    srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	    dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	    sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL){
	    srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	    dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	    sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
	    destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		sourceStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	}else if(oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		srcAccessMask = VK_ACCESS_SHADER_READ_BIT;           // reading from compute shader
		dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; // writing in depth pass
		sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	}*/else{
		printf("invalid layout transition [!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!]\n");
		return -1;
	}

	for(size_t i = 0; i < count; i++){
		barrier[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier[i].oldLayout = oldLayout;
		barrier[i].newLayout = newLayout;
		barrier[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier[i].srcAccessMask = srcAccessMask;
		barrier[i].dstAccessMask = dstAccessMask;
		barrier[i].image = image[i];
		barrier[i].subresourceRange = subresource; 
	}

	vkCmdPipelineBarrier(
	    cmd,
	    sourceStage, destinationStage,
	    0,
	    0, NULL,
	    0, NULL,
	    count, barrier
	);

	return 1;
}

VkImageView vkr_create_image_view(vkr_state* vkr, VkImage image, VkFormat format, VkImageAspectFlags flags, VkImageViewType type){
    VkImageViewCreateInfo viewInfo = {0};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = type;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = flags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    VK_CHECK(vkCreateImageView(vkr->device, &viewInfo, NULL, &imageView),VK_NULL_HANDLE);

    return imageView;
}

int vkr_create_image(vkr_state* vkr, uint32_t width, uint32_t height, uint32_t depth,
	VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, 
	VkMemoryPropertyFlags properties, VkImage* image, VkDeviceMemory* imageMemory) {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = depth>1?VK_IMAGE_TYPE_3D:VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = depth;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateImage(vkr->device, &imageInfo, NULL, image),-1);

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(vkr->device, *image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vkr_find_memtype(vkr->physical_device, memRequirements.memoryTypeBits, properties);

    VK_CHECK(vkAllocateMemory(vkr->device, &allocInfo, NULL, imageMemory),-1);

    vkBindImageMemory(vkr->device, *image, *imageMemory, 0);
    return 1;
}

vkr_texture vkr_create_texture(vkr_state* vkr, uint32_t width, uint32_t height, uint32_t depth, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
	VkMemoryPropertyFlags properties, VkImageAspectFlags flags){
	vkr_texture texture = {0};
	vkr_create_image(vkr,width,height,depth,format,tiling,usage,properties,&texture.image,&texture.memory);
	texture.view = vkr_create_image_view(vkr,texture.image,format,flags,depth>1?VK_IMAGE_VIEW_TYPE_3D:VK_IMAGE_VIEW_TYPE_2D);
	texture.format = format;
	texture.width = width;
	texture.height = height;
	texture.depth = depth;
	texture.type = depth>1?VK_IMAGE_VIEW_TYPE_3D:VK_IMAGE_VIEW_TYPE_2D;
	texture.layout = VK_IMAGE_LAYOUT_UNDEFINED;

	return texture;
}

void vkr_destroy_texture(vkr_state* vkr, vkr_texture* texture){
	vkDestroyImageView(vkr->device,texture->view,NULL);
	vkDestroyImage(vkr->device,texture->image,NULL);
	vkFreeMemory(vkr->device,texture->memory,NULL);
	*texture = (vkr_texture){0};
}

void vkr_destroy_buffer(vkr_state* vkr, vkr_buffer* buffer){
	vkDestroyBuffer(vkr->device,buffer->buffer,NULL);
	vkFreeMemory(vkr->device,buffer->memory,NULL);
	*buffer = (vkr_buffer){0};
}	