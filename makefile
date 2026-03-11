CC = gcc
GLSLC = glslc

# Detect platform
ifeq ($(OS),Windows_NT)
    PLATFORM = windows
else
    PLATFORM = linux
endif

ifeq ($(PLATFORM), windows)
    VK_INCLUDE = C:/devlib/VulkanSDK/Include
    VK_LIB = C:/devlib/VulkanSDK/Lib
    GLFW_INCLUDE = C:/devlib/glfw/include
    GLFW_LIB = C:/devlib/glfw/lib-mingw-w64
    LMDB_INCLUDE = C:/devlib/liblmdb
    LMDB_LIB = C:/devlib/liblmdb
    ZSTD_INCLUDE = C:/msys64/mingw64/include
    ZSTD_LIB = C:/msys64/mingw64/lib
    CGLM_INCLUDE = C:/devlib/cglm/include
    STBI_INCLUDE = C:/devlib/stbi
    OUT = vk.exe
    LDFLAGS = -L$(GLFW_LIB) -L$(VK_LIB) -L$(LMDB_LIB) -lglfw3 -lvulkan-1 -llmdb -lzstd -lgdi32 -luser32 -lshell32

    RM = del /Q
    RMDIR = rmdir /S /Q
    NULLDEV = >nul 2>&1

    MKDIR = if not exist $(SPIRV_DIR) mkdir $(SPIRV_DIR)
else
    VK_INCLUDE = /home/nate/dev/libs/vulkan/x86_64/include
    VK_LIB = /home/nate/dev/libs/vulkan/x86_64/lib
    CGLM_INCLUDE = /home/nate/dev/libs/cglm/include
    STBI_INCLUDE = /home/nate/dev/libs/stb
    OUT = vk
    LDFLAGS = -L$(VK_LIB) -lshaderc_combined -lstdc++ -lvulkan -lm -ldl -lX11 -lpthread -lGL

    RM = rm -f
    RMDIR = rm -rf
    NULLDEV = >/dev/null 2>&1

    MKDIR = mkdir -p $(SPIRV_DIR)
endif

CFLAGS = -g -O0 -march=native -I$(VK_INCLUDE) -I$(CGLM_INCLUDE)  -I$(STBI_INCLUDE)

SRC = $(wildcard src/*.c)
BUILD_DIR = build
OBJ = $(patsubst src/%.c, $(BUILD_DIR)/%.o, $(SRC))

# === Shader compilation ===
SHADER_SRC = $(wildcard glsl/*.vert glsl/*.frag glsl/*.comp glsl/*.geom glsl/*.tesc glsl/*.tese)
SPIRV_DIR = spv
SHADER_SPV = $(patsubst glsl/%.vert, $(SPIRV_DIR)/%.vert.spv, \
              $(patsubst glsl/%.frag, $(SPIRV_DIR)/%.frag.spv, \
              $(patsubst glsl/%.comp, $(SPIRV_DIR)/%.comp.spv, \
              $(patsubst glsl/%.geom, $(SPIRV_DIR)/%.geom.spv, \
              $(patsubst glsl/%.tesc, $(SPIRV_DIR)/%.tesc.spv, \
              $(patsubst glsl/%.tese, $(SPIRV_DIR)/%.tese.spv, $(SHADER_SRC)))))))

all: shaders $(OUT)

$(OUT): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(SPIRV_DIR)/%.spv: glsl/%
	@$(MKDIR)
	$(GLSLC) $< -o $@

shaders: $(SHADER_SPV)

clean:
	-$(RM) $(OUT) $(OBJ) $(SHADER_SPV) $(NULLDEV)
	-$(RMDIR) $(SPIRV_DIR) $(NULLDEV)
