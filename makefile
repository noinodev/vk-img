CC = gcc
GLSLC = glslc

# Detect platform
ifeq ($(OS),Windows_NT)
    PLATFORM = windows
else
    PLATFORM = linux
endif

ifeq ($(OS),Windows_NT)
    PLATFORM = windows
    OUT = vk.exe
    LIB_ROOT = C:/devlib

    INCLUDE_DIRS = \
        $(LIB_ROOT)/VulkanSDK/Include \
        $(LIB_ROOT)/cglm/include \
        $(LIB_ROOT)/stbi

    LIB_DIRS = $(LIB_ROOT)/VulkanSDK/Lib

    LIBS = vulkan-1 lua gdi32 user32 shell32
    
    RM = del /Q
    RMDIR = rmdir /S /Q
    MKDIR = if not exist $(@D) mkdir $(subst /,\,$(@D))
    NULLDEV = >nul 2>&1
else
    PLATFORM = linux
    OUT = vk
    LIB_ROOT = /home/nate/dev/libs

    INCLUDE_DIRS = \
        $(LIB_ROOT)/vulkan/x86_64/include \
        $(LIB_ROOT)/cglm/include \
        $(LIB_ROOT)/stb \
        $(LIB_ROOT)/lua-5.5.0/src

    LIB_DIRS = \
        $(LIB_ROOT)/vulkan/x86_64/lib \
        $(LIB_ROOT)/lua-5.5.0/src

    LIBS = shaderc_combined stdc++ vulkan m dl X11 pthread GL lua

    RM = rm -f
    RMDIR = rm -rf
    MKDIR = mkdir -p $(SPIRV_DIR)
    NULLDEV = >/dev/null 2>&1
endif

# --- 2. Compiler Flags ---
INC_FLAGS = $(addprefix -I, $(INCLUDE_DIRS))
LDFLAGS = $(addprefix -L, $(LIB_DIRS)) $(addprefix -l, $(LIBS))
CFLAGS = -g -O0 -march=native $(INC_FLAGS) -MMD -MP

# --- 3. Source & Object Discovery ---
SRC = $(wildcard src/*.c)
BUILD_DIR = build
OBJ = $(patsubst src/%.c, $(BUILD_DIR)/%.o, $(SRC))
DEP = $(OBJ:.o=.d)

# --- 4. Shader Discovery ---
SHADER_SRC = $(wildcard glsl/*)
SPIRV_DIR = spv
# Simplified shader mapping: just append .spv to the filename
SHADER_SPV = $(patsubst glsl/%, $(SPIRV_DIR)/%.spv, $(SHADER_SRC))

# --- 5. Rules ---
all: shaders $(OUT)

# Link
$(OUT): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile C (with auto-dir creation)
$(BUILD_DIR)/%.o: src/%.c
	@$(MKDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile Shaders
$(SPIRV_DIR)/%.spv: glsl/%
	@$(MKDIR)
	$(GLSLC) $< -o $@

shaders: $(SHADER_SPV)

# Include dependency files
-include $(DEP)

clean:
	-$(RM) $(OUT) $(OBJ) $(DEP) $(SHADER_SPV) $(NULLDEV)
	-$(RMDIR) $(SPIRV_DIR) $(BUILD_DIR) $(NULLDEV)

.PHONY: all shaders clean
