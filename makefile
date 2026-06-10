CC = gcc
GLSLC = glslc

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
    MKDIR = mkdir -p $(SPIRV_DIR) && mkdir -p build
    NULLDEV = >/dev/null 2>&1
endif

INC_FLAGS = $(addprefix -I, $(INCLUDE_DIRS))
LDFLAGS = $(addprefix -L, $(LIB_DIRS)) $(addprefix -l, $(LIBS))
CFLAGS = -g -O0 -march=native $(INC_FLAGS) -MMD -MP

SRC = $(wildcard src/*.c)
BUILD_DIR = build
OBJ = $(patsubst src/%.c, $(BUILD_DIR)/%.o, $(SRC))
DEP = $(OBJ:.o=.d)

all: $(OUT)

$(OUT): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

-include $(DEP)

clean:
	-$(RM) $(OUT) $(OBJ) $(DEP) $(SHADER_SPV) $(NULLDEV)
	-$(RMDIR) $(SPIRV_DIR) $(BUILD_DIR) $(NULLDEV)

.PHONY: all clean
