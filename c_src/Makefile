# Environment variables passed via elixir_make
# ERTS_INCLUDE_DIR
# MIX_APP_PATH
ERTS_INCLUDE_DIR=/Users/samrat/.asdf/installs/erlang/26.1.2/erts-14.1.1/include
MIX_APP_PATH=/Users/samrat/code/emlx

# MLX configuration
# MLX configuration
MLX_INSTALL_DIR =/Users/samrat/.local
MLX_LIB_DIR = $(MLX_INSTALL_DIR)/lib
MLX_INCLUDE_DIR = $(MLX_INSTALL_DIR)/include

# MLX_BUILD_DIR = c_src/mlx/build
# MLX_LIB_DIR = $(MLX_BUILD_DIR)/mlx
# MLX_INCLUDE_DIR = c_src/mlx

# Cache configuration
EMLX_CACHE_DIR = cache
EMLX_CACHE_SO = $(EMLX_CACHE_DIR)/libemlx.so
EMLX_CACHE_OBJ_DIR = $(EMLX_CACHE_DIR)/objs
EMLX_CACHE_LIB_DIR = $(EMLX_CACHE_DIR)/mlx/lib

# Private configuration
PRIV_DIR = $(MIX_APP_PATH)/priv
EMLX_SO = $(PRIV_DIR)/libemlx.so
EMLX_LIB_DIR = $(PRIV_DIR)/mlx/lib

# Build flags
CFLAGS = -fPIC -I$(ERTS_INCLUDE_DIR) -I$(MLX_INCLUDE_DIR) -Wall \
         -std=c++17 -O3

LDFLAGS = -L$(MLX_LIB_DIR) -lmlx -shared

# Platform-specific settings
UNAME_S = $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
    LDFLAGS += -flat_namespace -undefined dynamic_lookup -rpath @loader_path/mlx/lib
else
    LDFLAGS += -Wl,-rpath,'$$ORIGIN/mlx/lib'
endif

# Source files
SOURCES = c_src/emlx_nif.cpp
OBJECTS = $(patsubst c_src/%.cpp,$(EMLX_CACHE_OBJ_DIR)/%.o,$(SOURCES))

# Main targets
all: $(EMLX_SO)

$(EMLX_SO): $(EMLX_CACHE_SO)
	@ mkdir -p $(PRIV_DIR)
	@ mkdir -p $(PRIV_DIR)/mlx
	@ if [ "${MIX_BUILD_EMBEDDED}" = "true" ]; then \
		cp -a $(MLX_LIB_DIR) $(EMLX_LIB_DIR) ; \
		cp -a $(EMLX_CACHE_SO) $(EMLX_SO) ; \
	else \
		ln -sf ../$(MLX_LIB_DIR) $(EMLX_LIB_DIR) ; \
		ln -sf ../$(EMLX_CACHE_SO) $(EMLX_SO) ; \
	fi

$(EMLX_CACHE_OBJ_DIR)/%.o: c_src/%.cpp
	@ mkdir -p $(EMLX_CACHE_OBJ_DIR)
	$(CXX) $(CFLAGS) -c $< -o $@

$(EMLX_CACHE_SO): $(OBJECTS)
	@ mkdir -p $(EMLX_CACHE_LIB_DIR)
	cp $(MLX_LIB_DIR)/libmlx.dylib $(EMLX_CACHE_LIB_DIR)/
	cp $(MLX_LIB_DIR)/mlx.metallib $(EMLX_CACHE_LIB_DIR)/
	$(CXX) $(OBJECTS) -o $(EMLX_CACHE_SO) $(LDFLAGS)

clean:
	rm -rf $(EMLX_CACHE_DIR)
	rm -rf $(PRIV_DIR)

.PHONY: all clean