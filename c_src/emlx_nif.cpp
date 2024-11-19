#include "erl_nif.h"
#include "mlx/mlx.h"
#include "nx_nif_utils.hpp"
#include <map>
#include <string>
#include <numeric>
#include <iostream>
using namespace mlx::core;

std::map<const std::string, const mlx::core::Dtype> dtypes = {
    {"bool", mlx::core::bool_},
    {"uint8", mlx::core::uint8},
    {"uint16", mlx::core::uint16},
    {"uint32", mlx::core::uint32},
    {"uint64", mlx::core::uint64},
    {"int8", mlx::core::int8},
    {"int16", mlx::core::int16},
    {"int32", mlx::core::int32},
    {"int64", mlx::core::int64},
    {"float16", mlx::core::float16},
    {"float32", mlx::core::float32},
    {"bfloat16", mlx::core::bfloat16},
    {"complex64", mlx::core::complex64}
};

std::map<const std::string, const uint8_t> dtype_sizes = {
    {"bool", mlx::core::bool_.size()},
    {"uint8", mlx::core::uint8.size()},
    {"uint16", mlx::core::uint16.size()},
    {"uint32", mlx::core::uint32.size()},
    {"uint64", mlx::core::uint64.size()},
    {"int8", mlx::core::int8.size()},
    {"int16", mlx::core::int16.size()},
    {"int32", mlx::core::int32.size()},
    {"int64", mlx::core::int64.size()},
    {"float16", mlx::core::float16.size()},
    {"float32", mlx::core::float32.size()},
    {"bfloat16", mlx::core::bfloat16.size()},
    {"complex64", mlx::core::complex64.size()}
};

inline mlx::core::Dtype string2dtype(const std::string &atom) {
    auto it = dtypes.find(atom);
    if (it != dtypes.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown dtype: " + atom);
}

inline const std::string *dtype2string(const mlx::core::Dtype dtype) {
    for (const auto& pair : dtypes) {
        if (pair.second == dtype) {
            return &pair.first;
        }
    }
    return nullptr;
}

inline const mlx::core::Device string2device(const std::string &atom) {
    if (atom == "cpu") {
        return mlx::core::Device(mlx::core::Device::DeviceType::cpu, 0);
    } else if (atom == "gpu") {
        return mlx::core::Device(mlx::core::Device::DeviceType::gpu, 0);
    }
    throw std::runtime_error("Unknown device: " + atom);
}

// Class to manage the refcount of MLX tensors
class TensorP {
 public:
  TensorP(ErlNifEnv *env, const ERL_NIF_TERM arg) : ptr(nullptr) {
    // setup
    if (!enif_get_resource(env, arg, TENSOR_TYPE, (void **)&ptr)) {
      err = nx::nif::error(env, "Unable to get tensor param in NIF");
      return;
    }

    refcount = (std::atomic<int> *)(ptr + 1);
    deleted = (std::atomic_flag *)(refcount + 1);

    if (refcount->load() == 0) {
      // already deallocated
      ptr = nullptr;
      err = nx::nif::error(env, "Tensor has been deallocated");
      return;
    }

    if (is_valid()) {
      // increase reference count
      ++(*refcount);
    }
  }

  ~TensorP() {
    if (is_valid()) {
      // decrease reference count
      if (refcount->fetch_sub(1) == 0) {
        ptr->~array();  // Call MLX tensor destructor
      }
    }
  }

  bool deallocate() {
    if (is_valid() && atomic_flag_test_and_set(deleted) == false) {
      --(*refcount);
      return true;
    } else {
      return false;
    }
  }

  mlx::core::array *data() const {
    return ptr;
  }

  bool is_valid() const {
    return ptr != nullptr;
  }

  ERL_NIF_TERM error() {
    return err;
  }

 private:
  mlx::core::array *ptr;
  std::atomic<int> *refcount;
  std::atomic_flag *deleted;
  ERL_NIF_TERM err;
};

#define CATCH()                                                  \
  catch (const std::exception& e) {                             \
    std::ostringstream msg;                                     \
    msg << e.what() << " in NIF." << __func__ << "/" << argc;  \
    return nx::nif::error(env, msg.str().c_str());             \
  }                                                             \
  catch (...) {                                                 \
    return nx::nif::error(env, "Unknown error occurred");       \
  }

#define TENSOR(A)                                            \
  try {                                                      \
    return nx::nif::ok(env, create_tensor_resource(env, A)); \
  }                                                          \
  CATCH()

ERL_NIF_TERM
create_tensor_resource(ErlNifEnv *env, mlx::core::array tensor) {
  ERL_NIF_TERM ret;
  mlx::core::array *tensorPtr;
  std::atomic<int> *refcount;

  tensorPtr = (mlx::core::array *)enif_alloc_resource(TENSOR_TYPE, sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
  if (tensorPtr == NULL)
    return enif_make_badarg(env);

  new (tensorPtr) mlx::core::array(std::move(tensor));
  refcount = new (tensorPtr + 1) std::atomic<int>(1);
  new (refcount + 1) std::atomic_flag();

  ret = enif_make_resource(env, tensorPtr);
  enif_release_resource(tensorPtr);

  return ret;
}

#define NIF(NAME) ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

#define PARAM(ARGN, TYPE, VAR) \
  TYPE VAR;                    \
  GET(ARGN, VAR)

#define TENSOR_PARAM(ARGN, VAR)      \
  TensorP VAR##_tp(env, argv[ARGN]); \
  mlx::core::array *VAR;                \
  if (!VAR##_tp.is_valid()) {        \
    return VAR##_tp.error();         \
  } else {                           \
    VAR = VAR##_tp.data();           \
  }

#define LIST_PARAM(ARGN, TYPE, VAR)             \
TYPE VAR;                                      \
if (!nx::nif::get_list(env, argv[ARGN], VAR)) \
return nx::nif::error(env, "Unable to get " #VAR " list param.");

NIF(scalar_type) {
  TENSOR_PARAM(0, t);

  const std::string *type_name = dtype2string(t->dtype());

  if (type_name != nullptr)
    return nx::nif::ok(env, enif_make_atom(env, type_name->c_str()));
  else
    return nx::nif::error(env, "Could not determine tensor type.");
}

NIF(sum) {
    TENSOR_PARAM(0, t);  
    LIST_PARAM(1, std::vector<int>, axes);
    PARAM(2, bool, keep_dims);

    TENSOR(mlx::core::sum(*t, axes, keep_dims));
}

NIF(shape) {
  TENSOR_PARAM(0, t);

  std::vector<ERL_NIF_TERM> sizes;
  for (int64_t dim = 0; dim < t->ndim(); dim++)
    sizes.push_back(nx::nif::make(env, static_cast<int64_t>(t->shape()[dim])));

  return nx::nif::ok(env, enif_make_tuple_from_array(env, sizes.data(), sizes.size()));
}

NIF(ones) {
  SHAPE_PARAM(0, shape);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::ones(shape, type, device));
}

NIF(zeros) {
  SHAPE_PARAM(0, shape);
  TYPE_PARAM(1, type);
  DEVICE_PARAM(2, device);

  TENSOR(mlx::core::zeros(shape, type, device));
}

NIF(to_type) {
    TENSOR_PARAM(0, t);
    
    char type_str[32];
    if (!enif_get_atom(env, argv[1], type_str, sizeof(type_str), ERL_NIF_LATIN1)) {
        return enif_make_badarg(env);
    }

    try {
        mlx::core::Dtype new_dtype = string2dtype(type_str);
        mlx::core::array result = mlx::core::astype(*t, new_dtype);
        
        // Allocate and return new tensor resource
        void* resource = enif_alloc_resource(TENSOR_TYPE, 
            sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
        
        if (!resource) {
            return enif_make_tuple2(env, 
                enif_make_atom(env, "error"),
                enif_make_atom(env, "resource_allocation_failed"));
        }

        new (resource) mlx::core::array(std::move(result));
        
        // Initialize refcount and deleted flag
        std::atomic<int>* refcount = (std::atomic<int>*)(((mlx::core::array*)resource) + 1);
        std::atomic_flag* deleted = (std::atomic_flag*)(refcount + 1);
        new (refcount) std::atomic<int>(1);
        deleted->clear();

        ERL_NIF_TERM term = enif_make_resource(env, resource);
        enif_release_resource(resource);

        return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);
    } catch (const std::exception& e) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    }
}

NIF(to_blob) {
  TENSOR_PARAM(0, t);
  
  try {
    // Evaluate the array to ensure data is available
    mlx::core::eval(*t);
    
    size_t byte_size = t->nbytes();
    int64_t limit = 0;
    bool has_received_limit = (argc == 2);

    if (has_received_limit) {
      PARAM(1, int64_t, param_limit);
      limit = param_limit;
      byte_size = limit * t->itemsize();
    }

    // Get raw pointer to data
    const void* data_ptr = t->data<void>();
    if (data_ptr == nullptr) {
      return nx::nif::error(env, "Failed to get tensor data");
    }

    return nx::nif::ok(env, enif_make_resource_binary(env, t, data_ptr, byte_size));
  } catch (const std::exception& e) {
    return nx::nif::error(env, e.what());
  } catch (...) {
    return nx::nif::error(env, "Unknown error during data copy");
  }
}

uint64_t elem_count(std::vector<int> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
}

NIF(from_blob) {
  BINARY_PARAM(0, blob);
  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, type);

  if (blob.size / dtype_sizes[type_atom] < elem_count(shape))
    return nx::nif::error(env, "Binary size is too small for the requested shape");

  try {
    // Allocate MLX buffer and copy data from blob
    size_t byte_size = blob.size;
    allocator::Buffer mlx_buf = allocator::malloc(byte_size);
    void* buf_ptr = mlx_buf.raw_ptr();
    
    // Copy binary data to MLX buffer
    std::memcpy(buf_ptr, blob.data, byte_size);

    // Create deleter for the buffer
    auto deleter = [](allocator::Buffer buf) {
      allocator::free(buf);
    };

    // Create MLX array from the buffer
    TENSOR(mlx::core::array(mlx_buf, shape, type, deleter));
  } catch (const std::exception& e) {
    return nx::nif::error(env, e.what());
  } catch (...) {
    return nx::nif::error(env, "Unknown error creating tensor from binary data");
  }
}

NIF(scalar_tensor) {
  SCALAR_PARAM(0, scalar);
  TYPE_PARAM(1, type);

  TENSOR( mlx::core::array(scalar, type))
}


static void free_tensor(ErlNifEnv* env, void* obj) {
    mlx::core::array* arr = static_cast<mlx::core::array*>(obj);
    if (arr != nullptr) {
        arr->~array();
    }
}

static int open_resource_type(ErlNifEnv* env) {
    const char* name = "MLXArray";
    ErlNifResourceFlags flags = (ErlNifResourceFlags)(ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);

    TENSOR_TYPE = enif_open_resource_type(env, NULL, name, free_tensor, flags, NULL);
    if (TENSOR_TYPE == NULL) {
        return -1;
    }
    return 0;
}

// In your module load function:
static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) {
    if (open_resource_type(env) != 0) {
        return -1;
    }
    return 0;
}

static ErlNifFunc nif_funcs[] = {
    {"scalar_type", 1, scalar_type},
    {"sum", 3, sum},
    {"shape", 1, shape},
    {"to_type", 2, to_type},
    {"to_blob", 1, to_blob},
    {"to_blob", 2, to_blob},
    {"from_blob", 3, from_blob},
    {"scalar_tensor", 2, scalar_tensor},
    {"ones", 3, ones},
    {"zeros", 3, zeros},
};


// Update the NIF initialization
ERL_NIF_INIT(Elixir.EMLX.NIF, nif_funcs, load, NULL, NULL, NULL)
