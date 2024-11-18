#include "erl_nif.h"
#include "mlx/mlx.h"
#include "nx_nif_utils.hpp"
#include <map>
#include <string>

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

// Class to manage the refcount of MLX arrays
class ArrayP {
 public:
  ArrayP(ErlNifEnv *env, const ERL_NIF_TERM arg) : ptr(nullptr) {
    // setup
    if (!enif_get_resource(env, arg, ARRAY_TYPE, (void **)&ptr)) {
      err = nx::nif::error(env, "Unable to get array param in NIF");
      return;
    }

    refcount = (std::atomic<int> *)(ptr + 1);
    deleted = (std::atomic_flag *)(refcount + 1);

    if (refcount->load() == 0) {
      // already deallocated
      ptr = nullptr;
      err = nx::nif::error(env, "Array has been deallocated");
      return;
    }

    if (is_valid()) {
      // increase reference count
      ++(*refcount);
    }
  }

  ~ArrayP() {
    if (is_valid()) {
      // decrease reference count
      if (refcount->fetch_sub(1) == 0) {
        ptr->~array();  // Call MLX array destructor
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

#define NIF(NAME) ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

#define TENSOR_PARAM(ARGN, VAR)      \
  ArrayP VAR##_tp(env, argv[ARGN]); \
  mlx::core::array *VAR;                \
  if (!VAR##_tp.is_valid()) {        \
    return VAR##_tp.error();         \
  } else {                           \
    VAR = VAR##_tp.data();           \
  }


NIF(scalar_type) {
  TENSOR_PARAM(0, t);

  const std::string *type_name = dtype2string(t->dtype());

  if (type_name != nullptr)
    return nx::nif::ok(env, enif_make_atom(env, type_name->c_str()));
  else
    return nx::nif::error(env, "Could not determine array type.");
}

NIF(make_zeros) {
    if (argc != 1) {
        return enif_make_badarg(env);
    }

    unsigned int length;
    if (!enif_get_list_length(env, argv[0], &length)) {
        return enif_make_badarg(env);
    }

    std::vector<int> shape;
    ERL_NIF_TERM head, tail = argv[0];
    
    for (unsigned int i = 0; i < length; i++) {
        int dim;
        if (!enif_get_list_cell(env, tail, &head, &tail) ||
            !enif_get_int(env, head, &dim)) {
            return enif_make_badarg(env);
        }
        shape.push_back(dim);
    }

    try {
        // Create MLX array filled with zeros
        mlx::core::array result = mlx::core::zeros(shape, mlx::core::float32);
        
        // Allocate resource for the array
        void* resource = enif_alloc_resource(ARRAY_TYPE, 
            sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
        
        if (!resource) {
            return enif_make_tuple2(env, 
                enif_make_atom(env, "error"),
                enif_make_atom(env, "resource_allocation_failed"));
        }

        // Copy the array into the resource
        new (resource) mlx::core::array(std::move(result));
        
        // Initialize refcount and deleted flag
        std::atomic<int>* refcount = (std::atomic<int>*)(((mlx::core::array*)resource) + 1);
        std::atomic_flag* deleted = (std::atomic_flag*)(refcount + 1);
        new (refcount) std::atomic<int>(1);
        deleted->clear();

        // Create Erlang term
        ERL_NIF_TERM term = enif_make_resource(env, resource);
        enif_release_resource(resource);

        return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

    } catch (const std::exception& e) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    } catch (...) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_atom(env, "unknown_error"));
    }
}

NIF(make_ones) {
    if (argc != 1) {
        return enif_make_badarg(env);
    }

    unsigned int length;
    if (!enif_get_list_length(env, argv[0], &length)) {
        return enif_make_badarg(env);
    }

    std::vector<int> shape;
    ERL_NIF_TERM head, tail = argv[0];
    
    for (unsigned int i = 0; i < length; i++) {
        int dim;
        if (!enif_get_list_cell(env, tail, &head, &tail) ||
            !enif_get_int(env, head, &dim)) {
            return enif_make_badarg(env);
        }
        shape.push_back(dim);
    }

    try {
        // Create MLX array filled with ones
        mlx::core::array result = mlx::core::ones(shape, mlx::core::float32);
        
        // Allocate resource for the array
        void* resource = enif_alloc_resource(ARRAY_TYPE, 
            sizeof(mlx::core::array) + sizeof(std::atomic<int>) + sizeof(std::atomic_flag));
        
        if (!resource) {
            return enif_make_tuple2(env, 
                enif_make_atom(env, "error"),
                enif_make_atom(env, "resource_allocation_failed"));
        }

        // Copy the array into the resource
        new (resource) mlx::core::array(std::move(result));
        
        // Initialize refcount and deleted flag
        std::atomic<int>* refcount = (std::atomic<int>*)(((mlx::core::array*)resource) + 1);
        std::atomic_flag* deleted = (std::atomic_flag*)(refcount + 1);
        new (refcount) std::atomic<int>(1);
        deleted->clear();

        // Create Erlang term
        ERL_NIF_TERM term = enif_make_resource(env, resource);
        enif_release_resource(resource);

        return enif_make_tuple2(env, enif_make_atom(env, "ok"), term);

    } catch (const std::exception& e) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    } catch (...) {
        return enif_make_tuple2(env, 
            enif_make_atom(env, "error"),
            enif_make_atom(env, "unknown_error"));
    }
}

static ErlNifFunc nif_funcs[] = {
    {"zeros", 1, make_zeros},
    {"ones", 1, make_ones},
    {"scalar_type", 1, scalar_type}
};

static void free_array(ErlNifEnv* env, void* obj) {
    mlx::core::array* arr = static_cast<mlx::core::array*>(obj);
    if (arr != nullptr) {
        arr->~array();
    }
}

static int open_resource_type(ErlNifEnv* env) {
    const char* name = "MLXArray";
    ErlNifResourceFlags flags = (ErlNifResourceFlags)(ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);

    ARRAY_TYPE = enif_open_resource_type(env, NULL, name, free_array, flags, NULL);
    if (ARRAY_TYPE == NULL) {
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

// Update the NIF initialization
ERL_NIF_INIT(Elixir.Emlx, nif_funcs, load, NULL, NULL, NULL)
