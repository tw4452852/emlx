#include "erl_nif.h"
#include "mlx/mlx.h"
#include "nx_nif_utils.hpp"
#include <map>
#include <numeric>
#include <string>

using namespace mlx::core;

std::map<const std::string, const mlx::core::Dtype> dtypes = {
    {"bool", mlx::core::bool_},         {"uint8", mlx::core::uint8},
    {"uint16", mlx::core::uint16},      {"uint32", mlx::core::uint32},
    {"uint64", mlx::core::uint64},      {"int8", mlx::core::int8},
    {"int16", mlx::core::int16},        {"int32", mlx::core::int32},
    {"int64", mlx::core::int64},        {"float16", mlx::core::float16},
    {"float32", mlx::core::float32},    {"bfloat16", mlx::core::bfloat16},
    {"complex64", mlx::core::complex64}};

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
    {"complex64", mlx::core::complex64.size()}};

inline mlx::core::Dtype string2dtype(const std::string &atom) {
  auto it = dtypes.find(atom);
  if (it != dtypes.end()) {
    return it->second;
  }
  throw std::runtime_error("Unknown dtype: " + atom);
}

inline const std::string *dtype2string(const mlx::core::Dtype dtype) {
  for (const auto &pair : dtypes) {
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
        ptr->~array(); // Call MLX tensor destructor
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

  mlx::core::array *data() const { return ptr; }

  bool is_valid() const { return ptr != nullptr; }

  ERL_NIF_TERM error() { return err; }

private:
  mlx::core::array *ptr;
  std::atomic<int> *refcount;
  std::atomic_flag *deleted;
  ERL_NIF_TERM err;
};

#define CATCH()                                                                \
  catch (const std::exception &e) {                                            \
    std::ostringstream msg;                                                    \
    msg << e.what() << " in NIF." << __func__ << "/" << argc;                  \
    return nx::nif::error(env, msg.str().c_str());                             \
  }                                                                            \
  catch (...) {                                                                \
    return nx::nif::error(env, "Unknown error occurred");                      \
  }

#define TENSOR(A)                                                              \
  try {                                                                        \
    return nx::nif::ok(env, create_tensor_resource(env, A));                   \
  }                                                                            \
  CATCH()

ERL_NIF_TERM
create_tensor_resource(ErlNifEnv *env, mlx::core::array tensor) {
  ERL_NIF_TERM ret;
  mlx::core::array *tensorPtr;
  std::atomic<int> *refcount;

  tensorPtr = (mlx::core::array *)enif_alloc_resource(
      TENSOR_TYPE, sizeof(mlx::core::array) + sizeof(std::atomic<int>) +
                       sizeof(std::atomic_flag));
  if (tensorPtr == NULL)
    return enif_make_badarg(env);

  new (tensorPtr) mlx::core::array(std::move(tensor));
  refcount = new (tensorPtr + 1) std::atomic<int>(1);
  new (refcount + 1) std::atomic_flag();

  ret = enif_make_resource(env, tensorPtr);
  enif_release_resource(tensorPtr);

  return ret;
}

#define NIF(NAME)                                                              \
  ERL_NIF_TERM NAME(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])

#define PARAM(ARGN, TYPE, VAR)                                                 \
  TYPE VAR;                                                                    \
  GET(ARGN, VAR)

#define TENSOR_PARAM(ARGN, VAR)                                                \
  TensorP VAR##_tp(env, argv[ARGN]);                                           \
  mlx::core::array *VAR;                                                       \
  if (!VAR##_tp.is_valid()) {                                                  \
    return VAR##_tp.error();                                                   \
  } else {                                                                     \
    VAR = VAR##_tp.data();                                                     \
  }

#define LIST_PARAM(ARGN, TYPE, VAR)                                            \
  TYPE VAR;                                                                    \
  if (!nx::nif::get_list(env, argv[ARGN], VAR))                                \
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
  TYPE_PARAM(3, result_type);

  // If axes is empty, sum over all dimensions
  // MLX sums over all dimensions ONLY if axes is not specified.
  // If an empty vector is passed, it will not sum over any dimensions.
  if (axes.empty()) {
    for (int i = 0; i < t->ndim(); ++i) {
      axes.push_back(i);
    }
  }

  auto result =
      mlx::core::sum(mlx::core::astype(*t, result_type), axes, keep_dims);

  TENSOR(result);
}

NIF(shape) {
  TENSOR_PARAM(0, t);

  std::vector<ERL_NIF_TERM> sizes;
  for (int64_t dim = 0; dim < t->ndim(); dim++)
    sizes.push_back(nx::nif::make(env, static_cast<int64_t>(t->shape()[dim])));

  return nx::nif::ok(
      env, enif_make_tuple_from_array(env, sizes.data(), sizes.size()));
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
  TYPE_PARAM(1, type);

  TENSOR(mlx::core::astype(*t, type));
}

NIF(to_blob) {
  ERL_NIF_TERM result;
  TENSOR_PARAM(0, t);

  size_t byte_size = t->nbytes();
  int limit = 0;
  bool has_received_limit = (argc == 2);

  if (has_received_limit) {
    PARAM(1, int, param_limit);
    limit = param_limit;
    byte_size = limit * t->itemsize();
  }

  // flatten the tensor to compensate for operations which return
  // a column-major tensor. t->flatten() is a no-op if the tensor
  // is already row-major, which was verified by printing t->data_ptr
  // and reshaped.data_ptr and confirming they had the same value.
  // We also slice if a limit was received and it doesn't encompass the full
  // tensor.
  mlx::core::array flattened = mlx::core::flatten(*t);
  mlx::core::array reshaped =
      (has_received_limit && byte_size < t->nbytes())
          ?
          // We only care about slicing the first dimension
          mlx::core::slice(flattened, std::vector<int>{0},
                           std::vector<int>{limit})
          : flattened;

  // Evaluate the array to ensure data is available
  mlx::core::eval(reshaped);

  // Get raw pointer to data
  const void *data_ptr = reshaped.data<void>();

  if (data_ptr == nullptr) {
    return nx::nif::error(env, "Failed to get tensor data");
  }

  void *result_data = (void *)enif_make_new_binary(env, byte_size, &result);
  memcpy(result_data, data_ptr, byte_size);
  return nx::nif::ok(env, result);
}

uint64_t elem_count(std::vector<int> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
}

NIF(from_blob) {
  BINARY_PARAM(0, blob);
  SHAPE_PARAM(1, shape);
  TYPE_PARAM(2, type);

  if (blob.size / dtype_sizes[type_atom] < elem_count(shape))
    return nx::nif::error(env,
                          "Binary size is too small for the requested shape");

  try {
    // Allocate MLX buffer and copy data from blob
    size_t byte_size = blob.size;
    allocator::Buffer mlx_buf = allocator::malloc(byte_size);
    void *buf_ptr = mlx_buf.raw_ptr();

    // Copy binary data to MLX buffer
    std::memcpy(buf_ptr, blob.data, byte_size);

    // Create deleter for the buffer
    auto deleter = [](allocator::Buffer buf) { allocator::free(buf); };

    // Create MLX array from the buffer
    TENSOR(mlx::core::array(mlx_buf, shape, type, deleter));
  } catch (const std::exception &e) {
    return nx::nif::error(env, e.what());
  } catch (...) {
    return nx::nif::error(env,
                          "Unknown error creating tensor from binary data");
  }
}

NIF(scalar_tensor) {
  SCALAR_PARAM(0, scalar);
  TYPE_PARAM(1, type);

  TENSOR(mlx::core::array(scalar, type))
}

NIF(eye) {
  PARAM(0, int, m);
  PARAM(1, int, n);
  TYPE_PARAM(2, type);
  DEVICE_PARAM(3, device);

  TENSOR(mlx::core::eye(m, n, 0, type, device));
}

NIF(broadcast_to) {
  TENSOR_PARAM(0, t);
  SHAPE_PARAM(1, shape);
  DEVICE_PARAM(2, device);

  auto result = mlx::core::broadcast_to(*t, shape, device);

  TENSOR(result);
}

NIF(tensordot) {
  TENSOR_PARAM(0, a);
  TENSOR_PARAM(1, b);
  LIST_PARAM(2, std::vector<int>, axes1);
  LIST_PARAM(3, std::vector<int>, axes2);
  DEVICE_PARAM(4, device);

  TENSOR(mlx::core::tensordot(*a, *b, axes1, axes2, device));
}

static void free_tensor(ErlNifEnv *env, void *obj) {
  mlx::core::array *arr = static_cast<mlx::core::array *>(obj);
  if (arr != nullptr) {
    arr->~array();
  }
}

static int open_resource_type(ErlNifEnv *env) {
  const char *name = "MLXArray";
  ErlNifResourceFlags flags =
      (ErlNifResourceFlags)(ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);

  TENSOR_TYPE =
      enif_open_resource_type(env, NULL, name, free_tensor, flags, NULL);
  if (TENSOR_TYPE == NULL) {
    return -1;
  }
  return 0;
}

// In your module load function:
static int load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  if (open_resource_type(env) != 0) {
    return -1;
  }
  return 0;
}

static ErlNifFunc nif_funcs[] = {
    {"scalar_type", 1, scalar_type},
    {"sum", 4, sum},
    {"shape", 1, shape},
    {"to_type", 2, to_type},
    {"to_blob", 1, to_blob},
    {"to_blob", 2, to_blob},
    {"from_blob", 3, from_blob},
    {"scalar_tensor", 2, scalar_tensor},
    {"ones", 3, ones},
    {"zeros", 3, zeros},
    {"eye", 4, eye},
    {"broadcast_to", 3, broadcast_to},
    {"tensordot", 5, tensordot},
};

// Update the NIF initialization
ERL_NIF_INIT(Elixir.EMLX.NIF, nif_funcs, load, NULL, NULL, NULL)
