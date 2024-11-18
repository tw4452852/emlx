#include "erl_nif.h"
#include "mlx/mlx.h"
#include <iostream>
using namespace mlx::core;

static ERL_NIF_TERM make_zeros(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
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

    array result = zeros(shape);
    // TODO: Convert MLX array to Erlang term
    // For now just return :ok
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM make_ones(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
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

     array result = ones(shape);

    std::cout << result << std::endl;

    size_t size = result.size();
    printf("Size: %zu\n", size);

    // Print array 
    // for (int i = 0; i < result.size(); i++) {
    //     printf("%f ", result[i]);
    // }
    // printf("\n");

    // TODO: Convert MLX array to Erlang term
    // For now just return :ok
    return enif_make_atom(env, "ok");
}

static ErlNifFunc nif_funcs[] = {
    {"zeros", 1, make_zeros},
    {"ones", 1, make_ones}
};

ERL_NIF_INIT(Elixir.Emlx, nif_funcs, NULL, NULL, NULL, NULL)
