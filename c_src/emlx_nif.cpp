#include "erl_nif.h"

static ERL_NIF_TERM hello(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return enif_make_atom(env, "hello_from_nif");
}

static ErlNifFunc nif_funcs[] = {
    {"hello", 0, hello}
};

ERL_NIF_INIT(Elixir.Emlx, nif_funcs, NULL, NULL, NULL, NULL)
