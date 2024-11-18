# Emlx

**TODO: Add description**

## Current setup(probably needs to be improved)

MLX source is placed in `c_src/mlx`

```
cmake -B c_src/mlx/build -DCMAKE_INSTALL_PREFIX=$HOME/.local -DBUILD_SHARED_LIBS=ON
make -j8 -C c_src/mlx/build
make -C c_src/mlx/build install
```

libemlx.so built to `cache/`

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `emlx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:emlx, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/emlx>.

