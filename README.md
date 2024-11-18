# Emlx

**TODO**

- [ ] from_blob isn't working properly. Returns wrong tensor

  iex(1)> Nx.from_binary(<<1::32,2::32>>, :u32)

#Nx.Tensor<
  u32[2]
  EMLX.Backend
  [0, 0]
>
iex(2)>

## Current setup(probably needs to be improved)

MLX source is placed in `priv/mlx-src`

```
cmake -B priv/mlx-src/build -DCMAKE_INSTALL_PREFIX=$HOME/.local -DBUILD_SHARED_LIBS=ON
make -j8 -C priv/mlx-src/build
make -C priv/mlx-src/build install
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

