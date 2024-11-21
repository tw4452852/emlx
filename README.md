# EMLX

**TODO**

- [ ] broadcasting with eye gives incorrect tensor 

MLX seems to be doing th right thing, but the expansion gets zero'd out by the time we get it for some reason.
```
  {:cpu, #Reference<0.4099841517.3590193156.176885>}
iex(2)>
nil
iex(3)> broadcasted = EMLX.broadcast_to(base_eye, {2,2,2})Tensor: array([1, 0], dtype=uint8) array([0, 1], dtype=uint8)
                                  Result shape: (2,2,2)
                                                       Result: array([[1, 0],
                                                                                    [0, 1]], dtype=uint8)
                   array([[1, 0],
                                        [0, 1]], dtype=uint8)

```

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

