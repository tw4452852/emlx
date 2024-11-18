defmodule EMLX.Backend do
  @behaviour Nx.Backend

  alias Nx.Tensor, as: T
  alias EMLX.Backend, as: MB

  defstruct [:ref, :shape, :type, :data]

  @doc """
  Converts from an Nx tensor to an MLX array.
  """
  def from_nx(%T{data: %MB{ref: device_ref}}), do: device_ref
  def from_nx(%T{} = tensor), do: Nx.backend_transfer(tensor, MB) |> from_nx()

  @doc """
  Converts an MLX array back to an Nx tensor.
  """
  def to_nx(ref, %T{type: type, shape: shape} = t) when is_reference(ref) do
    # Get the MLX array's type
    mlx_type = EMLX.scalar_type(ref)

    # Convert if needed (similar to the torch byte conversion)
    array =
      if needs_type_conversion?(type, mlx_type) do
        EMLX.to_type(ref, nx_type_to_mlx(type))
      else
        ref
      end

    %T{t | data: %MB{ref: check_shape_and_type!(array, shape, type)}}
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
    |> maybe_add_signature(tensor)
  end

  @impl true
  def to_binary(tensor, limit) do
    blob = EMLX.to_blob(from_nx(tensor), limit)

    case tensor.type do
      {:u, 16} -> for <<x::32-native <- blob>>, do: <<x::16-native>>, into: <<>>
      {:u, 32} -> for <<x::64-native <- blob>>, do: <<x::32-native>>, into: <<>>
      _ -> blob
    end
  end

  defp maybe_add_signature(result, %T{data: %MB{ref: _}}) do
    Inspect.Algebra.concat([
      "EMLX.Backend",
      Inspect.Algebra.line(),
      result
    ])
  end

  # Helper functions
  defp needs_type_conversion?({:u, 8}, :bool), do: true
  defp needs_type_conversion?(_, _), do: false

  defp nx_type_to_mlx({:u, 8}), do: :uint8
  defp nx_type_to_mlx({:u, 16}), do: :uint16
  defp nx_type_to_mlx({:u, 32}), do: :uint32
  defp nx_type_to_mlx({:u, 64}), do: :uint64
  defp nx_type_to_mlx({:s, 8}), do: :int8
  defp nx_type_to_mlx({:s, 16}), do: :int16
  defp nx_type_to_mlx({:s, 32}), do: :int32
  defp nx_type_to_mlx({:s, 64}), do: :int64
  defp nx_type_to_mlx({:f, 16}), do: :float16
  defp nx_type_to_mlx({:f, 32}), do: :float32
  defp nx_type_to_mlx(:bool), do: :bool

  defp check_shape_and_type!(array, expected_shape, expected_type) do
    actual_shape = EMLX.shape(array)
    actual_type = EMLX.scalar_type(array) |> mlx_type_to_nx()

    if actual_shape != expected_shape do
      raise ArgumentError, """
      Shape mismatch in MLX array conversion:
      Expected shape: #{inspect(expected_shape)}
      Got shape: #{inspect(actual_shape)}
      """
    end

    if actual_type != expected_type do
      raise ArgumentError, """
      Type mismatch in MLX array conversion:
      Expected type: #{inspect(expected_type)}
      Got type: #{inspect(actual_type)}
      """
    end

    array
  end

  defp mlx_type_to_nx(:uint8), do: {:u, 8}
  defp mlx_type_to_nx(:uint16), do: {:u, 16}
  defp mlx_type_to_nx(:uint32), do: {:u, 32}
  defp mlx_type_to_nx(:uint64), do: {:u, 64}
  defp mlx_type_to_nx(:int8), do: {:s, 8}
  defp mlx_type_to_nx(:int16), do: {:s, 16}
  defp mlx_type_to_nx(:int32), do: {:s, 32}
  defp mlx_type_to_nx(:int64), do: {:s, 64}
  defp mlx_type_to_nx(:float16), do: {:f, 16}
  defp mlx_type_to_nx(:float32), do: {:f, 32}
  defp mlx_type_to_nx(:bool), do: :bool
end
