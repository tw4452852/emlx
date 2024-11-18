defmodule EMLX do
  alias EMLX.NIF, as: NIF

  # FIXME: Add back device
  defguard is_tensor(ref) when is_reference(ref)

  ## Non-dirty non-tensor return values
  def scalar_type(ref) when is_tensor(ref), do: NIF.scalar_type(ref) |> unwrap!()
  def shape(ref) when is_tensor(ref), do: NIF.shape(ref) |> unwrap!()
  def to_blob(ref, limit) when is_tensor(ref), do: NIF.to_blob(ref, limit) |> unwrap!()
  def from_blob(shape, type, binary), do: NIF.from_blob(shape, type, binary) |> unwrap!()

  # TODO: Use macros like Torchx
  def to_type(ref, type) when is_tensor(ref), do: NIF.to_type(ref, type) |> unwrap!()
  def ones(shape), do: NIF.ones(shape) |> unwrap!()
  def zeros(shape), do: NIF.zeros(shape) |> unwrap!()

  def scalar_tensor(value, type), do: NIF.scalar_tensor(value, type) |> unwrap!()

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("EMLX: " <> List.to_string(error))
end
