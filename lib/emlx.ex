defmodule EMLX do
  alias EMLX.NIF, as: NIF

  defguard is_tensor(ref) when is_reference(ref)

  ## Non-dirty non-tensor return values

  def scalar_type(ref) when is_tensor(ref), do: NIF.scalar_type(ref) |> unwrap!()
  def shape(ref) when is_tensor(ref), do: NIF.shape(ref) |> unwrap!()
  def to_blob(ref, limit) when is_tensor(ref), do: NIF.to_blob(ref, limit) |> unwrap!()

  # TODO: Use macros like Torchx
  def to_type(ref, type) when is_tensor(ref), do: NIF.to_type(ref, type) |> unwrap!()
  def ones(shape), do: NIF.ones(shape) |> unwrap!()
  def zeros(shape), do: NIF.zeros(shape) |> unwrap!()

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("EMLX: " <> List.to_string(error))
end
