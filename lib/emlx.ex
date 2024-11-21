defmodule EMLX do
  alias EMLX.NIF, as: NIF

  defguard is_tensor(device, ref) when is_reference(ref) and is_atom(device)

  def ones(shape, type, device), do: NIF.ones(shape, type, device) |> unwrap_tensor!(device)
  def zeros(shape, type, device), do: NIF.zeros(shape, type, device) |> unwrap_tensor!(device)

  ## Non-dirty non-tensor return values
  def scalar_type({device, ref}) when is_tensor(device, ref),
    do: NIF.scalar_type(ref) |> unwrap!()

  def shape({device, ref}) when is_tensor(device, ref),
    do: NIF.shape(ref) |> unwrap!()

  def to_blob({device, ref}) when is_tensor(device, ref),
    do: NIF.to_blob(ref) |> unwrap!()

  def to_type({device, ref}, type) when is_tensor(device, ref),
    do: NIF.to_type(ref, type) |> unwrap_tensor!(device)

  def to_blob({device, ref}, limit) when is_tensor(device, ref),
    do: NIF.to_blob(ref, limit) |> unwrap!()

  def from_blob(shape, type, binary, device),
    do: NIF.from_blob(shape, type, binary) |> unwrap_tensor!(device)

  def scalar_tensor(value, type, device),
    do: NIF.scalar_tensor(value, type) |> unwrap_tensor!(device)

  def sum({device, ref}, axes, keep_dims, result_type) when is_tensor(device, ref) do
    NIF.sum(ref, axes, keep_dims, result_type) |> unwrap_tensor!(device)
  end

  def eye(m, n, type, device), do: NIF.eye(m, n, type, device) |> unwrap_tensor!(device)

  def broadcast_to({device, ref}, shape),
    do: NIF.broadcast_to(ref, shape, device) |> unwrap_tensor!(device)

  def tensordot({device, refA} = tensorA, {_, refB} = tensorB, axes_a, axes_b),
    do: NIF.tensordot(refA, refB, axes_a, axes_b, device) |> unwrap_tensor!(device)

  def abs({device, ref} = tensor),
    do: NIF.abs(ref, device) |> unwrap_tensor!(device)

  def add({device, refA} = tensorA, {_, refB} = tensorB),
    do: NIF.add(refA, refB, device) |> unwrap_tensor!(device)

  def subtract({device, refA} = tensorA, {_, refB} = tensorB),
    do: NIF.subtract(refA, refB, device) |> unwrap_tensor!(device)

  def multiply({device, refA} = tensorA, {_, refB} = tensorB),
    do: NIF.multiply(refA, refB, device) |> unwrap_tensor!(device)

  def equal({device, refA} = tensorA, {_, refB} = tensorB),
    do: NIF.equal(refA, refB, device) |> unwrap_tensor!(device)

  def not_equal({device, refA} = tensorA, {_, refB} = tensorB),
    do: NIF.not_equal(refA, refB, device) |> unwrap_tensor!(device)

  def less_equal({device, refA} = tensorA, {_, refB} = tensorB),
    do: NIF.less_equal(refA, refB, device) |> unwrap_tensor!(device)

  def greater_equal({device, refA} = tensorA, {_, refB} = tensorB),
    do: NIF.greater_equal(refA, refB, device) |> unwrap_tensor!(device)

  def reshape({device, ref}, shape),
    do: NIF.reshape(ref, shape, device) |> unwrap_tensor!(device)

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, result}), do: result
  defp unwrap!({:error, error}), do: raise("EMLX: " <> List.to_string(error))

  defp unwrap_tensor!(tagged_result, device) do
    case unwrap!(tagged_result) do
      ref when is_reference(ref) ->
        {device, ref}

      list when is_list(list) ->
        Enum.map(list, &{device, &1})

      tuple when is_tuple(tuple) ->
        tuple |> Tuple.to_list() |> Enum.map(&{device, &1}) |> List.to_tuple()
    end
  end
end
