defmodule EMLX.NIF do
  @moduledoc """
  Elixir bindings for MLX array operations.
  """

  @doc """
  Creates an array filled with zeros.

  ## Parameters
    - shape: A list of integers specifying the dimensions of the array

  ## Examples
      iex> Emlx.zeros([2, 3])
      # Returns a 2x3 array filled with zeros
  """
  def zeros(_shape) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Creates an array filled with ones.

  ## Parameters
    - shape: A list of integers specifying the dimensions of the array

  ## Examples
      iex> Emlx.ones([2, 3])
      # Returns a 2x3 array filled with ones
  """
  def ones(_shape) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Gets the data type of an array.

  ## Parameters
    - array: An MLX array

  ## Examples
      iex> array = Emlx.zeros([2, 3])
      iex> Emlx.scalar_type(array)
      # Returns the data type (e.g., :float32)
  """
  def scalar_type(_array) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def sum(_array, _axes, _keep_dims) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def shape(_array) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_type(_array, _type) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def to_blob(_array, _limit) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def from_blob(_shape, _type, _binary) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @on_load :load_nifs
  def load_nifs do
    path = :filename.join(:code.priv_dir(:emlx), ~c"libemlx")
    :erlang.load_nif(path, 0)
  end

  def scalar_tensor(_value, _type) do
    :erlang.nif_error(:nif_not_loaded)
  end
end
