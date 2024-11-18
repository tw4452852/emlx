defmodule Emlx do
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

  @on_load :load_nifs
  def load_nifs do
    path = :filename.join(:code.priv_dir(:emlx), ~c"libemlx")
    :erlang.load_nif(path, 0)
  end
end
