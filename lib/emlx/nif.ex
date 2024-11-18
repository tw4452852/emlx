defmodule EMLX.NIF do
  @moduledoc """
  Elixir bindings for MLX array operations.
  """

  def zeros(_shape, _type, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def ones(_shape, _type, _device) do
    :erlang.nif_error(:nif_not_loaded)
  end

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

  def to_blob(_array) do
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
