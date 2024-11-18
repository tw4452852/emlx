defmodule Emlx do
  @moduledoc """
  Documentation for `Emlx`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> Emlx.hello()
      :world

  """
  def hello do
    :nif_not_loaded
  end

  @on_load :on_load
  def on_load do
    # path = Path.join(System.get_env("HOME"), ".local/lib/libmlx")
    path = "/Users/samrat/code/emlx/cache/libemlx"
    :ok = :erlang.load_nif(path, 0)
  end
end
