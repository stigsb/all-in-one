"""Pure-PyTorch neighborhood attention, for devices NATTEN has no kernels for.

NATTEN ships CPU and CUDA kernels only, so Apple-silicon GPU (MPS) tensors have
no native path. These functions reproduce the semantics of
``natten.functional.natten{1,2}d{qkrpb,av}`` from NATTEN 0.15 exactly, using
only device-agnostic torch ops, at the cost of speed.

The index arithmetic mirrors ``get_window_start``/``get_pb_start`` in
``csrc/include/natten/cpu/naive/natten_cpu_commons.h``: a query's neighborhood
is centred where possible and clamped to stay in bounds at the edges, keeping
the window size constant.
"""

import torch


def _window_start(length: int, kernel_size: int, dilation: int, device) -> torch.Tensor:
  """First neighbor index for each query position. Shape: (length,)."""
  ns = kernel_size // 2
  i = torch.arange(length, device=device)
  if dilation <= 1:
    return torch.clamp(i - ns, min=0) + (i + ns >= length).long() * (length - i - ns - 1)

  imodd = i % dilation
  a = (length // dilation) * dilation
  b = length - a
  tail = torch.where(imodd < b,
                     length - b + imodd - 2 * ns * dilation,
                     a + imodd - kernel_size * dilation)
  return torch.where(i - ns * dilation < 0, imodd,
                     torch.where(i + ns * dilation >= length, tail, i - ns * dilation))


def _bias_start(length: int, kernel_size: int, dilation: int, device) -> torch.Tensor:
  """First relative-positional-bias index for each query position. Shape: (length,)."""
  ns = kernel_size // 2
  i = torch.arange(length, device=device)
  if dilation <= 1:
    return (ns
            + (i < ns).long() * (ns - i)
            + (i + ns >= length).long() * (length - i - 1 - ns))

  floor_div = lambda x, y: torch.div(x, y, rounding_mode='floor')
  return torch.where(i - ns * dilation < 0, kernel_size - 1 - floor_div(i, dilation),
                     torch.where(i + ns * dilation >= length, floor_div(length - i - 1, dilation),
                                 torch.full_like(i, ns)))


def _check(length: int, kernel_size: int, dilation: int, axis: str):
  if length < kernel_size * dilation:
    raise ValueError(
      f'{axis} of size {length} is too small for kernel size {kernel_size} '
      f'with dilation {dilation}.'
    )


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
  """query, key: (B, H, L, D); rpb: (H, 2 * kernel_size - 1) -> (B, H, L, kernel_size)."""
  length = query.shape[2]
  _check(length, kernel_size, dilation, 'Length')
  start = _window_start(length, kernel_size, dilation, query.device)
  bias_start = _bias_start(length, kernel_size, dilation, query.device)

  attn = query.new_empty(*query.shape[:3], kernel_size)
  for ki in range(kernel_size):
    neighbor = key.index_select(2, start + ki * dilation)
    attn[..., ki] = (query * neighbor).sum(-1) + rpb.index_select(1, bias_start + ki).unsqueeze(0)
  return attn


def natten1dav(attn, value, kernel_size, dilation):
  """attn: (B, H, L, kernel_size); value: (B, H, L, D) -> (B, H, L, D)."""
  length = value.shape[2]
  _check(length, kernel_size, dilation, 'Length')
  start = _window_start(length, kernel_size, dilation, value.device)

  out = torch.zeros_like(value)
  for ki in range(kernel_size):
    out += attn[..., ki:ki + 1] * value.index_select(2, start + ki * dilation)
  return out


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
  """query, key: (B, H, X, Y, D); rpb: (H, 2k-1, 2k-1) -> (B, H, X, Y, kernel_size ** 2)."""
  height, width = query.shape[2], query.shape[3]
  _check(height, kernel_size, dilation, 'Height')
  _check(width, kernel_size, dilation, 'Width')
  start_x = _window_start(height, kernel_size, dilation, query.device)
  start_y = _window_start(width, kernel_size, dilation, query.device)
  bias_x = _bias_start(height, kernel_size, dilation, query.device)
  bias_y = _bias_start(width, kernel_size, dilation, query.device)

  attn = query.new_empty(*query.shape[:4], kernel_size * kernel_size)
  for ki in range(kernel_size):
    key_x = key.index_select(2, start_x + ki * dilation)
    rpb_x = rpb.index_select(1, bias_x + ki)
    for kj in range(kernel_size):
      neighbor = key_x.index_select(3, start_y + kj * dilation)
      bias = rpb_x.index_select(2, bias_y + kj)
      attn[..., ki * kernel_size + kj] = (query * neighbor).sum(-1) + bias.unsqueeze(0)
  return attn


def natten2dav(attn, value, kernel_size, dilation):
  """attn: (B, H, X, Y, kernel_size ** 2); value: (B, H, X, Y, D) -> (B, H, X, Y, D)."""
  height, width = value.shape[2], value.shape[3]
  _check(height, kernel_size, dilation, 'Height')
  _check(width, kernel_size, dilation, 'Width')
  start_x = _window_start(height, kernel_size, dilation, value.device)
  start_y = _window_start(width, kernel_size, dilation, value.device)

  out = torch.zeros_like(value)
  for ki in range(kernel_size):
    value_x = value.index_select(2, start_x + ki * dilation)
    for kj in range(kernel_size):
      weights = attn[..., ki * kernel_size + kj].unsqueeze(-1)
      out += weights * value_x.index_select(3, start_y + kj * dilation)
  return out
