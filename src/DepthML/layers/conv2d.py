from typing import Any, Tuple

from .base_layer import BaseLayer
from ..typing import tensor_types, InitializerLike
from ..initializers import GlorotUniform, Zeros
from ..utils import get_xp
from DepthTensor import Tensor, Function

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None

_kernel_cache = {}


def get_im2col_kernel(dtype):
    if dtype not in _kernel_cache:
        ctype = "double" if dtype == np.float64 else "float"

        code = r"""
        extern "C" __global__
        void im2col(const {0}* x, {0}* cols, 
                    int N, int C, int H, int W, 
                    int out_h, int out_w, 
                    int kh, int kw, int stride, int padding) {{
                    
            int n_rows = C * kh * kw;
            int n_cols = N * out_h * out_w;
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (idx < n_rows * n_cols) {{
                int row = idx / n_cols; 
                int col = idx % n_cols; 
                
                int kx = row % kw;
                int ky = (row / kw) % kh;
                int c  = row / (kw * kh);
                
                int ox = col % out_w;
                int oy = (col / out_w) % out_h;
                int n  = col / (out_w * out_h);
                
                int h_in = oy * stride - padding + ky;
                int w_in = ox * stride - padding + kx;
                
                {0} val = 0.0;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {{
                    val = x[n * (C*H*W) + c * (H*W) + h_in * W + w_in];
                }}
                cols[idx] = val;
            }}
        }}
        """.format(
            ctype
        )

        _kernel_cache[dtype] = cp.RawKernel(code, "im2col")

    return _kernel_cache[dtype]


_col2im_cache = {}


def get_col2im_kernel(dtype):
    if dtype not in _col2im_cache:
        ctype = "double" if dtype == np.float64 else "float"

        code = r"""
        extern "C" __global__
        void col2im(const {0}* cols, {0}* dx, 
                    int N, int C, int H, int W, 
                    int out_h, int out_w, 
                    int kh, int kw, int stride, int padding) {{
                    
            int n_rows = C * kh * kw;
            int n_cols = N * out_h * out_w;
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (idx < n_rows * n_cols) {{
                int row = idx / n_cols; 
                int col = idx % n_cols; 
                
                int kx = row % kw;
                int ky = (row / kw) % kh;
                int c  = row / (kw * kh);
                
                int ox = col % out_w;
                int oy = (col / out_w) % out_h;
                int n  = col / (out_w * out_h);
                
                int h_in = oy * stride - padding + ky;
                int w_in = ox * stride - padding + kx;
                
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {{
                    int dx_idx = n * (C*H*W) + c * (H*W) + h_in * W + w_in;
                    atomicAdd(&dx[dx_idx], cols[idx]);
                }}
            }}
        }}
        """.format(
            ctype
        )

        _col2im_cache[dtype] = cp.RawKernel(code, "col2im")

    return _col2im_cache[dtype]


def im2col(
    x: Any, field_height: int, field_width: int, padding: int = 1, stride: int = 1
):
    xp = get_xp(x)

    if xp == cp:
        if not x.flags.c_contiguous:
            x = cp.ascontiguousarray(x)

        N, C, H, W = x.shape
        out_h = (H + 2 * padding - field_height) // stride + 1
        out_w = (W + 2 * padding - field_width) // stride + 1

        cols = cp.empty(
            (C * field_height * field_width, N * out_h * out_w), dtype=x.dtype
        )

        threads = 512
        blocks = (cols.size + threads - 1) // threads

        kernel = get_im2col_kernel(x.dtype)

        kernel(
            (blocks,),
            (threads,),
            (
                x,
                cols,
                N,
                C,
                H,
                W,
                out_h,
                out_w,
                field_height,
                field_width,
                stride,
                padding,
            ),
        )
        return cols

    p = padding
    x_padded = xp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    N, C, H, W = x_padded.shape
    out_h = (H - field_height) // stride + 1
    out_w = (W - field_width) // stride + 1

    cols = xp.empty((N, C, field_height, field_width, out_h, out_w), dtype=x.dtype)

    for r in range(field_height):
        for c in range(field_width):
            cols[:, :, r, c, :, :] = x_padded[
                :, :, r : r + stride * out_h : stride, c : c + stride * out_w : stride
            ]

    # Transpose: (C, K_h, K_w, N, out_h, out_w)
    cols = cols.transpose(1, 2, 3, 0, 4, 5)
    cols = cols.reshape(C * field_height * field_width, -1)

    return cols


def col2im(
    cols: Any,
    x_shape: Tuple[int, ...],
    field_height: int,
    field_width: int,
    padding: int = 1,
    stride: int = 1,
):
    xp = get_xp(cols)

    if xp == cp:
        N, C, H, W = x_shape
        out_h = (H + 2 * padding - field_height) // stride + 1
        out_w = (W + 2 * padding - field_width) // stride + 1

        if not cols.flags.c_contiguous:
            cols = cp.ascontiguousarray(cols)

        dx = cp.zeros(x_shape, dtype=cols.dtype)

        threads = 512
        blocks = (cols.size + threads - 1) // threads

        # 2. Get Correct Kernel for Dtype
        kernel = get_col2im_kernel(cols.dtype)

        kernel(
            (blocks,),
            (threads,),
            (
                cols,
                dx,
                N,
                C,
                H,
                W,
                out_h,
                out_w,
                field_height,
                field_width,
                stride,
                padding,
            ),
        )
        return dx

    N, C, H, W = x_shape
    H_pad, W_pad = H + 2 * padding, W + 2 * padding

    x_padded = xp.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    out_h = (H_pad - field_height) // stride + 1
    out_w = (W_pad - field_width) // stride + 1

    cols_reshaped = cols.reshape(C, field_height, field_width, N, out_h, out_w)

    for r in range(field_height):
        for c in range(field_width):
            patch_grad = cols_reshaped[:, r, c, :, :]
            patch_grad = patch_grad.transpose(1, 0, 2, 3)

            x_padded[
                :, :, r : r + stride * out_h : stride, c : c + stride * out_w : stride
            ] += patch_grad

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Conv2DFunc(Function):
    def link(
        self,
        y: Tensor,
        x: Tensor,
        w: Tensor,
        b: Tensor | None,
        stride: int,
        pad: int,
        cache: dict,
    ) -> None:

        def backward() -> None:
            if y.grad is None:
                y.zero_grad()

            for p in [x, w, b]:
                if isinstance(p, Tensor) and p.requires_grad and p.grad is None:
                    p.zero_grad()

            x_col = cache.pop("x_col")
            x_shape = cache["x_shape"]

            dout = y.grad
            xp = get_xp(dout)
            N, F, H_out, W_out = dout.shape

            # Transpose to (F, N, H, W) then flatten
            dout_flat = dout.transpose(1, 0, 2, 3).reshape(F, -1)

            if b is not None and b.requires_grad:
                b.grad += xp.sum(dout, axis=(0, 2, 3))

            if w.requires_grad:
                dw_flat = dout_flat @ x_col.T
                w.grad += dw_flat.reshape(w.shape)

            if x.requires_grad:
                F, C, kH, kW = w.shape
                w_flat = w.data.reshape(F, -1)

                # (C*K, N*H*W)
                dx_col = w_flat.T @ dout_flat

                dx = col2im(
                    cols=dx_col,
                    x_shape=x_shape,
                    field_height=kH,
                    field_width=kW,
                    padding=pad,
                    stride=stride,
                )

                x.grad += dx

            del x_col

        prev = [p for p in [x, w, b] if isinstance(p, Tensor)]
        y.prev = tuple(prev)
        y.backward = backward

    def __call__(
        self, x: Tensor, w: Tensor, b: Tensor | None, stride: int = 1, padding: int = 0
    ) -> Tensor:
        N, C, H, W = x.shape
        F, _, kH, kW = w.shape

        # =(C*Kh*Kw, N*H_out*W_out)
        x_col = im2col(x.data, kH, kW, padding, stride)

        w_col = w.data.reshape(F, -1)
        # (F, N*H_out*W_out)
        out_col = w_col @ x_col

        if b is not None:
            out_col += b.data.reshape(-1, 1)

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1

        # (F, N, H, W)
        out_reshaped = out_col.reshape(F, N, H_out, W_out).transpose(1, 0, 2, 3)

        require_grad = (
            x.requires_grad or w.requires_grad or (b.requires_grad if b else False)
        )

        y = Tensor(out_reshaped, requires_grad=require_grad, device=x.device)

        if require_grad:
            cache = {"x_col": x_col, "x_shape": x.shape, "h_out": H_out, "w_out": W_out}
            self.link(y, x, w, b, stride, padding, cache)

        return y


func = Conv2DFunc()


class Conv2d(BaseLayer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        weight_initializer: InitializerLike = GlorotUniform(),
        bias_initializer: InitializerLike = Zeros(),
        name: str = "conv2d",
        trainable: bool = True,
    ) -> None:
        super().__init__(name=name, trainable=trainable)

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.w: Tensor | None = None
        self.b: Tensor | None = None

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def build(
        self,
        input_shape: Tuple[int, ...],
        device: tensor_types.Device,
        **kwargs: Any,
    ) -> None:
        self.init_parameters(input_shape=input_shape, device=device)
        self.built = True

    def init_parameters(
        self,
        input_shape: Tuple[int, ...],
        device: tensor_types.Device = "cpu",
        **kwargs: Any,
    ) -> None:
        in_channels = input_shape[0]

        w_shape = (self.filters, in_channels, self.kernel_size, self.kernel_size)

        self.w = self.weight_initializer(
            shape=w_shape, device=device, requires_grad=True
        )
        self.b = self.bias_initializer(
            shape=(self.filters,), device=device, requires_grad=True
        )

    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        if not self.built:
            self.build(X.shape[1:], X.device)
        return func(X, self.w, self.b, stride=self.stride, padding=self.padding)

    def compute_output_shape(
        self, input_shape: Tuple[int, ...], **kwargs
    ) -> Tuple[int, ...]:
        c, h, w = input_shape

        h_out = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_out = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        return (self.filters, h_out, w_out)
