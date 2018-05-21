# The MIT License (MIT)
# 
# Copyright (c) 2015 Andrej Karpathy
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# build with:
# python setup_cython.py build_ext --inplace

import numpy as np

def im2col_cython(x, field_height, field_width, padding, stride):
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]

    HH = (H + 2 * padding - field_height) / stride + 1
    WW = (W + 2 * padding - field_width) / stride + 1

    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    cols = np.zeros((C * field_height * field_width, int(N * HH * WW)), dtype=x.dtype)

    # Moving the inner loop to a C function with no bounds checking works, but does
    # not seem to help performance in any measurable way.

    im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride)
    return cols



def im2col_cython_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride):
    HH=int(HH)
    WW=int(WW)
    N=int(N)
    for c in range(C):
        for yy in range(HH):
            for xx in range(WW):
                for ii in range(field_height):
                    for jj in range(field_width):
                        row = c * field_width * field_height + ii * field_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]

def col2im_cython(cols, N, C, H, W, field_height, field_width, padding, stride):
    x = np.empty((N, C, H, W), dtype=cols.dtype)
    HH = (H + 2 * padding - field_height) / stride + 1
    WW = (W + 2 * padding - field_width) / stride + 1
    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_cython_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


def col2im_cython_inner(cols, x_padded, N, C, H, W, HH, WW, field_height, field_width, padding, stride):
    HH=int(HH)
    WW=int(WW)
    N=int(N)
    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            x_padded[i, c, stride * yy + ii, stride * xx + jj] += cols[row, col]


