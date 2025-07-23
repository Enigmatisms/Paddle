# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import paddle
from paddle.static import InputSpec


def symbolic_arange(x):
    return paddle.arange(
        paddle.to_tensor([4], dtype=paddle.int64),
        x,
        paddle.to_tensor([2], dtype=paddle.int64),
    )


def generate_shape_arange1(x):
    batch_size = paddle.shape(x)[1]

    stop = batch_size * 2
    # input_1 = paddle.full([1], 0, dtype = "float32")
    # input_2 = paddle.full([1], 10, dtype = "int64")
    return paddle.arange(0, stop, 2, dtype="int64")


def generate_shape_arange2(x, y, z):
    start = paddle.shape(x)[0] * 2
    stop = paddle.shape(y)[1] - 3
    step = -paddle.shape(z)[2]
    return paddle.arange(start, stop, step, dtype="int64") + 1


def generate_shape_arange3(x, y, z):
    start = paddle.shape(x)[0] * 2
    stop = paddle.shape(y)[1] - 3
    step = -paddle.shape(z)[2]
    res = paddle.arange(start, stop, step, dtype="int64")
    return paddle.arange(res.shape[0], 0, -1, dtype="int64") + res


def generate_shape_arange4(x, y, z):
    start = paddle.shape(x)[0] * 2
    stop = paddle.shape(y)[1] - 3
    step = -paddle.shape(z)[2]
    res = paddle.arange(start, stop, step, dtype="int64")
    return paddle.ones([res.shape[0], 1], dtype="int64") + res


if __name__ == "__main__":
    # x = paddle.to_tensor([
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # ], dtype = "int64")

    # st_func1 = paddle.jit.to_static(
    #     generate_shape_arange1,
    #     input_spec=[
    #         paddle.static.InputSpec(shape=[3, -1], dtype="int64"),
    #     ],
    #     full_graph=True
    # )

    # res = st_func1(x)
    # print(res)
    # print(res.dtype)

    x_spec = InputSpec(shape=[-1, 2, 3], dtype="int64")
    y_spec = InputSpec(shape=[2, -1, 3], dtype="int64")
    z_spec = InputSpec(shape=[2, 3, -1], dtype="int64")

    x = paddle.zeros([11, 2, 3], dtype="int64")
    y = paddle.zeros([2, 0, 3], dtype="int64")
    z = paddle.zeros([2, 3, 2], dtype="int64")
    dyn_res = generate_shape_arange4(x, y, z)
    print("HQY Dynamic: ", dyn_res)

    st_func2 = paddle.jit.to_static(
        generate_shape_arange4,
        input_spec=[x_spec, y_spec, z_spec],
        full_graph=True,
    )
    res = st_func2(x, y, z)
    print(res)
    print(res.dtype)

    np.testing.assert_allclose(
        dyn_res.numpy(),
        res.numpy(),
        rtol=1e-05,
        atol=1e-08,
    )
