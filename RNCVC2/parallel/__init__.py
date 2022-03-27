from pycuda import driver
import numpy as np


def setup():
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    f = open('kernel.cu', 'r')
    kernel = ''.join(f.readlines())  # stores kernel in variable

    mod = SourceModule(kernel)
    return mod

mod = setup()

nn_hausdorff = mod.get_function('nn_hausdorff')

vec_size = 3
n_vecs = 3

output_size = n_vecs ** 2

A = np.array([1, 2, 3], dtype=np.float32)
B = np.array([4, 5, 6], dtype=np.float32)
C = np.array([4, 5, 20], dtype=np.float32)

vecs = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [4, 5, 20]
    ], dtype=np.float32
).ravel()

out = np.empty(output_size, dtype=np.float32)

nn_hausdorff(
    driver.Out(out),
    np.int32(vec_size),
    np.int32(n_vecs),
    driver.In(vecs),
    block=(output_size, 1, 1),  # controls the number of streams to be initialized
    grid=(1, 1)
)

print 'out:', out
