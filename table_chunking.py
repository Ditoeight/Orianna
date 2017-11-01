import numpy as np
from numba import vectorize, float32

@vectorize([float32(float32, float32)], target='cpu', nopython=True)
def vector_mult(x, y):
    return x * y

x = np.zeros((500000000, 2), dtype=np.float16)
x[:] = np.random.randn(*x.shape)
x = x.transpose()
# print(sum(vector_mult.reduce(x, axis=0))) # cpu: 57.093, parallel:46.646
print(np.sum(np.prod(x, axis=0))) # 25.897



#
print(x.nbytes/1000000000)
#
# def max_memory(instances, memory):
#     return ((memory*1000000000)//instances)//4
#
# print(max_memory(5, 4))
# 6 seconds for 10k rows(for 75k instances), meaning 60k rows per minute, 3.6 million rows per hour,
