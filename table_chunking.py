import numpy as np
from numba import vectorize, float32

@vectorize([float32(float32, float32)], target='parallel', nopython=True)
def vector_mult(x, y):
    return x * y

x = np.ones((200000000, 5), dtype=np.float32)
x = x.transpose()
print(sum(vector_mult.reduce(x, axis=0))) # cpu: 57.093, parallel:46.646
# print(np.sum(np.prod(x, axis=0))) # 25.897



#
# print(x.nbytes/1000000000)
#
# def max_memory(instances, memory):
#     return ((memory*1000000000)//instances)//4
# #
# print(max_memory(5, 4))
# 6 seconds for 10k rows, meaning 60k rows per minute, 3.6 million rows per hour,
