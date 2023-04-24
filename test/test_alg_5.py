import numpy as np
import sys
import os
import time
f_path = os.path.abspath("..")
sys.path.append(f_path)
import algorithm as alg
import alg_test as algt

def random_test(m_max: int, n_max: int, seed = None) -> np.ndarray:
    assert m_max > 0 and n_max > 0
    if type(seed) == int:
        np.random.seed(seed)
    # 首先，产生一个随机矩阵
    p = np.random.rand(m_max, n_max)
    l = np.random.randint(10, 100, size=m_max)
    s = 2
    time_use = np.zeros([m_max, n_max])
    for i in range(1, m_max + 1):
        for j in range(1, n_max + 1):
            p_cur = p[:i, :j]
            l_cur = l[:i]
            start_time = time.time()
            _, _, _ = alg.stochasticDP(p_cur, l_cur, i, j, s)
            end_time = time.time()
            time_use[i-1, j-1] = end_time-start_time
    return time_use

if __name__ == "__main__":
    p_test = [[0.5, 0.8], [0.4, 0.6]]
    l_test = [80, 50]
    m = 2
    n = 2
    s = 2
    res_PI, res_V, res_T = alg.stochasticDP(p_test, l_test, m, n, s)
    print(res_PI)
    print(res_V)
    print(res_T)


    res_time = random_test(5, 4, 3)
    print(res_time)
    res_time = random_test(4, 5, 3)
    print(res_time)
    res_time = random_test(5, 5, 3)
    print(res_time)