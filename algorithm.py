import os
import numpy as np
from util import hamming, S_insert, A_m, B_z
import time

def targetSurvivalProbs(p: np.ndarray, m: int, n: int) -> np.ndarray:
    assert len(p.shape) == 2 and p.shape[0] == m and p.shape[1] == n
    u_num = (m + 1) ** n
    q = np.ones([u_num, m])      ## q 矩阵：[(m+1)^n，m] 
    for u in range(1, u_num):
        u_ = u
        for j in range(n):
            i_ = u_ % (m + 1)
            i = i_ - 1
            if i >= 0:
                q[u, i] = q[u, i] * (1 - p[i, j])
            u_ = u_ // (m + 1)
    return q

def targetTransitProbs(p: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    m: 目标个数
    n: 武器个数
    p: 概率矩阵 p_ij 表示目标i被武器j的击毁概率
    """
    assert len(p.shape) == 2 and p.shape[0] == m and p.shape[1] == n
    U, X, Y = (m + 1) ** n, 1 << (m + n), 1 << n
    ret = np.zeros([X, U, X])
    Q = targetSurvivalProbs(p, m, n)
    for x in range(1, X):
        ## y: 武器状态，2进制，n位； 
        ## z：目标状态，2进制，m位
        y, z = x % Y, x // Y
        for d1 in range(1, (m + 1)**hamming(y, n)):     # 去掉分配方案0
            u1 = S_insert(m+1, d1, y, n)
            for d2 in range(1, 2**hamming(z, m)):       # 
                z_ = S_insert(2, d2, z, m)
                x_ = z_ * Y + (y - A_m(u1, m+1, n))
                ret[x, u1, x_] = 1
                for i in range(0, m):
                    a = B_z(z, i+1)
                    b = B_z(z^z_, i+1)
                    ret[x, u1, x_] *= a * ((1-b)*Q[u1, i]+b*(1-Q[u1, i])) + (1-a)
    return ret

def stochasticDP(p: np.ndarray, l: np.ndarray, m: int, n: int, s: int) -> np.ndarray:
    if type(p) == list:
        p = np.array(p)
    if type(l) == list:
        l = np.array(l)
    assert len(p.shape) == 2 and p.shape[0] == m and p.shape[1] == n
    U, X, Y = (m + 1) ** n, 1 << (m + n), 1 << n
    V = np.zeros([X])
    for x in range(X):
        tmp_z = x // Y
        V[x] = sum(B_z(tmp_z, i+1)*l[i] for i in range(m))
    Q = np.zeros([X, U])
    PI = np.zeros([s, X])
    T = targetTransitProbs(p, m, n)
    for k in range(s):
        tmp_V = np.zeros([X])
        Q = np.zeros([X, U])
        for x in range(1, X):
            y, z = x % Y, x // Y
            v_min, pi_min = -1, -1
            for d1 in range(1, (m+1)**hamming(y, n)):
                vu = S_insert(m+1, d1, y, n)
                for d2 in range(1, 1 << hamming(z, m)):
                    x_ = S_insert(2, d2, z, m) * Y + (y - A_m(vu, m+1, n))
                    Q[x, vu] += T[x, vu, x_] * V[x_]
                if v_min == -1 or v_min > Q[x, vu]:
                    v_min, pi_min = Q[x, vu], vu 
            if v_min != -1:
                tmp_V[x], PI[k, x] = v_min, pi_min
            else:
                tmp_V[x], PI[k, x] = V[x], 0
        V = tmp_V
    return PI, V, T

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
            _, _, _ = stochasticDP(p_cur, l_cur, i, j, s)
            end_time = time.time()
            time_use[i-1, j-1] = end_time-start_time
    return time_use

if __name__ == "__main__":
    # p_test = [[0.5, 0.8], [0.4, 0.6]]
    # l_test = [80, 50]
    # m = 2
    # n = 2
    # s = 2
    # res_PI, res_V, res_T = stochasticDP(p_test, l_test, m, n, s)
    # print(res_PI)
    # print(res_V)
    # # print(res_T)

    res_time = random_test(5, 4, 3)
    print(res_time)

    res_time = random_test(4, 5, 3)
    print(res_time)

    res_time = random_test(5, 5, 3)
    print(res_time)