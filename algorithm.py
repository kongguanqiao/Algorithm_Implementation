import os
import numpy as np
from util import hamming, S_insert, A_m, B_z
from alg_test import display_assignment

import time

def targetSurvivalProbs(p: np.ndarray, m: int, n: int) -> np.ndarray:
    """计算当给定武器对目标杀伤概率条件下，指定分配方案下目标的存活概率

    Args:
        p (np.ndarray): m * n 矩阵，p[i, j]表示目标i-1 被武器j-1 击毁的概率
        m (int): 目标的个数
        n (int): 武器的个数

    Returns:
        np.ndarray: (m+1)^n * m 维矩阵，[i, j] 表示在分配方案i下，第j+1个目标存活的概率
    """
    assert len(p.shape) == 2 and p.shape[0] == m and p.shape[1] == n
    u_num = (m + 1) ** n
    q = np.ones([u_num, m])                     # q 矩阵：[(m+1)^n，m] 
    for u in range(1, u_num):
        u_ = u
        for j in range(n):
            i_ = u_ % (m + 1)                   # 解码， 分配方案编码成 (m+1)进制的数，总共有n位，第i位表示第i个武器分配的目标（0表示不分配目标）
            i = i_ - 1
            if i >= 0:
                q[u, i] = q[u, i] * (1 - p[i, j])
            u_ = u_ // (m + 1)                  # 更新 u_，查看下一个武器的分配目标
    return q

def brutalSearch(p: np.ndarray, l: np.ndarray, m: int, n: int) -> tuple:
    """根据TargetSurvivalProbs 得到的存活概率矩阵，通过暴力搜索 得到打击效能最优的方法

    Args:
        p (np.ndarray): m*n 矩阵，p[i, j]表示目标i 被武器j击毁的概率
        l (np.ndarray): m维向量，表示m个目标的价值
        m (int): 目标个数
        n (int): 武器个数

    Returns:
        tuple: 返回最优方案，及在此方案下的效能
    """
    p = np.array(p) if type(p) == list else p
    l = np.array(l) if type(l) == list else l
    assert len(p.shape) == 2 and p.shape[0] == m and p.shape[1] == n
    u_num = (m + 1) ** n
    q = targetSurvivalProbs(p, m, n)
    f = np.array([sum(q[i, j] * l[j] for j in range(m)) for i in range(u_num)])
    return np.argmin(f), min(f)

def targetTransitProbs(p: np.ndarray, m: int, n: int) -> np.ndarray:
    """计算状态间的转移概率

    Args:
        p (np.ndarray): 概率矩阵
        m (int): 目标个数
        n (int): 武器个数

    Returns:
        np.ndarray: T [2^(m+n), (m+1)^n, 2^(m+n)] 的矩阵，T[i, j, k]表示状态i经由方案j到达状态k的概率
    """
    p = np.array(p) if type(p) == list else p
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

# 算法五： 动态算法
def stochasticDP(p: np.ndarray, l: np.ndarray, m: int, n: int, s: int, dump=False) -> np.ndarray:
    """根据概率矩阵以及提供的目标价值信息，计算有s次拦截机会时，对目标的最优拦截方案及其效能

    Args:
        p (np.ndarray): 毁伤概率矩阵
        l (np.ndarray): 目标价值向量
        m (int): 目标个数
        n (int): 武器个数
        s (int): 拦截次数

    Returns:
        np.ndarray: 返回策略函数 PI [2^(m+n), (m+1)^n] PI[i, j] 表示在状态i下，采取的最优拦截方案是j
    """
    p = np.array(p) if type(p) == list else p
    l = np.array(l) if type(l) == list else l
    assert len(p.shape) == 2 and p.shape[0] == m and p.shape[1] == n
    U, X, Y = (m + 1) ** n, 1 << (m + n), 1 << n
    V = np.zeros([X])
    dump_V = np.zeros([s+1, X]) if dump else None
    for x in range(X):
        tmp_z = x // Y
        V[x] = sum(B_z(tmp_z, i+1)*l[i] for i in range(m))  ## V 函数，其中的值随着
    if dump:
        dump_V[0] = V
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
        if dump:
            dump_V[k+1] = tmp_V
    return (PI, V, T) if not dump else (PI, dump_V, T)

# 算法六， 贪心算法
def MMR(p: np.ndarray, l: np.ndarray, m: int, n: int) -> tuple:
    """根据输入的毁伤概率和价值向量，计算分配方案和此方案下的

    Args:
        p (np.ndarray): m*n 
        l (np.ndarray): m m=目标 价值矩阵
        m (int): 目标个数
        n (int): 武器个数

    Returns:
        tuple: 返回分配方案，及此方案下的收益
    """
    p = np.array(p) if type(p) == list else p
    l = np.array(l) if type(l) == list else l
    assert len(p.shape) == 2 and p.shape[0] == m and p.shape[1] == n
    u_array = np.zeros([n], dtype=int)
    p_array = np.sum(p, axis=1) / n
    delta = l * p_array
    for j in range(n):
        try:
            u_array[j] = int(np.argmax(delta))
            delta[u_array[j]] *= (1-p_array[u_array[j]])
        except:
            import pdb; pdb.set_trace()
    u_, bei = 0, 1
    for j in range(n):
        # u_j = u_array[j] + 1 if u_array[j] != 0 else 0
        u_j = u_array[j] + 1
        u_ += u_j * bei
        bei *= (m + 1)
    f_ = np.sum(delta / p_array)
    return u_, f_

if __name__ == "__main__":
    p_test = [[0.5, 0.8], [0.4, 0.6]]
    l_test = [80, 50]
    m = 2
    n = 2

    res_u, res_v = MMR(p_test, l_test, m, n)
    display_assignment(res_u, instate=-1, m=m, n=n)

    print("u: ", res_u)
    print("v: ", res_v)


