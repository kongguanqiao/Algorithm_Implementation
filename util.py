import numpy as np

def hamming(y: int, n: int) -> int:
    """求y的前n位中 1 的个数

    Args:
        y (int): 被求数
        n (int): 截取前n位计算

    Returns:
        int: 1的个数
    """
    mo = (1 << n) - 1
    y_ = y & mo
    ret = 0
    while y_:
        y_ = y_ & (y_ - 1)
        ret += 1
    return ret

def S_insert(sys: int, d: int, y: int, n: int) -> int:
    """根据原状态y，以及遍历序号数d，插入0，得到当前遍历数

    Args:
        sys (int): 采取的进制数
        d (int): 遍历序号数，是一个sys进制的数字
        y (int): 状态数，是一个2进制的数字，共有n位
        n (int): 状态y的位数

    Returns:
        int: 返回补全后的遍历数
    """
    u = 0
    for i in range(n):
        s, y = y & 1, y >> 1 # 获取当前武器状态
        if s == 1:
            m, d = d % (sys), d // sys  # 当前武器可用，则取到该武器的目标 m
            u += (sys ** i) * m
    return u

def A_m(u: int, sys: int, n: int) -> int:
    """统计sys进制的数u，各位非零的个数

    Args:
        u (int): 被求数，sys进制的数，共n位
        sys (int): u采取的进制
        n (int): 被求数u的位数

    Returns:
        int: 返回u非零位的个数
    """
    ret, k = 0, 1
    while u:
        if u % sys != 0:
            ret += k
        k <<= 1
        u //= sys
    return ret

def B_z(z: int, i: int) -> int:
    """判断z的第i位是否为1

    Args:
        z (int): 被判断的数，2进制表示
        i (int): 要判断的位

    Returns:
        int: 该位置是否为1,
    """
    return (z >> (i - 1)) & 1

if __name__ == "__main__":
    ## 测试s函数
    m, n = 2, 3
    sys = m + 1
    d = 7 # 21 (3)
    y = 5 # 101 (2)
    res = S_insert(sys, d, y, n)
    print(res)

