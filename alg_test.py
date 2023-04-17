import numpy as np
import algorithm as alg

def build(i: int, n: int):
    """
    """
    r = bin(i)[2:]
    r = "0"*(n-len(r)) + r
    r = list(r)
    r.reverse()
    return " ".join(r)
    
def describe(result, i, j, k, m, n):
    """
    仅针对 targetTransitProbs 的输出结果，解释某个数据
    """
    Y, U = 1 << n, (m + 1) ** n
    print(result[i, j, k])
    ## x: 武器状态
    ## y: 目标状态
    x, y = i % Y, i // Y
    x_, y_ = k % Y, k // Y
    fen = []
    for l in range(n):
        c_m = j % (m + 1)
        j //= (m + 1)
        fen.append(str(int(c_m)))
    print("初始武器状态为：\t", build(x, n))
    print("初始目标状态为：\t", build(y, m))
    print("目标分配方案为：\t", " ".join(fen))
    print("结果武器状态为：\t", build(x_, n))
    print("结果目标状态为：\t", build(y_, m))

if __name__ == "__main__":
    p_test = np.array([[0.5, 0.8], [0.4, 0.6]])
    l_test = [180, 50]
    m = 2
    n = 2
    q = alg.targetSurvivalProbs(p_test, m, n)
    result = alg.targetTransitProbs(p_test, m, n)
    result
    # describe(result, 15, 6, 13, 2, 2)
    # describe(result, 15, 6, 5, 2, 2)
    describe(result, 5, 2, 4, m, n)