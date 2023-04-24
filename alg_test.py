import numpy as np

def build(i: int, n: int, ret="str"):
    """
    """
    r = bin(i)[2:]
    r = "0"*(n-len(r)) + r
    r = list(r)
    r.reverse()
    return " ".join(r) if ret == "str" else r

def describe_assignment(assign: int, m: int, n: int):
    fen = []
    for _ in range(n):
        c_m = assign % (m + 1)
        assign //= (m + 1)
        fen.append(str(int(c_m)))
    return fen

def display_assignment(assign: int, instate: int, m: int, n: int):
    instate = (1 << m + n) - 1 if instate < 0 else instate
    Y = 1 << n
    in_weapon, in_target = instate % Y, instate // Y
    in_weapon, in_target = build(in_weapon, n, ret="list"), build(in_target, m, ret="list")
    print("输入武器状态:\t", "\t".join(in_weapon))
    print("输入目标状态:\t", "\t".join(in_target))

    fen = describe_assignment(assign, m, n)
    print("")
    print("武器\目标\t", "\t".join(in_target))
    print("---------------"+"-------"*len(in_target))
    for i in range(n):
        cur = ["-"] * m
        if fen[i] != "0":
            cur[int(fen[i])-1] = "1" 
        print(f"w{i+1} : {in_weapon[i]}  |\t", "\t".join(cur))
        
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
    import algorithm as alg
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

    display_assignment(2, 5, m, n)

    P = np.array([
        [.8, .9, .6, .7],
        [.3, .5, .4, .6],
        [.5, .6, .3, .7],
        [.5, .7, .6, .4],
        [.8, .6, .6, .8]
    ])
    V = np.array([.3, .1, .2, .5, .4])

    PI, dump_v, _ = alg.stochasticDP(P, V, P.shape[0], P.shape[1], s=2, dump=True)
    print(PI[:, -1])
    print(dump_v[:, -1])
