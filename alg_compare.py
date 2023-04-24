import numpy as np
import time
from algorithm import targetSurvivalProbs, brutalSearch, targetTransitProbs, stochasticDP

# exam_p = np.array([[0.5, 0.8], [0.4, 0.6]])
# exam_l = np.array([180, 50])
# m, n = exam_p.shape
# s = 1
# PI_1, V_1, T_1 = stochasticDP(exam_p, exam_l, m, n, s=s)
# print(f"当 s={s} 时, 最优方案效能为", V_1[-1])

# s = 2
# PI_2, V_2, T_2 = stochasticDP(exam_p, exam_l, m, n, s=s)
# print(f"当 s={s} 时, 最优方案效能为", V_2[-1])

def cal_v_diff(m: int, n: int, s: int, seed = None) -> np.ndarray:
    assert m > 0 and n > 0 and s > 0
    if type(seed) == int:
        np.random.seed(seed)
    # 1. 首先，产生一个随机矩阵
    p = np.random.rand(m, n)
    l = np.random.randint(10, 100, size=m)
    _, dump_v, _ = stochasticDP(p, l, m, n, s, dump=True)
    print(dump_v)
    for i in range(s):
        print(f"{i+1}次拦截时，最优方案期望为{dump_v[i+1][-1]}")

# cal_v_diff(m=4, n=4, s=3, seed=1)

def _cal_v_compare(m: int, n: int, s: int = 2, seed=None) -> np.ndarray:
    assert m > 0 and n > 0 and s > 0
    if type(seed) == int:
        np.random.seed(seed)
    # 1. 首先，产生一个随机矩阵
    p = np.random.rand(m, n)
    l = np.random.randint(10, 100, size=m)
    _, dump_v, _ = stochasticDP(p, l, m, n, s, dump=True)
    return dump_v[1, -1], dump_v[2, -1]

def cal_v_compare(m: int, n: int, s: int=2, max_seed=10) -> np.ndarray:
    assert m > 0 and n > 0 and s > 0 and type(max_seed)==int
    ret = np.zeros([max_seed, 2])
    print(f"m={m}, n={n}, s={s}, max_seed={max_seed}")
    for i in range(max_seed):
        v1, v2 = _cal_v_compare(m, n, s, seed=i)
        if v1 <= v2:
            print("出现情况, v2 >= v1")
        ret[i, 0] = v1
        ret[i, 1] = v2
        print(f"seed={i}, v1={v1}, v2={v2}")
    diff = ret[:, 0] - ret[:, 1]
    min_v, argmin_v = np.min(diff), np.argmin(diff)
    max_v, argmax_v = np.max(diff), np.argmax(diff)
    print(f"seed={argmin_v}, min_v={min_v}")
    print(f"seed={argmax_v}, max_v={max_v}")
    return ret, min_v, argmin_v, max_v, argmax_v

if __name__ == "__main__":
    ret, minV, argminV, maxV, argmaxV = cal_v_compare(m=4, n=4, max_seed=100)