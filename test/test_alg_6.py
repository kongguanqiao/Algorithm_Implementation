import numpy as np
import sys
import os
f_path = os.path.abspath("..")
sys.path.append(f_path)
import algorithm as alg
import alg_test as algt

def display_result(name: str, u: int, v: float, m: int, n: int):
    print(f"============={name} 算法结果：===============")
    print("分配方案   ", u)
    print("方案效能  ", v)
    algt.display_assignment(u, -1, m, n)

def _build(m: int, n: int, seed: int, greed=False):
    np.random.seed(seed)
    if greed:
        p = np.random.rand(m, 1)
        p = np.repeat(p, n, 1)
    else:
        p = np.random.rand(m, n)
    l = np.random.randint(10, 100, size=m)
    return p, l

def _correct_test(m: int, n: int, seed: int, debug=False):
    p, l = _build(m, n, seed=seed, greed=True)
    MMR_u, MMR_v = alg.MMR(p, l, m, n)
    BS_u, BS_v = alg.brutalSearch(p, l, m, n)
    if debug:
        print("p 构造结果为 ", p)
        print("l 构造结果为 ", l)
        display_result("MMR", MMR_u, MMR_v, m, n)
        display_result("brutalSearch", BS_u, BS_v, m, n)
    if MMR_v - BS_v > 1e-4:
        import pdb; pdb.set_trace()
    else:
        print(f"seed = {seed} 时，相等 → 正确")

def correct_test(m: int, n: int, max_seed: int, debug=False):
    for i in range(max_seed):
        _correct_test(m, n, seed=i, debug=debug)


def _diff_test(m: int, n: int, seed: int, debug=False):
    p, l = _build(m, n, seed=seed, greed=False)
    MMR_u, MMR_v = alg.MMR(p, l, m, n)
    BS_u, BS_v = alg.brutalSearch(p, l, m, n)
    if debug:
        print("p 构造结果为 ", p)
        print("l 构造结果为 ", l)
        display_result("MMR", MMR_u, MMR_v, m, n)
        display_result("brutalSearch", BS_u, BS_v, m, n)
    if MMR_v < BS_v + 1e-4:
        import pdb; pdb.set_trace()
    print(f"seed = {seed} 时， MMR_v = {MMR_v}, BS_v = {BS_v}, diff = {MMR_v - BS_v}")
    return MMR_v, BS_v

def diff_test(m: int, n: int, max_seed: int, debug=False):
    for i in range(max_seed):
        MMR_v, BS_v = _diff_test(m, n, i, debug=debug)

if __name__ == "__main__":
    p_test = [[0.5, 0.8], [0.4, 0.6]]
    l_test = [80, 50]
    m, n = 2, 2

    res_u, res_v = alg.MMR(p_test, l_test, m, n)
    algt.display_assignment(res_u, instate=-1, m=m, n=n)

    print("u: ", res_u)
    print("v: ", res_v)


    # 步骤一，根据贪心算法的性质，测试其正确性
    print("======检查正确性======")
    correct_test(3, 3, 10, debug=False)
    # 步骤二，贪心和静态算法
    print("======检查差别=====")
    diff_test(3, 3, 10, debug=False)



