import numpy as np
from algorithm import targetSurvivalProbs, brutalSearch, targetTransitProbs, stochasticDP

exam_p = np.array([[0.5, 0.8], [0.4, 0.6]])
exam_l = np.array([180, 50])
m, n = exam_p.shape
s = 1
PI_1, V_1, T_1 = stochasticDP(exam_p, exam_l, m, n, s=s)
print(f"当 s={s} 时, 最优方案效能为", V_1[-1])

s = 2
PI_2, V_2, T_2 = stochasticDP(exam_p, exam_l, m, n, s=s)
print(f"当 s={s} 时, 最优方案效能为", V_2[-1])
    # p_test = [[0.5, 0.8], [0.4, 0.6]]
    # l_test = [80, 50]

        # m = 2
    # n = 2
    # s = 2
    # res_PI, res_V, res_T = stochasticDP(p_test, l_test, m, n, s)
    # print(res_PI)
    # print(res_V)
    # # print(res_T)