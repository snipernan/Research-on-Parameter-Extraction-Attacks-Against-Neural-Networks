import numpy as np
import matplotlib.pyplot as plt

# ========= 神经网络模型定义 =========
def relu(z):
    return max(0, z)

def dnn_2_3_3_1(x, y):
    w1 = [0.2, -0.3]; b1 = 0.1
    w2 = [0.4, 0.5];  b2 = -0.2
    w3 = [-0.6, 0.1]; b3 = 0.3

    w4 = [0.3, -0.5, 0.2]; b4 = -0.1
    w5 = [-0.4, 0.1, 0.6]; b5 = 0.2
    w6 = [0.7, 0.3, -0.2]; b6 = -0.3

    wo = [0.7, -0.5, 0.2]; bo = 0.1

    z1 = w1[0]*x + w1[1]*y + b1
    z2 = w2[0]*x + w2[1]*y + b2
    z3 = w3[0]*x + w3[1]*y + b3
    a1 = relu(z1)
    a2 = relu(z2)
    a3 = relu(z3)

    z4 = w4[0]*a1 + w4[1]*a2 + w4[2]*a3 + b4
    z5 = w5[0]*a1 + w5[1]*a2 + w5[2]*a3 + b5
    z6 = w6[0]*a1 + w6[1]*a2 + w6[2]*a3 + b6
    a4 = relu(z4)
    a5 = relu(z5)
    a6 = relu(z6)

    output = wo[0]*a4 + wo[1]*a5 + wo[2]*a6 + bo
    return output

# ========= 断点搜索方法（简洁版 do_better_sweep） =========
def do_better_sweep_blackbox(model, u, v, low=0.0, high=1.0, tol=1e-6):
    """ 沿路径 u -> v 搜索激活断点 """
    critical_points = []

    def f(alpha):
        point = (1 - alpha) * u + alpha * v
        return model(*point)

    def search(low, high):
        mid = (low + high) / 2
        f_low, f_high, f_mid = f(low), f(high), f(mid)
        lin_interp = (f_low + f_high) / 2

        if abs(f_mid - lin_interp) < 1e-7 or high - low < tol:
            return
        else:
            # 断点判定成功，记录点
            alpha_crit = mid
            critical_point = (1 - alpha_crit) * u + alpha_crit * v
            critical_points.append(critical_point)
            return  # 不再递归，避免重复记录

    # 扫描细分区间，适应多个断点
    N = 4096
    alphas = np.linspace(low, high, N + 1)
    for i in range(N):
        search(alphas[i], alphas[i+1])

    return np.array(critical_points)

# ========= 路径批量运行 =========
def run_sweep_multiple_paths(model, num_paths=10, input_range=2.0):
    all_breakpoints = []

    for i in range(num_paths):
        u = np.random.uniform(-input_range, input_range, 2)
        v = np.random.uniform(-input_range, input_range, 2)
        print(f"\n路径 {i + 1}:")
        breakpoints = do_better_sweep_blackbox(model, u, v)

        for pt in breakpoints:
            print(f"  断点坐标: ({pt[0]:.8f}, {pt[1]:.8f})")

        all_breakpoints.extend(breakpoints)

    print(f"\n共找到断点总数: {len(all_breakpoints)}")
    return np.array(all_breakpoints)

# ========= 执行 =========
if __name__ == "__main__":
    all_pts = run_sweep_multiple_paths(dnn_2_3_3_1)
