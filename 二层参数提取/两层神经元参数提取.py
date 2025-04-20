import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict

# 1. 定义黑盒模型 f
def relu(z):
    return max(0, z)

def dnn_2_3_3_1(x, y):
    # 第一隐藏层 z = w·x + b
    z1 = w1[0]*x + w1[1]*y + b1
    z2 = w2[0]*x + w2[1]*y + b2
    z3 = w3[0]*x + w3[1]*y + b3
    
    a1 = relu(z1)
    a2 = relu(z2)
    a3 = relu(z3)

    # 第二隐藏层：输入是第一层的输出 a1, a2, a3
    z4 = w4[0]*a1 + w4[1]*a2 + w4[2]*a3 + b4
    z5 = w5[0]*a1 + w5[1]*a2 + w5[2]*a3 + b5
    z6 = w6[0]*a1 + w6[1]*a2 + w6[2]*a3 + b6

    a4 = relu(z4)
    a5 = relu(z5)
    a6 = relu(z6)

    # 输出层：输入是第二层的输出 a4, a5, a6
    output = wo[0]*a4 + wo[1]*a5 + wo[2]*a6 + bo
    return output

# 第一层参数（输入 -> 第一隐藏层）
w1 = [0.2, -0.3]; b1 = 0.1
w2 = [0.4, 0.5];  b2 = -0.2
w3 = [-0.6, 0.1]; b3 = 0.3

# 第二层参数（第一隐藏层 -> 第二隐藏层）
w4 = [0.3, -0.5, 0.2]; b4 = -0.1
w5 = [-0.4, 0.1, 0.6]; b5 = 0.2
w6 = [0.7, 0.3, -0.2]; b6 = -0.3

# 输出层参数（第二隐藏层 -> 输出）
wo = [0.7, -0.5, 0.2]; bo = 0.1

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


    # 随机生成路径

    for i in range(num_paths):
        np.random.seed(i*2)  # 随机种子
        u = np.random.uniform(-input_range, input_range, 2)
        v = np.random.uniform(-input_range, input_range, 2)
        print(f"\n路径 {i + 1}:")
        breakpoints = do_better_sweep_blackbox(model, u, v)

        for pt in breakpoints:
            print(f"  断点坐标: ({pt[0]:.8f}, {pt[1]:.8f})")

        all_breakpoints.extend(breakpoints)

    print(f"\n共找到断点总数: {len(all_breakpoints)}")
    return np.array(all_breakpoints)


witnesses = run_sweep_multiple_paths(dnn_2_3_3_1)

# 3. 差分参数
epsilon = 1e-3   # 用于穿过激活边界
delta   = 1e-4   # 用于数值导数

# 4. 真实参数（用于对比）
true_ws = [
    np.array([0.2, -0.3]),
    np.array([0.4,  0.5]),
    np.array([-0.6, 0.1]),
]
true_bs = [0.1, -0.2, 0.3]

# 5. 计算 Δ1, Δ2 的函数
def compute_deltas(x_star):
    x1, x2 = x_star
    deltas = []
    for i in [0, 1]:  # i=0 对应 x1，i=1 对应 x2
        plus_eps  = [x1, x2]
        minus_eps = [x1, x2]
        plus_eps[0]  += epsilon
        minus_eps[0] -= epsilon

        def finite_diff_at(x_pt):
            x_p = x_pt.copy(); x_m = x_pt.copy()
            x_p[i] += delta; x_m[i] -= delta
            return (dnn_2_3_3_1(*x_p) - dnn_2_3_3_1(*x_m)) / (2 * delta)

        alpha_plus  = finite_diff_at(plus_eps)
        alpha_minus = finite_diff_at(minus_eps)
        deltas.append(alpha_plus - alpha_minus)
    return deltas  # [Δ1, Δ2]

def recover_signs(witnesses, Z, dnn_fn, epsilon=0.01):
    """
    使用 23年shamir Neuron Wiggle 方法恢复隐藏层神经元的符号

    参数:
        witnesses: List of (x, y)，每个神经元的 witness（输入使 ReLU 输入接近 0）
        Z: 无符号权重矩阵，形状为 (3, 2)
        dnn_fn: 黑盒神经网络函数 dnn_2_3_3_1(x, y)
        epsilon: wiggle 扰动强度（默认 0.01）

    返回:
        signs: List of [+1, -1, ...]，每个隐藏神经元的符号
    """
    signs = []

    for i, (x, y) in enumerate(witnesses):
        z = Z[i]
        direction = z / np.linalg.norm(z) * epsilon  # wiggle δ

        # 正负方向扰动
        x_plus, y_plus = x + direction[0], y + direction[1]
        x_minus, y_minus = x - direction[0], y - direction[1]

        f0     = dnn_fn(x, y)
        f_plus = dnn_fn(x_plus, y_plus)
        f_minus= dnn_fn(x_minus, y_minus)

        delta_plus  = abs(f_plus - f0)
        delta_minus = abs(f_minus - f0)

        # 如果正方向扰动输出变化大 → sign 为正
        sign = 1 if delta_plus > delta_minus else -1
        signs.append(sign)

    return signs



def recover_output_layer_via_linsolve(inputs, recovered_ws, recovered_bs, dnn_fn):
    A = []
    y = []

    for x1, x2 in inputs:
        a = []
        for w, b in zip(recovered_ws, recovered_bs):
            z = np.dot(w, [x1, x2]) + b
            a.append(relu(z))
        A.append(a + [1.0])  # 最后一列是偏置项 b_o
        y.append(dnn_fn(x1, x2))

    A = np.array(A)  # shape (n, 4)
    y = np.array(y)  # shape (n,)

    # 解线性方程组 A @ [wo..., bo] = y
    sol, residuals, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    return sol[:3], sol[3]


def rebuilt_dnn(x1, x2, ws, bs, wo, bo):
    """
    用恢复出的参数重建神经网络并返回输出

    参数:
        x1, x2: 输入
        ws: List of 3 权重向量，每个是 shape=(2,)
        bs: List of 3 偏置
        wo: 输出层权重向量 (3,)
        bo: 输出层偏置 (float)

    返回:
        输出值
    """
    a = []
    for w, b in zip(ws, bs):
        z = np.dot(w, [x1, x2]) + b
        a.append(relu(z))
    return np.dot(wo, a) + bo

Z=[]
recovered_ws = []  # 权重方向未筛选
recovered_bs = []  # 偏置项

for j, x_star in enumerate(witnesses, 1):
    Δ1, Δ2 = compute_deltas(x_star)

    # 归一化权重（无符号）
    w_dir = np.array([Δ1, Δ2])
    w_dir /= np.linalg.norm(w_dir)
    
    # 收集结果
    recovered_ws.append(w_dir)

print("\n恢复的权重方向（未筛选）：")
for i, w in enumerate(recovered_ws):
    print(f"神经元 {i + 1}: {w}")

def filter_repeated_directions(recovered_ws, min_count=2, decimals=6):
    def normalize_direction(w):
        norm = np.linalg.norm(w)
        if norm == 0:
            return (0.0, 0.0)
        unit = w / norm
        # 将方向归一化为统一半球
        if unit[0] < 0 or (unit[0] == 0 and unit[1] < 0):
            unit = -unit
        return tuple(np.round(unit, decimals=decimals))

    direction_counts = defaultdict(list)
    for idx, w in enumerate(recovered_ws):
        dir_key = normalize_direction(w)
        direction_counts[dir_key].append(idx)

    results = []
    witnesses_1=[]
    print(f"\n出现超过 {min_count - 1} 次的权重方向（忽略符号）：")
    for direction, indices in direction_counts.items():
        if len(indices) >= min_count:
            print(f"方向 {direction}")
            results.append(direction)
            witnesses_1.append(witnesses[indices[0]])
            print(f"  对应的 witness: {witnesses[indices[0]]}")
    return results, witnesses_1

filtered_directions,witnesses = filter_repeated_directions(recovered_ws, min_count=3)
w_dir=np.array(filtered_directions)
# 恢复符号
sign = recover_signs([x_star], w_dir.reshape(1, -1), dnn_2_3_3_1)[0]
w_dir *= sign  # 恢复符号

print("\n恢复的权重方向（筛选后）：")
for i, w in enumerate(w_dir):
    print(f"神经元 {i + 1}: {w}")



