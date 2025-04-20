import numpy as np
from sklearn.linear_model import LinearRegression
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

def deduplicate_breakpoints_by_projection(break_points, u, v, eps=1e-2):
    """
    对 breakpoints 沿路径方向进行投影，然后根据投影值合并相近的点
    """
    direction = v - u
    direction = direction / np.linalg.norm(direction)  # 单位向量

    projections = np.array([np.dot(p - u, direction) for p in break_points])
    sorted_idx = np.argsort(projections)
    break_points = break_points[sorted_idx]
    projections = projections[sorted_idx]

    # 合并相近投影点
    merged = []
    current_group = [break_points[0]]
    for i in range(1, len(break_points)):
        if abs(projections[i] - projections[i - 1]) < eps:
            current_group.append(break_points[i])
        else:
            merged.append(np.mean(current_group, axis=0))
            current_group = [break_points[i]]
    if current_group:
        merged.append(np.mean(current_group, axis=0))

    return np.array(merged)


def dnn_along_path_blackbox(u, v, model, num_points=100000, std_factor=1.5, dedup_eps=1e-3, plot=True):
    
    # 生成路径上的点
    alphas = np.linspace(0, 1, num_points)
    points = np.array([u * (1 - a) + v * a for a in alphas])
    outputs = np.array([model(x[0], x[1]) for x in points])

    # 计算梯度和导数差异
    grads = np.gradient(outputs, alphas)
    grad_diff = np.abs(np.diff(grads))

    # 自适应阈值
    threshold = np.std(grad_diff) * std_factor
    breaks_alpha = alphas[1:][grad_diff > threshold]
    break_points = np.array([u * (1 - a) + v * a for a in breaks_alpha])

    # 去重：去掉彼此靠得太近的点
    filtered = []
    for pt in break_points:
        if not any(np.linalg.norm(pt - f) < dedup_eps for f in filtered):
            filtered.append(pt)
    break_points = np.array(filtered)

    # 输出
    break_points = deduplicate_breakpoints_by_projection(break_points, u, v, eps=1e-2)

    # refine 每一个 break point（断点），提升精度
    refined_break_points = []
    for pt in break_points:
        # 沿路径前后稍微移动一点，构造一个小区间
        direction = v - u
        direction /= np.linalg.norm(direction)
        delta = 1e-3 * direction
        x_left = pt - delta
        x_right = pt + delta

        refined_pt = refine_breakpoint(model, x_left, x_right)
        refined_break_points.append(refined_pt)

    refined_break_points = np.array(refined_break_points)

    print("\n精修后的断点坐标（高精度）：")
    for pt in refined_break_points:
        print(f"({pt[0]:.10f}, {pt[1]:.10f})")
    return refined_break_points


def refine_breakpoint(model, x_left, x_right, tol=1e-10, max_iter=100):
    """
    在 [x_left, x_right] 区间中对断点精细定位，使误差 < tol
    """
    for _ in range(max_iter):
        x_mid = (x_left + x_right) / 2
        y_left = model(*x_left)
        y_mid = model(*x_mid)
        
        if np.abs(np.linalg.norm(x_right - x_left)) < tol:
            return x_mid

        # 用一阶差值检查是否过了分段
        grad_left = (y_mid - y_left) / np.linalg.norm(x_mid - x_left)

        if abs(grad_left) > 1e-6:  # 激活了
            x_right = x_mid
        else:
            x_left = x_mid

    return x_mid

# 你已有的模型定义、ReLU、dnn_2_3_3_1、refine_breakpoint 等函数保持不变...

def run_multiple_paths(model, num_paths=7, input_range=2.0):
    all_break_points = []
    np.random.seed(42)  # 设置随机种子为 42
    for i in range(num_paths):
        np.random.seed(42+i)  # 设置随机种子
        u = np.random.uniform(-input_range, input_range, size=2)
        v = np.random.uniform(-input_range, input_range, size=2)

        print(f"\n路径 {i + 1}: u = {u}, v = {v}")

        break_pts = dnn_along_path_blackbox(u, v, model, plot=False)
        all_break_points.append(break_pts)

    all_break_points = np.vstack(all_break_points)
    print("\n==============================")
    print("所有路径上的精修断点汇总：")
    for pt in all_break_points:
        print(f"({pt[0]:.10f}, {pt[1]:.10f})")
    print(f"\n共计断点数量：{len(all_break_points)}")
    return all_break_points


# 调用执行
witnesses = run_multiple_paths(dnn_2_3_3_1)

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




