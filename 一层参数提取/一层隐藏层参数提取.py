import numpy as np
from sklearn.linear_model import LinearRegression
# 1. 定义黑盒模型 f
def relu(z):
    return max(0, z)

def simple_dnn(x1, x2):
    z1 = 0.2*x1 + (-0.3)*x2 + 0.1
    z2 = 0.4*x1 +  0.5 *x2 - 0.2
    z3 = -0.6*x1 + 0.1 *x2 + 0.3
    a1 = relu(z1)
    a2 = relu(z2)
    a3 = relu(z3)
    return 0.7*a1 + (-0.5)*a2 + 0.2*a3 + 0.1

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

def plot_dnn_along_path_blackbox(model, num_points=300000, std_factor=1.5, dedup_eps=1e-3, plot=True):

    u = np.array([-1.0, 0.1])
    v = np.array([1.0, 0.1])

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

    return break_points

#plot_dnn_along_path(u, v)
witnesses=plot_dnn_along_path_blackbox(simple_dnn)

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
            return (simple_dnn(*x_p) - simple_dnn(*x_m)) / (2 * delta)

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
        dnn_fn: 黑盒神经网络函数 simple_dnn(x, y)
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

''' 19年论文的符号恢复方法，只能恢复不含偏置的神经网络
def construct_X_in_same_cell(Z, base_point=None, r=0.05):
    """
    构造一组落在同一个 cell 中的线性无关输入点 X
    Z: shape (h, d) —— 每行是一个神经元方向（你恢复出的无符号方向）
    base_point: 起点，如果不给就随机生成一个
    r: 小扰动半径
    返回: X ∈ R^{d × h}
    """
    h, d = Z.shape

    if base_point is None:
        # 随机起点，尽量不太靠近边界
        base_point = np.random.uniform(low=-0.5, high=0.5, size=(d,))
    
    # 确保 base_point 在某个 cell 中
    g0 = (Z @ base_point >= 0).astype(int)  # 初始激活模式

    X = []
    count = 0
    while len(X) < h and count < 1000:
        # 在 base_point 附近加小扰动
        delta = np.random.randn(d)
        delta /= np.linalg.norm(delta)
        point = base_point + r * delta

        g = (Z @ point >= 0).astype(int)
        if np.all(g == g0):  # 与 base_point 同一 cell
            X.append(point)
        count += 1

    if len(X) < h:
        raise ValueError("未能成功构造足够的 X")

    return np.stack(X, axis=1)  # 返回形状 d × h

def recover_S(Z):
    """
    算法1b：恢复s
    Z: 每行是一个神经元恢复出的无符号归一化方向（形状为 h × d）
    witnesses: 一个包含多个 witness 点的列表
    r: 小扰动半径
    返回：恢复的参数 s
    """
    # 步骤 2: 构造 X，使得每个点落在同一个 cell 中
    X = construct_X_in_same_cell(Z)

    print("构造的 X:")
    print(X)
    print("Z:")
    print(Z)

    # 步骤 3: 计算 M，构造方阵
    ZX = Z @ X  # 这是一个 h × h 的矩阵，代表系统的方程系数
    M = np.block([
        [np.maximum(ZX, 0).T, np.maximum(-ZX, 0).T],
        [np.maximum(-ZX, 0).T, np.maximum(ZX, 0).T]
    ])  # 使用 ReLU 操作构造 M

    target = np.array([simple_dnn(*X[:, j]) for j in range(X.shape[1])] + [simple_dnn(*(-X[:, j])) for j in range(X.shape[1])])

    print("构造的 target:")
    print(target)
    # 这里的 target 是一个 2h × 1 的向量，包含了 f(x) 和 -f(x) 的值

    # 5. 解线性方程 M @ s = y
    s, residuals, rank, _ = np.linalg.lstsq(M, target, rcond=None)

    # 返回恢复的参数 s
    return s
'''

def recover_output_layer_via_linsolve(inputs, recovered_ws, recovered_bs, dnn_fn):
    """
    构造一个线性系统并精确求解输出层权重和偏置

    参数:
        inputs: List of input pairs [(x1, x2), ...]，至少 4 个
        recovered_ws: 恢复的隐藏层权重向量
        recovered_bs: 恢复的隐藏层偏置
        dnn_fn: 原始网络（用于查询真实输出）

    返回:
        wo: 输出层权重（长度 3）
        bo: 输出层偏置（float）
    """
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
recovered_ws = []  # 权重方向带符号
recovered_bs = []  # 偏置项

for j, x_star in enumerate(witnesses, 1):
    Δ1, Δ2 = compute_deltas(x_star)

    # 归一化权重（无符号）
    w_dir = np.array([Δ1, Δ2])
    #w_dir /= np.linalg.norm(w_dir)

    Z.append(w_dir.copy())  # 👉 保存恢复符号前的方向向量（注意要用 copy 防止后续修改）

    # 恢复符号
    sign = recover_signs([x_star], w_dir.reshape(1, -1), simple_dnn)[0]
    w_dir *= sign  # 恢复符号

    # 偏置恢复
    b = -np.dot(w_dir, x_star)
    
    # 收集结果
    recovered_ws.append(w_dir)
    recovered_bs.append(b)

    #'''调试内容
    # 与真实参数对比
    true_w = true_ws[j - 1]
    true_w_norm = true_w / np.linalg.norm(true_w)
    true_b = true_bs[j - 1] / np.linalg.norm(true_w)

    print(f"神经元 η{j}:")
    print(f"  Δ1 = {Δ1:.6f}, Δ2 = {Δ2:.6f}")
    print(f"  提取方向 = [{w_dir[0]:.6f}, {w_dir[1]:.6f}], 真实方向 = [{true_w_norm[0]:.6f}, {true_w_norm[1]:.6f}]")
    print(f"  提取偏置 = {b:.6f}, 真实偏置 = {true_b:.6f}")
    print()
    #'''

# 使用 10 个样本来保证稳健性
X_inputs = np.random.uniform(-1, 1, size=(10, 2))
X_inputs = list(X_inputs)
print(X_inputs)

wo, bo = recover_output_layer_via_linsolve(X_inputs, recovered_ws, recovered_bs, simple_dnn)
print("恢复的输出层参数:")
print("wo:", wo)
print("bo:", bo)

# 比较两个网络输出
diffs = []
for x1, x2 in X_inputs:
    original = simple_dnn(x1, x2)
    rebuilt = rebuilt_dnn(x1, x2, recovered_ws, recovered_bs, wo, bo)
    diffs.append(abs(original - rebuilt))

diffs = np.array(diffs)
print("最大误差:", np.max(diffs))
print("平均误差:", np.mean(diffs))
print("是否完全一致（误差<1e-5）:", np.all(diffs < 1e-5))




