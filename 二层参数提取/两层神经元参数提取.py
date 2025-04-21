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

def random_with_sign(original_weights, seed=42, low=0.1, high=1.0):
    np.random.seed(seed)
    rand_vals = np.random.uniform(low, high, size=len(original_weights))
    return [np.sign(w) * r for w, r in zip(original_weights, rand_vals)]

# 设置随机种子
seed = 2020

# 第一层参数（输入 -> 第一隐藏层）
w1 = random_with_sign([0.2, -0.3], seed + 1); b1 = np.sign(0.1) * np.random.uniform(0.1, 1.0)
w2 = random_with_sign([0.4, 0.5], seed + 2);  b2 = np.sign(-0.2) * np.random.uniform(0.1, 1.0)
w3 = random_with_sign([-0.6, 0.1], seed + 3); b3 = np.sign(0.3) * np.random.uniform(0.1, 1.0)

# 第二层参数（第一隐藏层 -> 第二隐藏层）
w4 = random_with_sign([0.3, -0.5, 0.2], seed + 4); b4 = np.sign(-0.1) * np.random.uniform(0.1, 1.0)
w5 = random_with_sign([-0.4, 0.1, 0.6], seed + 5); b5 = np.sign(0.2) * np.random.uniform(0.1, 1.0)
w6 = random_with_sign([0.7, 0.3, -0.2], seed + 6); b6 = np.sign(-0.3) * np.random.uniform(0.1, 1.0)

# 输出层参数（第二隐藏层 -> 输出）
wo = random_with_sign([0.7, -0.5, 0.2], seed + 7); bo = np.sign(0.1) * np.random.uniform(0.1, 1.0)

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
    N = 10000
    alphas = np.linspace(low, high, N + 1)
    for i in range(N):
        search(alphas[i], alphas[i+1])

    return np.array(critical_points)

# ========= 路径批量运行 =========
def run_sweep_multiple_paths(model, num_paths=100, input_range=10.0):
    all_breakpoints = []

    # 随机生成路径
    for i in range(num_paths):
        #np.random.seed(i*2)  # 随机种子
        u = np.random.uniform(-input_range, input_range, 2)
        v = np.random.uniform(-input_range, input_range, 2)
        print(f"\n路径 {i + 1}:")
        breakpoints = do_better_sweep_blackbox(model, u, v)

        for pt in breakpoints:
            print(f"  断点坐标: ({pt[0]:.8f}, {pt[1]:.8f})")

        all_breakpoints.extend(breakpoints)

    print(f"\n共找到断点总数: {len(all_breakpoints)}")
    return np.array(all_breakpoints)

# 3. 差分参数
epsilon = 1e-2   # 用于穿过激活边界 这里参数是取的随机点u和v的千分之一
delta   = 1e-3   # 用于数值导数


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

def recover_signs_with_votes(witnesses_list, Z, dnn_fn, epsilon=0.01, alpha0=0.6):
    signs = []
    votes_record = []  # 新增

    for i, witnesses in enumerate(witnesses_list):
        z = Z[i]
        direction = z / np.linalg.norm(z) * epsilon

        votes = []
        for x, y in witnesses:
            x_plus, y_plus = x + direction[0], y + direction[1]
            x_minus, y_minus = x - direction[0], y - direction[1]

            f0 = dnn_fn(x, y)
            f_plus = dnn_fn(x_plus, y_plus)
            f_minus = dnn_fn(x_minus, y_minus)

            delta_plus = abs(f_plus - f0)
            delta_minus = abs(f_minus - f0)

            vote = +1 if delta_plus > delta_minus else -1
            votes.append(vote)

        s_plus = votes.count(+1)
        s_minus = votes.count(-1)
        s_total = len(votes)

        alpha = max(s_plus, s_minus) / s_total
        votes_record.append((s_minus, s_plus, alpha))

        if s_plus / s_total >= alpha0:
            signs.append(+1)
        elif s_minus / s_total >= alpha0:
            signs.append(-1)
        else:
            signs.append(0)

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


witnesses = run_sweep_multiple_paths(dnn_2_3_3_1)

Z=[]
recovered_ws = []  # 权重方向未筛选
recovered_bs = []  # 偏置项

for j, x_star in enumerate(witnesses, 1):
    Δ1, Δ2 = compute_deltas(x_star)

    # 归一化权重（无符号）
    w_dir = np.array([Δ1, Δ2])
    #w_dir /= np.linalg.norm(w_dir)
    
    # 收集结果
    recovered_ws.append(w_dir)

print("\n恢复的权重方向（未筛选）：")
for i, w in enumerate(recovered_ws):
    print(f"神经元 {i + 1}: {w}")

from collections import defaultdict

def filter_repeated_directions(recovered_ws, k=3, decimals=6, deviation_threshold=0.1):
    from collections import defaultdict
    import numpy as np

    def unify_direction(w, ndigits=11):
        if np.allclose(w, 0):  # 判断是否全 0 向量
            return None
        if w[0] < 0 or (w[0] == 0 and w[1] < 0):
            w = -w
        return tuple(round(x, ndigits) for x in w)

    direction_counts = defaultdict(list)
    for idx, w in enumerate(recovered_ws):
        dir_key = unify_direction(w)
        if dir_key is not None:
            direction_counts[dir_key].append(idx)

    results = []
    all_witnesses = []

    sorted_directions = sorted(direction_counts.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (direction, indices) in enumerate(sorted_directions[:k]):
        print(f"\n方向 {i+1}: {direction}，原始出现次数: {len(indices)}")
        dir_vec = np.array(direction)
        dot_products = []

        # 收集点乘值
        for idx in indices:
            witness = np.array(witnesses[idx])
            dot = np.dot(dir_vec, witness)
            dot_products.append(dot)

        dot_products = np.array(dot_products)
        median_dot = np.median(dot_products)

        # 过滤异常 witness
        filtered_witnesses = []
        for idx, dot in zip(indices, dot_products):
            if abs(dot - median_dot) <= deviation_threshold:
                filtered_witnesses.append(witnesses[idx])
            else:
                print(f"  ⚠️ 异常 witness 被筛除: {witnesses[idx]}，点乘值: {dot:.5f}，偏离中位数: {abs(dot - median_dot):.5f}")

        print(f"  剩余合法 witness 数: {len(filtered_witnesses)}")
        for j, w in enumerate(filtered_witnesses):
            print(f"    witness {j+1}: {w}")

        results.append(direction)
        all_witnesses.append(filtered_witnesses)

    return results, all_witnesses




filtered_directions,witnesses_list = filter_repeated_directions(recovered_ws, k=3)
print(filtered_directions)

# 恢复符号
sign = recover_signs_with_votes(witnesses_list,filtered_directions, dnn_2_3_3_1)
filtered_directions = np.array(filtered_directions) * np.array(sign)[:, np.newaxis]  # 恢复符号
b=[]
#'''调试代码
print("\n恢复的权重方向（筛选后）：")
for i, w in enumerate(filtered_directions):
    print(f"神经元 {i + 1}: {w}")
#'''
filtered_directions = np.array(filtered_directions, dtype=np.float64)  # 防止 tuple 出错

b = []
for i, w in enumerate(filtered_directions):
    witness_points = witnesses_list[i]
    dot_products = [np.dot(w, x) for x in witness_points]
    b_i = -np.mean(dot_products)
    b.append(b_i)


print("\n恢复的偏置项：")   
for i, b_i in enumerate(b):
    print(f"神经元 {i + 1}: {b_i}")



print("正确的权重（单位向量）和对应归一化偏置：")
for i, (w_i, b_i) in enumerate(zip([w1, w2, w3], [b1, b2, b3])):
    norm = np.linalg.norm(w_i)
    w_unit = np.array(w_i) / norm
    b_scaled = b_i / norm
    print(f"神经元 {i + 1}: 权重 = {w_unit}，偏置 = {b_scaled}")

from scipy.spatial.distance import cosine

print("\n恢复方向与真实方向的匹配结果：")

true_weights = [w1, w2, w3]
true_biases = [b1, b2, b3]    
matched = set()  # 避免重复匹配同一个真实神经元

for i, (w_rec, b_rec) in enumerate(zip(filtered_directions, b)):
    best_match = None
    best_cos_sim = -np.inf
    best_j = -1

    for j, w_true in enumerate(true_weights):
        if j in matched:
            continue
        w_true_unit = np.array(w_true) / np.linalg.norm(w_true)
        cos_sim = np.dot(w_rec, w_true_unit)
        if abs(cos_sim) > best_cos_sim:  # 注意取绝对值，允许符号相反
            best_cos_sim = abs(cos_sim)
            best_match = (j, cos_sim)
    
    j, cos_sim = best_match
    matched.add(j)
    w_true = np.array(true_weights[j])
    norm = np.linalg.norm(w_true)
    w_true_unit = w_true / norm
    b_true_scaled = true_biases[j] / norm

    # 确保方向一致（符号对齐）
    if np.dot(w_rec, w_true_unit) < 0:
        w_rec = -w_rec
        b_rec = -b_rec

    # 偏置比例
    bias_ratio = b_rec / b_true_scaled if b_true_scaled != 0 else float('inf')

    print(f"恢复神经元 {i+1} 匹配到真实神经元 {j+1}")
    print(f"  方向余弦相似度: {cos_sim:.8f}")
    print(f"  偏置比值: {bias_ratio:.8f}")



