# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict    


VERBOSE = True  # 设置为 False 以关闭所有调试输出

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# 1. 定义黑盒模型 f
def relu(z):
    return max(0, z)

def dnn_2_3_3_1_test(x, y):
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
    return output, [z1, z2, z3], [z4, z5, z6]

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

# 设置随机种子，差分参数
seed = 2020
epsilon = 1e-2   # 用于穿过激活边界 这里参数是取的随机点u和v的千分之一
delta   = 1e-3   # 用于数值导数

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

def check_zero_neurons(witnesses, tol=1e-4):
    for idx, point in enumerate(witnesses):
        x, y = point
        output, layer1, layer2 = dnn_2_3_3_1_test(x, y)
        zero_layer1 = [i+1 for i, a in enumerate(layer1) if abs(a) < tol]
        zero_layer2 = [i+4 for i, a in enumerate(layer2) if abs(a) < tol]
        
        print(f"点 {idx+1}: ({x:.6f}, {y:.6f})")
        print(f"  第一层 ReLU 输出近似为 0 的神经元编号: {zero_layer1}")
        print(f"  第二层 ReLU 输出近似为 0 的神经元编号: {zero_layer2}")
        print(f"  网络输出值: {output:.4f}\n")

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
        vprint(f"\n路径 {i + 1}:")
        breakpoints = do_better_sweep_blackbox(model, u, v)

        for pt in breakpoints:
            vprint(f"  断点坐标: ({pt[0]:.8f}, {pt[1]:.8f})")

        all_breakpoints.extend(breakpoints)

    vprint(f"\n共找到断点总数: {len(all_breakpoints)}")
    return np.array(all_breakpoints)

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

def filter_repeated_directions(recovered_ws, witnesses, k=3, decimals=6, deviation_threshold=0.1):
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

    used_indices = set()

    sorted_directions = sorted(direction_counts.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (direction, indices) in enumerate(sorted_directions[:k]):
        vprint(f"\n方向 {i+1}: {direction}，原始出现次数: {len(indices)}")
        dir_vec = np.array(direction)
        dot_products = []

        for idx in indices:
            witness = np.array(witnesses[idx])
            dot = np.dot(dir_vec, witness)
            dot_products.append(dot)

        dot_products = np.array(dot_products)
        median_dot = np.median(dot_products)

        filtered_witnesses = []
        for idx, dot in zip(indices, dot_products):
            if abs(dot - median_dot) <= deviation_threshold:
                filtered_witnesses.append(witnesses[idx])
                used_indices.add(idx)  # 记录已使用 witness
            else:
                vprint(f"  ⚠️ 异常 witness 被筛除: {witnesses[idx]}，点乘值: {dot:.5f}，偏离中位数: {abs(dot - median_dot):.5f}")

        vprint(f"  剩余合法 witness 数: {len(filtered_witnesses)}")
        for j, w in enumerate(filtered_witnesses):
            vprint(f"    witness {j+1}: {w}")

        results.append(direction)
        all_witnesses.append(filtered_witnesses)

    updated_witnesses = [w for i, w in enumerate(witnesses) if i not in used_indices]
    return results, all_witnesses, updated_witnesses

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


if __name__=='__main__':
    print("=========================================")
    print("两层 ReLU 网络的参数提取 (2-3-3-1)")
    print("=========================================")
    print("输入范围: [-10, 10]")
    print("激活边界: 0")
    print("=========================================")
    print("模型参数:")
    print(f"第一层参数: w1 = {w1}, b1 = {b1:.4f}")
    print(f"             w2 = {w2}, b2 = {b2:.4f}")
    print(f"             w3 = {w3}, b3 = {b3:.4f}")
    print(f"第二层参数: w4 = {w4}, b4 = {b4:.4f}")
    print(f"             w5 = {w5}, b5 = {b5:.4f}")
    print(f"             w6 = {w6}, b6 = {b6:.4f}")
    print(f"输出层参数: wo = {wo}, bo = {bo:.4f}")
    print("=========================================")
    print("模型结构:")
    print("输入层: 2个节点 (x, y)")
    print("第一隐藏层: 3个神经元")
    print("第二隐藏层: 3个神经元")
    print("输出层: 1个输出节点")
    print("===============开始恢复==================")
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

    vprint("\n恢复的权重方向（未筛选）：")
    for i, w in enumerate(recovered_ws):
        vprint(f"神经元 {i + 1}: {w}")

    recovered_ws,witnesses_list,witnesses = filter_repeated_directions(recovered_ws,witnesses, k=3)

    # 恢复符号
    sign = recover_signs_with_votes(witnesses_list,recovered_ws, dnn_2_3_3_1)
    recovered_ws = np.array(recovered_ws) * np.array(sign)[:, np.newaxis]  # 恢复符号
    recovered_ws = np.array(recovered_ws, dtype=np.float64)  # 防止 tuple 出错

    for i, w in enumerate(recovered_ws):
        witness_points = witnesses_list[i]
        dot_products = [np.dot(w, x) for x in witness_points]
        b_i = -np.mean(dot_products)
        recovered_bs.append(b_i)

    '''测试代码
    vprint("\n恢复的权重方向（筛选后）：")
    for i, w in enumerate(recovered_ws):
        vprint(f"神经元 {i + 1}: {w}")
    vprint("\n恢复的偏置项：")   
    for i, b_i in enumerate(recovered_bs):
        vprint(f"神经元 {i + 1}: {b_i}")



    vprint("正确的权重（单位向量）和对应归一化偏置：")
    for i, (w_i, b_i) in enumerate(zip([w1, w2, w3], [b1, b2, b3])):
        norm = np.linalg.norm(w_i)
        w_unit = np.array(w_i) / norm
        b_scaled = b_i / norm
        vprint(f"神经元 {i + 1}: 权重 = {w_unit}，偏置 = {b_scaled}")


    from scipy.spatial.distance import cosine

    vprint("\n恢复方向与真实方向的匹配结果：")



    '''
    true_weights = [w1, w2, w3]
    true_biases = [b1, b2, b3]    
    matched = set()  # 避免重复匹配同一个真实神经元

    for i, (w_rec, b_rec) in enumerate(zip(recovered_ws, recovered_bs)):
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
    print("===============第一层恢复完成==================")

# %%
def get_hidden_output(x):
    z = np.dot(recovered_ws, x) + recovered_bs  
    h = np.maximum(0, z)    # ReLU 或 sigmoid(z)
    return h

   
witnesses_update = []
for xstar in witnesses:
    result = get_hidden_output(xstar)
    if all(result):  # 检查三个都非零
        witnesses_update.append(xstar)

# %%
def dnn_forward1(x):
    return dnn_2_3_3_1(x[0], x[1])

def compute_weights(x_star, dnn_forward, hidden_layer_output, d_hidden=3, delta=1e-4):
    """
    恢复第二层权重中一个神经元的权重向量 w（对应 y = h·w）
    输入：
        x_star: witness 点，shape = (d_input,)
        dnn_forward: 神经网络前向函数，输入 shape=(2,) 返回输出标量
        hidden_layer_output: 返回隐藏层输出向量 h，输入 shape=(2,) 返回 shape=(d_hidden,)
        d_hidden: 隐藏层维度（即需要恢复的权重维度）
        delta: 有限差分使用的扰动大小
    输出：
        w: 当前神经元的权重向量 shape=(d_hidden,)
    """
    d_input = x_star.shape[0]
    num_directions = d_hidden + 1  # 至少 d_hidden+1 个扰动方向构成线性方程组
    delta_vectors = np.random.normal(0, 1.0, size=(num_directions, d_input))  # 方向向量 δ_i

    h_list = []
    y_list = []

    for i in range(num_directions):
        δi = delta_vectors[i]
        δ1 = delta_vectors[0]

        # 二阶导数近似 ∂²f / ∂δ1∂δi
        f_pp = dnn_forward(x_star + δ1 * delta + δi * delta)
        f_pm = dnn_forward(x_star + δ1 * delta - δi * delta)
        f_mp = dnn_forward(x_star - δ1 * delta + δi * delta)
        f_mm = dnn_forward(x_star - δ1 * delta - δi * delta)

        y_i = (f_pp - f_pm - f_mp + f_mm) / (4 * delta ** 2)
        y_list.append(y_i)

        h_i = hidden_layer_output(x_star + δi * delta)  # 前 j-1 层输出
        h_list.append(h_i)

    # 解线性系统 H·w = y
    H = np.array(h_list)      # shape=(d_hidden+1, d_hidden)
    y = np.array(y_list)      # shape=(d_hidden+1,)
    w, _, _, _ = np.linalg.lstsq(H, y, rcond=None)
    return w

recovered_w2 = []

for x_star in witnesses_update:  # 多个 witness 点恢复多个第二层神经元的权重行向量
    w_row = compute_weights(
        x_star=x_star,
        dnn_forward=dnn_forward1,
        hidden_layer_output=get_hidden_output,  # 你要实现这个函数
        d_hidden=3
    )
    recovered_w2.append(w_row)

recovered_w2 = np.array(recovered_w2)
print("恢复的第二层权重矩阵：")
print(recovered_w2)



import numpy as np
import networkx as nx

def is_proportional(v1, v2, atol=1e-2):
    """ 判断两个向量是否成比例 """
    idx = (v1 != 0) & (v2 != 0)
    if np.sum(idx) < 2:  # 至少要有两个非零才能判断比例
        return False

    ratio = v1[idx] / v2[idx]
    return np.allclose(ratio, ratio[0], atol=atol)

def normalize(v):
    """ 归一化向量，除以其最大值的绝对值 """
    nonzero = v[np.abs(v) > 1e-8]
    if len(nonzero) == 0:
        return v
    scale = nonzero[np.argmax(np.abs(nonzero))]
    return v / scale

def merge_vectors(vectors):
    """ 合并一组成比例向量（取非零项平均）"""
    vectors = np.array(vectors)
    normalized = np.array([normalize(v) for v in vectors])
    result = np.zeros_like(normalized[0])
    for i in range(result.shape[0]):
        nonzeros = normalized[:, i][np.abs(normalized[:, i]) > 1e-8]
        if len(nonzeros) > 0:
            result[i] = np.mean(nonzeros)
    return result

# 你的矩阵
W = np.array(recovered_w2)

# 构建图
G = nx.Graph()
G.add_nodes_from(range(len(W)))

for i in range(len(W)):
    for j in range(i + 1, len(W)):
        if is_proportional(W[i], W[j]):
            G.add_edge(i, j)

# 找到所有连通分量（成比例组）
components = list(nx.connected_components(G))

print(f"找到成比例组的数量: {len(components)}")

# 合并每个组
merged_weights = []
for comp in components:
    group = [W[i] for i in comp]
    merged = merge_vectors(group)
    merged_weights.append(merged)

merged_weights = np.array(merged_weights)
print("合并后的权重矩阵：")
print(merged_weights)





# %%
