# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict    
def matmul(a,b,c,np=np):
    if c is None:
        c = np.zeros(1)

    return np.dot(a,b)+c

class KnownT:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def extend_by(self, a, b):
        return KnownT(self.A+[a], self.B+[b])
        
    def forward(self, x, with_relu=False, np=np):
        for i,(a,b) in enumerate(zip(self.A,self.B)):
            x = matmul(x,a,b,np)
            if (i < len(self.A)-1) or with_relu:
                x = x*(x>0)
        return x
    def forward_at(self, point, d_matrix):
        if len(self.A) == 0:
            return d_matrix

        mask_vectors = [layer > 0 for layer in self.get_hidden_layers(point)]

        h_matrix = np.array(d_matrix)
        for i,(matrix,mask) in enumerate(zip(self.A, mask_vectors)):
            h_matrix = matmul(h_matrix, matrix, None) * mask
        
        return h_matrix
    def get_hidden_layers(self, x, flat=False, np=np):
        if len(self.A) == 0: return []
        region = []
        for i,(a,b) in enumerate(zip(self.A,self.B)):
            x = matmul(x,a,b,np=np)
            region.append(np.copy(x))
            if i < len(self.A)-1:
                x = x*(x>0)
        if flat:
            region = np.concatenate(region,axis=0)
        return region
    def get_polytope(self, x):
        if len(self.A) == 0: return tuple()
        h = self.get_hidden_layers(x)
        h = np.concatenate(h, axis=0)
        return tuple(np.int32(np.sign(h)))



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


recovered_ws= np.array([w1, w2, w3])
recovered_bs= np.array([b1, b2, b3])
known_T = KnownT([recovered_ws.T], [recovered_bs])

def get_hidden_output(x):
    z = np.dot(recovered_ws, x) + recovered_bs  
    h = np.maximum(0, z)    # ReLU 或 sigmoid(z)
    return h

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

def check_zero_neurons(witnesses, tol=1e-3):
    for idx, point in enumerate(witnesses):
        x, y = point
        output, layer1, layer2 = dnn_2_3_3_1_test(x, y)
        zero_layer1 = [i+1 for i, a in enumerate(layer1) if abs(a) < tol]
        zero_layer2 = [i+4 for i, a in enumerate(layer2) if abs(a) < tol]
        
        print(f"点 {idx+1}: ({x:.4f}, {y:.4f})")
        print(f"  第一层 ReLU 输出近似为 0 的神经元编号: {zero_layer1}")
        print(f"  第二层 ReLU 输出近似为 0 的神经元编号: {zero_layer2}")
        print(f"  网络输出值: {output:.4f}\n")

def filter_witnesses_by_layer2_zeros(witnesses, tol=1e-3):
    filtered = []
    for point in witnesses:
        x, y = point
        _, _, layer2 = dnn_2_3_3_1_test(x, y)
        zero_layer2 = [i+4 for i, a in enumerate(layer2) if abs(a) < tol]
        if any(i in zero_layer2 for i in [4, 5, 6]):
            filtered.append(point)
    return filtered

witnesses = run_sweep_multiple_paths(dnn_2_3_3_1)



# %%
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
print("第二层参数(归一):")
print(f"             w4 = {w4/(w4[0])}, b4 = {b4:.4f}")
print(f"             w5 = {w5/(w5[0])}, b5 = {b5:.4f}")
print(f"             w6 = {w6/(w6[0])}, b6 = {b6:.4f}")
# %%
class AcceptableFailure(Exception):
    """
    Sometimes things fail for entirely acceptable reasons (e.g., we haven't
    queried enough points to have seen all the hyperplanes, or we get stuck
    in a constant zero region). When that happens we throw an AcceptableFailure
    because life is tough but we should just back out and try again after
    making the appropriate correction.
    """
    def __init__(self, *args, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


MASK = np.array([1,-1,1,-1])

def dnn_forward1(x):
    return dnn_2_3_3_1(x[0], x[1])

def get_hidden_output(x):
    z = np.dot(recovered_ws, x) + recovered_bs  
    h = np.maximum(0, z)    # ReLU 或 sigmoid(z)
    return h

def get_second_grad_unsigned(x, direction, eps, eps2):
    # 计算四个扰动点
    x_perturbed = np.array([
        x + direction * (eps - eps2),
        x + direction * eps,
        x - direction * (eps - eps2),
        x - direction * eps
    ])

    # 使用 dnn_forward1 来处理每个扰动点的输入，计算输出
    out = np.array([dnn_forward1(xi) for xi in x_perturbed])

    # 计算二阶导数近似（这里加了 MASK 投影）
    return np.dot(out.flatten(), MASK) / eps


def get_ratios_lstsq(critical_points, known_T, eps=1e-6):
    ratios = []
    for i,point in enumerate(critical_points):
        d_matrix = []
        ys = []
        for i in range(np.sum(known_T.forward(point) != 0)+2):
            d = np.random.normal(0,1,point.shape)#生成一个与 point 同形状的高斯随机向量
            d_matrix.append(d)
            ratio_val = get_second_grad_unsigned(point, d, eps, eps/3)
            if len(ys) > 0:
                both_ratio_val = get_second_grad_unsigned(point, (d+d_matrix[0])/2, eps, eps/3)

                positive_error = abs(abs(ys[0]+ratio_val)/2 - abs(both_ratio_val))
                negative_error = abs(abs(ys[0]-ratio_val)/2 - abs(both_ratio_val))
                if positive_error > 1e-4 and negative_error > 1e-4:
                    print("Probably something is borked")
                    print("d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))", positive_error, negative_error)
                    raise AcceptableFailure()
                if negative_error < positive_error:
                    ratio_val *= -1
            
            ys.append(ratio_val)

        d_matrix = np.array(d_matrix)

        h_matrix = np.array(known_T.forward_at(point, d_matrix))

        column_is_zero = np.mean(np.abs(h_matrix)<1e-8,axis=0) > .5
        assert np.all((known_T.forward(point, with_relu=True) == 0) == column_is_zero)

        soln, *rest = np.linalg.lstsq(np.array(h_matrix, dtype=np.float32),
                                      np.array(ys, dtype=np.float32), 1e-5)
    
        soln[column_is_zero] = np.nan

        ratios.append(soln)
        
    return ratios

filtered_witnesses = filter_witnesses_by_layer2_zeros(witnesses)
#check_zero_neurons(filtered_witnesses)

witnesses_update = []
for xstar in filtered_witnesses:
    result = get_hidden_output(xstar)
    if all(result):  # 检查三个都非零
        witnesses_update.append(xstar)



def compute_weights(x_star, dnn_forward, hidden_layer_output, d_hidden=3, delta=1e-4):
    """
    使用有限差分 + 符号纠正 + 最小二乘恢复第二层单个神经元的权重行向量
    """
    d_input = x_star.shape[0]
    num_directions = d_hidden + 1  # 稍微多取几个方向做 least squares 更稳定
    delta_vectors = np.random.normal(0, 1.0, size=(num_directions, d_input))
    delta_vectors /= np.linalg.norm(delta_vectors, axis=1, keepdims=True)

    h_list = []
    y_list = []

    for i in range(num_directions):
        δi = delta_vectors[i]
        δ1 = delta_vectors[0]  # 第一个方向用于与其他方向进行符号校正

        # 二阶导数近似 ∂²f / ∂δ1∂δi
        f_pp = dnn_forward(x_star + delta * δ1 + delta * δi)
        f_pm = dnn_forward(x_star + delta * δ1 - delta * δi)
        f_mp = dnn_forward(x_star - delta * δ1 + delta * δi)
        f_mm = dnn_forward(x_star - delta * δ1 - delta * δi)

        y_i = (f_pp - f_pm - f_mp + f_mm) / (4 * delta ** 2)

        if i > 0:
            # 符号对齐：判断 y_i 应该是正还是负
            f_mix = dnn_forward(x_star + delta * (δ1 + δi))
            expected_plus = abs(y_list[0] + y_i) / 2
            expected_minus = abs(y_list[0] - y_i) / 2
            mix_val = f_mix - dnn_forward(x_star)

            if abs(expected_minus - abs(mix_val)) < abs(expected_plus - abs(mix_val)):
                y_i *= -1

        y_list.append(y_i)

        h_i = hidden_layer_output(x_star + delta * δi)
        h_list.append(h_i)

    H = np.array(h_list)  # shape=(num_directions, d_hidden)
    y = np.array(y_list)  # shape=(num_directions,)

    # 检查某些维度是否恒为 0（ReLU 截断），这些维度无法恢复
    column_is_zero = np.mean(np.abs(H) < 1e-8, axis=0) > 0.5

    # 最小二乘求解 H·w = y
    w, *_ = np.linalg.lstsq(H, y, rcond=None)

    # 对于无解维度标 NaN
    w[column_is_zero] = np.nan
    return w

def normalize_rows_with_nan(matrix):
    matrix = np.array(matrix, dtype=np.float64)
    result = np.empty_like(matrix)
    
    for i in range(matrix.shape[0]):
        row = matrix[i]
        abs_row = np.abs(row)
        valid = ~np.isnan(abs_row)
        if not np.any(valid):
            result[i] = row  # 全是 nan，不动
            continue
        
        scale = np.nanmax(abs_row)
        if scale < 1e-8:
            result[i] = row  # 近似为零的行，不动
        else:
            result[i] = row / scale

    return result

recovered_w2 = []

for x_star in filtered_witnesses:  # 多个 witness 点恢复多个第二层神经元的权重行向量
    
    w_row = get_ratios_lstsq( [x_star],known_T)[0].flatten()

    if np.abs(w_row[0]) > 1e-14:
        w_row = w_row / w_row[0]
    recovered_w2.append(w_row)


recovered_w2 = np.array(recovered_w2)
recovered_w2 = normalize_rows_with_nan(recovered_w2)
print("恢复的第二层权重矩阵：")
print(recovered_w2)



import numpy as np
import networkx as nx

def is_proportional_sparse(v1, v2, atol=1e-2):
    """只在两个向量的非零交集上判断是否成比例"""
    idx = (np.abs(v1) > 1e-6) & (np.abs(v2) > 1e-6)
    if np.sum(idx) < 2:
        return False
    ratio = v1[idx] / v2[idx]
    return np.allclose(ratio, ratio[0], atol=atol)

def normalize_sparse(v):
    """稀疏向量的归一化，只除以最大非零值"""
    nonzero = v[np.abs(v) > 1e-16]
    if len(nonzero) == 0:
        return v
    scale = nonzero[np.argmax(np.abs(nonzero))]
    return v / scale

def merge_sparse_vectors(vectors):
    """将一组稀疏、成比例的向量合并为一个完整向量（非零位置平均）"""
    normalized = np.array([normalize_sparse(v) for v in vectors])
    dim = normalized[0].shape[0]
    result = np.zeros(dim)
    for i in range(dim):
        values = normalized[:, i][np.abs(normalized[:, i]) > 1e-6]
        if len(values) > 0:
            result[i] = np.mean(values)
    return result

def graph_solve_sparse(vectors):
    """
    vectors: List[np.ndarray]，每个都是稀疏向量（部分神经元方向）
    返回合并后的权重向量集合，每个向量代表一个神经元方向
    """
    n = len(vectors)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if is_proportional_sparse(vectors[i], vectors[j]):
                G.add_edge(i, j)

    components = list(nx.connected_components(G))
    merged = []
    for comp in components:
        group = [vectors[i] for i in comp]
        merged.append(merge_sparse_vectors(group))

    return np.array(merged)


merged_w2 = graph_solve_sparse(recovered_w2)
print("合并后的第二层权重：")
print(merged_w2)
# %%