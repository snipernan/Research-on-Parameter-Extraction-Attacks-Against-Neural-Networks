import numpy as np

# 1. 定义黑盒模型 f
def relu(z):
    return max(0, z)

def o_dnn_2_3_3_1(x, y):
    # 第一隐藏层 z = w·x + b
    z1 = w1[0]*x + w1[1]*y + b1
    z2 = w2[0]*x + w2[1]*y + b2
    z3 = w3[0]*x + w3[1]*y + b3
    print("\n第一隐藏层线性输出:")
    print("z1, z2, z3:", z1, z2, z3)
    
    a1 = relu(z1)
    a2 = relu(z2)
    a3 = relu(z3)

    # 第二隐藏层：输入是第一层的输出 a1, a2, a3
    z4 = w4[0]*a1 + w4[1]*a2 + w4[2]*a3 + b4
    z5 = w5[0]*a1 + w5[1]*a2 + w5[2]*a3 + b5
    z6 = w6[0]*a1 + w6[1]*a2 + w6[2]*a3 + b6
    print("\n第二隐藏层线性输出:")
    print("z4, z5, z6:", z4, z5, z6)

    a4 = relu(z4)
    a5 = relu(z5)
    a6 = relu(z6)

    # 输出层：输入是第二层的输出 a4, a5, a6
    output = wo[0]*a4 + wo[1]*a5 + wo[2]*a6 + bo
    return output

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



x1=(0.09583715 ,-2.59299655)
x2=(-4.2395366 , -3.30341243)

output1=o_dnn_2_3_3_1(*x1)
output2=o_dnn_2_3_3_1(*x2)

Δ1, Δ2 = compute_deltas(x1)
Δ3, Δ4 = compute_deltas(x2)

# 归一化权重（无符号）
w_dir1 = np.array([Δ1, Δ2])

print("权重1:")
print(w_dir1)

w_dir2 = np.array([Δ3, Δ4])

print("权重2:")
print(w_dir2)
