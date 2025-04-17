import numpy as np

# 黑盒神经网络
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

# 差分估计方向（梯度）
def estimate_gradient(f, x, epsilon=1e-3):
    grad = []
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = epsilon
        grad_i = (f(*(x + dx)) - f(*(x - dx))) / (2 * epsilon)
        grad.append(grad_i)
    return np.array(grad)

# 恢复带符号的方向向量
def recover_signed_directions(f, witnesses):
    directions = []
    for i, x in enumerate(witnesses):
        x = np.array(x)
        grad = estimate_gradient(f, x)

        # 方向归一化（当前是带输出层权重的）
        norm = np.linalg.norm(grad)
        if norm < 1e-6:
            print(f"⚠️ 神经元 η{i+1} 梯度太小，可能无法恢复")
            directions.append((grad, None))
            continue
        unit_direction = grad / norm

        # 做正负扰动判断符号
        eps = 1e-2
        x_plus = x + eps * unit_direction
        x_minus = x - eps * unit_direction
        delta = f(*x_plus) - f(*x_minus)

        if abs(delta) < 1e-4:
            sign = 0  # 无法判断
            print(f"⚠️ 神经元 η{i+1}: 输出变化太小，无法判断符号")
        else:
            sign = np.sign(delta)

        # 应用符号
        final_direction = sign * unit_direction
        print(f"神经元 η{i+1}:")
        print(f"  差分方向 = {grad}")
        print(f"  单位方向 = {unit_direction}")
        print(f"  符号判断 = {sign:+.0f}")
        print(f"✅ 恢复方向 = {final_direction}")
        print()

        directions.append((final_direction, sign))

    return directions

# 三个神经元的见证点（在边界上）
witnesses = [
    (-0.3498, 0.1000),  # η1
    ( 0.3751, 0.1000),  # η2
    ( 0.5172, 0.1000),  # η3
]

# 执行恢复
recover_signed_directions(simple_dnn, witnesses)
