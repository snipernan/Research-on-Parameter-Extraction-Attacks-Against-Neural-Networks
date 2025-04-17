import numpy as np
import matplotlib.pyplot as plt

# 网络参数
w1 = [0.2, -0.3]; b1 = 0.1
w2 = [0.4, 0.5];  b2 = -0.2
w3 = [-0.6, 0.1]; b3 = 0.3
wo = [0.7, -0.5, 0.2]; bo = 0.1

def relu(z):
    return np.maximum(0, z)

def simple_dnn(x1, x2):
    z1 = w1[0]*x1 + w1[1]*x2 + b1
    z2 = w2[0]*x1 + w2[1]*x2 + b2
    z3 = w3[0]*x1 + w3[1]*x2 + b3
    a1 = relu(z1)
    a2 = relu(z2)
    a3 = relu(z3)
    y = wo[0]*a1 + wo[1]*a2 + wo[2]*a3 + bo
    return y

#找变化点时直接调用了模型参数
def plot_dnn_along_path(u, v, eps=1e-3, num_points=500):
    """
    在输入空间中，从 u 到 v 路径上画出模型输出曲线，并标注分段位置
    """
    plt.style.use("dark_background")
    
    # 路径点采样
    alphas = np.linspace(0, 1, num_points)
    points = np.array([u * (1 - a) + v * a for a in alphas])
    outputs = np.array([simple_dnn(x[0], x[1]) for x in points])

    # 计算 z_i
    z1_vals = np.array([w1[0]*x[0] + w1[1]*x[1] + b1 for x in points])
    z2_vals = np.array([w2[0]*x[0] + w2[1]*x[1] + b2 for x in points])
    z3_vals = np.array([w3[0]*x[0] + w3[1]*x[1] + b3 for x in points])

    # 找近似为 0 的点作为分段边界
    breaks_alpha = alphas[(np.abs(z1_vals) < eps) | (np.abs(z2_vals) < eps) | (np.abs(z3_vals) < eps)]

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    ax.plot(alphas, outputs, color='orange', lw=1.8)
    for a in breaks_alpha:
        ax.axvline(x=a, color='white', linestyle='-', linewidth=1)
    
    ax.set_xlabel(r"$\alpha$", fontsize=14, color='white')
    ax.set_ylabel(r"$f(u(1-\alpha) + v\alpha)$", fontsize=14, color='white')
    ax.set_title("模型输出与分段位置", fontsize=16, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    plt.grid(False)
    plt.show()

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


def plot_dnn_along_path_blackbox(u, v, model, num_points=3000, std_factor=1.5, dedup_eps=1e-3, plot=True):
    plt.style.use("dark_background")
    
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

    

    # 可视化
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        ax.plot(alphas, outputs, color='orange', lw=1.8)
        for a in breaks_alpha:
            ax.axvline(x=a, color='white', linestyle='-', linewidth=1)

        ax.set_xlabel(r"$\alpha$", fontsize=14, color='white')
        ax.set_ylabel(r"$f(u(1-\alpha) + v\alpha)$", fontsize=14, color='white')
        ax.set_title("模型输出与推测的分段位置", fontsize=16, color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        plt.grid(False)
        plt.show()

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




u = np.array([-1.0, 0.1])
v = np.array([1.0, 0.1])
#plot_dnn_along_path(u, v)
plot_dnn_along_path_blackbox(u, v, simple_dnn)





