import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# 设置中文字体（如果需要）
rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
rcParams['axes.unicode_minus'] = False  # 正常显示负号

def plot_activation_regions_with_breaks(break_points, grid_size=300):
    # 随便定义一些权重参数（固定以便可视化）
    # 第一层（2 -> 3）
    W1 = np.array([[0.2, -0.3],
                   [0.4, 0.5],
                   [-0.6, 0.1]])
    b1 = np.array([0.1, -0.2, 0.3])

    # 第二层（3 -> 3）
    W2 = np.array([[0.3, -0.5, 0.2],
                   [-0.4, 0.1, 0.6],
                   [0.7, 0.3, -0.2]])
    b2 = np.array([-0.1, 0.2, -0.3])

    # 激活模式编码函数：返回一个字符串，例如 "101|011"
    def activation_pattern(x1, x2):
        x = np.array([x1, x2])
        z1 = W1 @ x + b1
        a1 = np.maximum(z1, 0)
        z2 = W2 @ a1 + b2
        a2 = np.maximum(z2, 0)
        code1 = ''.join(['1' if z > 0 else '0' for z in z1])
        code2 = ''.join(['1' if z > 0 else '0' for z in z2])
        return f"{code1}|{code2}"

    # 构造输入网格
    x_range = np.linspace(-2, 2, grid_size)
    y_range = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x_range, y_range)

    # 为每个点计算激活模式
    patterns = np.empty((grid_size, grid_size), dtype=object)
    for i in range(grid_size):
        for j in range(grid_size):
            patterns[i, j] = activation_pattern(X[i, j], Y[i, j])

    # 将激活模式映射为整数编码
    unique_patterns = list(sorted(set(patterns.flatten())))
    pattern_to_int = {pat: idx for idx, pat in enumerate(unique_patterns)}
    int_patterns = np.vectorize(lambda p: pattern_to_int[p])(patterns)

    # 绘图
    plt.figure(figsize=(8, 8), facecolor='white')
    plt.imshow(int_patterns, extent=[-2, 2, -2, 2], origin='lower', cmap='tab20')

    # 标注断点（可选）
    for pt in break_points:
        plt.plot(pt[0], pt[1], marker='+', color='white', markersize=10, markeredgewidth=2)

    plt.title("两层 ReLU 网络的激活分区图 (2-3-3-1)", fontsize=16, color='black')
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14)
    plt.tick_params(colors='black')
    plt.grid(False)
    plt.show()

# 示例：你可以加自己的断点坐标
break_points = [
    (1.1069839389, -0.5962324643),
    (1.1220176966, -0.4987486310),
    (1.1399850472, -0.3822424162),
    (1.3670958350, 1.0904186830),
    (0.7008097302, 1.2085095775),
    (0.8575328898, 0.9062272429),
    (1.1802100225, 0.2838584925),
    (1.5631454471, -0.4547344150),
    (1.6290511621, -0.5818511312),
    (1.1272897151, 1.0857264908),
    (0.8845133133, -0.2749096208),
    (0.8795855049, -0.3025274366),
    (0.8379791235, -0.5357096761),
    (0.7897606915, -0.8059490373),
]
plot_activation_regions_with_breaks(break_points)
