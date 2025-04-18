import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为思源黑体（Noto Sans CJK SC）
rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
rcParams['axes.unicode_minus'] = False  # 正常显示负号


def plot_activation_regions_with_breaks(break_points, grid_size=300):
    import numpy as np
    import matplotlib.pyplot as plt
    

    # 网络参数
    w1 = [0.2, -0.3]; b1 = 0.1
    w2 = [0.4, 0.5];  b2 = -0.2
    w3 = [-0.6, 0.1]; b3 = 0.3

    # 激活模式编码函数：返回一个字符串，例如 "101"
    def activation_pattern(x1, x2):
        z1 = w1[0]*x1 + w1[1]*x2 + b1
        z2 = w2[0]*x1 + w2[1]*x2 + b2
        z3 = w3[0]*x1 + w3[1]*x2 + b3
        return f"{int(z1 > 0)}{int(z2 > 0)}{int(z3 > 0)}"

    # 输入网格
    x_range = np.linspace(-1, 1, grid_size)
    y_range = np.linspace(-1, 1, grid_size)
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
    plt.imshow(int_patterns, extent=[-1, 1, -1, 1], origin='lower', cmap='tab10')

    # 标注断点（break_points）
    #for pt in break_points:
    #    plt.plot(pt[0], pt[1], marker='+', color='white', markersize=10, markeredgewidth=2)

    plt.title("ReLU 激活分区图", fontsize=16, color='black')
    plt.xlabel("$x_1$", fontsize=14, color='black')
    plt.ylabel("$x_2$", fontsize=14, color='black')
    plt.tick_params(colors='black')
    plt.grid(False)
    plt.show()

break_points = [
(-0.3498, 0.1000),
(0.3751, 0.1000),
(0.5172, 0.1000),
]


plot_activation_regions_with_breaks(break_points)
