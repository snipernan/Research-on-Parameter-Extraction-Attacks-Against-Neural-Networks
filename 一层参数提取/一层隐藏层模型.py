def relu(z):
    return max(0,z)

def simple_dnn(x,y):
    #隐藏层神经元
    z1=w1[0]*x + w1[1]*y + b1
    print(z1)
    z2=w2[0]*x + w2[1]*y + b2
    print(z2)
    z3=w3[0]*x + w3[1]*y + b3
    print(z3)

    a1 = relu(z1)
    a2 = relu(z2)
    a3 = relu(z3)     

    #输出层
    output = wo[0]*a1+wo[1]*a2+wo[2]*a3+bo
    return output

# 参数定义（可以放外面，也可以写进函数里）
w1 = [0.2, -0.3]; b1 = 0.1
w2 = [0.4, 0.5];  b2 = -0.2
w3 = [-0.6, 0.1]; b3 = 0.3
wo = [0.7, -0.5, 0.2]; bo = 0.1

output = simple_dnn(-0.3507832611,0.1)
print("预测结果:", output)
