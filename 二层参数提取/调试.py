import numpy as np
import re

def parse_witness_text(text):
    lines = text.strip().split('\n')
    result = []

    for line in lines:
        # 提取括号内的数字
        match = re.search(r'\[([^\]]+)\]', line)
        if match:
            numbers_str = match.group(1)
            numbers = list(map(float, numbers_str.strip().split()))
            result.append(numbers)

    return result

text = """
    witness 1: [-0.29889871  1.14713093]
    witness 2: [-2.00917006  2.68735171]
    witness 3: [-0.66552667  1.47482011]
    witness 4: [ 5.94511008 -4.46639887]
    witness 5: [ 2.68754413 -1.53903861]
    witness 6: [ 2.79308235 -1.6315911 ]
    witness 7: [-2.38311671  3.02271365]
    witness 8: [ 1.97397633 -0.89638994]
    witness 9: [ 1.3935114  -0.37337289]
    witness 10: [ 1.85996533 -0.79279534]
    witness 11: [-2.41896425  3.0533838 ]
    witness 12: [-4.35365104  4.79320439]
    witness 13: [0.90981384 0.0606299 ]
    witness 14: [-0.34452741  1.1881614 ]
    witness 15: [0.11298767 0.77651784]
    witness 16: [0.87822666 0.08945013]
    witness 17: [ 6.72662249 -5.16817686]
    witness 18: [-8.11199865  8.17194329]
    witness 19: [-0.85642798  1.64831972]
    witness 20: [ 2.41055167 -1.29007678]
    witness 21: [ 1.57428153 -0.53595355]
    witness 22: [ 4.10310823 -2.81024817]
    witness 23: [-5.80969124  6.10204839]
    witness 24: [ 6.87864933 -5.30583459]
    witness 25: [ 2.55613175 -1.41935037]
    witness 26: [-1.34254548  2.08603625]
    witness 27: [-6.17661284  6.43281996]
    witness 28: [-2.69152266  3.29853894]
    witness 29: [-1.81754832  2.51263023]
    witness 30: [-5.46261727  5.79064008]
    witness 31: [ 1.23950916 -0.23383636]
    witness 32: [0.48890747 0.43926146]
    witness 33: [-0.6006919   1.41894239]
    witness 34: [-0.76955444  1.57263301]
    witness 35: [-2.76731118  3.36654196]
    witness 36: [-0.44133395  1.27359152]
    witness 37: [0.50194477 0.4268731 ]
    witness 38: [ 1.00473956 -0.02392141]
    witness 39: [ 7.56250486 -5.92008411]
    witness 40: [0.92896879 0.04317233]
    witness 41: [-1.53307102  2.25514098]
    witness 42: [ 1.94725479 -0.87162857]
    witness 43: [-4.39557898  4.83083991]
    witness 44: [-4.38583635  4.82243647]
    witness 45: [-2.54230221  3.16245367]
"""

witnesses = parse_witness_text(text)
# 固定方向向量
direction_2 = np.array([0.66860969799, 0.74361352311])



# 计算并输出每个点的点积结果
for i, w in enumerate(witnesses, start=1):
    dot_product = np.dot(w, direction_2)
    print(f"Witness {i}: dot = {dot_product:.6f}")
