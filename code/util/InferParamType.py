import numpy as np


def _infer_param_type(param_values):
    """
    根据参数可能的取值推断其类型
    支持的类型包括：
    - categorical: 分类变量（字符串或无明显顺序的数值）
    - ordinal: 顺序变量（如"low", "medium", "high"这样有顺序关系的字符串）
    - continuous: 连续数值变量
    - discrete: 离散数值变量（通常为少量整数值）

    参数:
    param_values: 参数的可能取值列表

    返回:
    推断的参数类型字符串
    """
    # 处理空值情况，默认视为分类变量
    if not param_values:
        return 'categorical'

    # 检查是否包含字符串类型的值
    # 若包含字符串，则进一步判断是否为有顺序关系的字符串
    has_string = any(isinstance(v, str) for v in param_values)
    if has_string:
        # 预定义常见的顺序型字符串集合
        ordered_sets = [
            {'low', 'medium', 'high'},  # 低中高
            {'small', 'large'}  # 小大
        ]
        param_set = set(param_values)

        # 检查参数值是否是某个预定义顺序集合的子集
        for ordered in ordered_sets:
            if param_set.issubset(ordered):
                return 'ordinal'  # 属于预定义顺序集合，则视为顺序变量

        # 否则视为普通分类变量
        return 'categorical'

    # 纯数值类型的参数判断
    all_numeric = all(isinstance(v, (int, float, np.number)) for v in param_values)
    if not all_numeric:
        return 'categorical'

    # 对纯数值类型进行进一步细分
    unique_values = list(set(param_values))  # 去重后的取值
    num_unique = len(unique_values)  # 唯一值数量
    num_total = len(param_values)  # 总样本量
    unique_ratio = num_unique / num_total  # 唯一值比例
    std_dev = np.std(param_values) if num_total > 1 else 0  # 标准差

    # 根据唯一值比例和标准差判断是否为连续变量
    if unique_ratio > 0.1 and std_dev > 1:
        return 'continuous'
    # 根据唯一值数量判断是否为离散变量
    elif num_unique <= 10:
        return 'discrete'
    # 其他情况视为分类变量
    else:
        return 'categorical'