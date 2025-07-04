import numpy as np

# 在 util.InferParamType.py 中完善 _infer_param_type 函数
def _infer_param_type(param_values):
    if not param_values:
        return 'categorical'  # 空值视为枚举型

    # 检查是否包含字符串（即使混合了数值和字符串，也视为枚举型）
    has_string = any(isinstance(v, str) for v in param_values)
    if has_string:
        # 进一步判断是否为顺序型字符串（如 "low"/"medium"/"high"）
        ordered_sets = [{'low', 'medium', 'high'}, {'small', 'large'}]
        param_set = set(param_values)
        for ordered in ordered_sets:
            if param_set.issubset(ordered):
                return 'ordinal'  # 顺序型枚举
        return 'categorical'  # 普通字符串枚举

    # 纯数值类型的判断（原逻辑保留）
    all_numeric = all(isinstance(v, (int, float, np.number)) for v in param_values)
    if not all_numeric:
        return 'categorical'

    # 数值型参数的细分（连续/离散）
    unique_values = list(set(param_values))
    num_unique = len(unique_values)
    num_total = len(param_values)
    unique_ratio = num_unique / num_total
    std_dev = np.std(param_values) if num_total > 1 else 0

    if unique_ratio > 0.1 and std_dev > 1:
        return 'continuous'
    elif num_unique <= 5:
        return 'discrete'
    else:
        return 'categorical'