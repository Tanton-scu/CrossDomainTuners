import numpy as np
from sklearn.preprocessing import MinMaxScaler
from InferParamType import _infer_param_type


def get_objective_score_with_similarity(dict_search, config, param_types=None):
    """
    查找与给定配置最相似的已知配置并返回其目标值
    根据参数类型（连续/离散/布尔/分类等）采用差异化的相似度计算策略
    """
    # 尝试直接查找配置
    config_tuple = tuple(config)
    if config_tuple in dict_search:
        return dict_search[config_tuple], list(config)

    # 自动推断参数类型（如果未提供）
    if param_types is None:
        param_types = [_infer_param_type([config[i] for config in dict_search.keys()])
                      for i in range(len(next(iter(dict_search.keys()))))]

    # 分离不同类型参数的索引
    continuous_indices = [i for i, t in enumerate(param_types) if t in ['continuous', 'discrete']]
    categorical_indices = [i for i, t in enumerate(param_types) if t in ['categorical', 'ordinal']]
    boolean_indices = [i for i, t in enumerate(param_types) if t == 'boolean']

    # 计算所有配置与查询配置的相似度
    best_distance = float('inf')
    best_config = None
    best_value = None

    # 预处理连续型参数（标准化）
    scaled_continuous = None
    scaler = None
    if continuous_indices:
        continuous_configs = [
            [config[i] for i in continuous_indices]
            for config in dict_search.keys()
        ]
        scaler = MinMaxScaler()
        scaled_continuous = scaler.fit_transform(continuous_configs)
        query_continuous = [config[i] for i in continuous_indices]
        query_scaled = scaler.transform([query_continuous])[0]

    # 遍历所有已知配置计算距离
    for idx, key in enumerate(dict_search.keys()):
        total_distance = 0

        # 1. 连续/离散型参数：标准化欧氏距离
        if continuous_indices:
            num_config = scaled_continuous[idx]
            numerical_distance = np.linalg.norm(num_config - query_scaled)
            total_distance += numerical_distance

        # 2. 布尔型参数：不匹配数量（权重更高）
        if boolean_indices:
            bool_distance = sum(
                1 for i in boolean_indices if key[i] != config[i]
            )
            total_distance += bool_distance * 2  # 布尔不匹配影响更大，权重×2

        # 3. 分类/顺序型参数：不匹配数量 + 顺序差异（如有）
        if categorical_indices:
            cat_distance = 0
            for i in categorical_indices:
                if key[i] != config[i]:
                    # 顺序型参数（如low/medium/high）的差异加权
                    if param_types[i] == 'ordinal':
                        # 定义顺序权重（可根据实际场景调整）
                        order_map = {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}
                        if key[i] in order_map and config[i] in order_map:
                            cat_distance += abs(order_map[key[i]] - order_map[config[i]])
                        else:
                            cat_distance += 1  # 非预定义顺序视为普通不匹配
                    else:
                        cat_distance += 1  # 普通分类参数不匹配计数
            total_distance += cat_distance

        # 更新最优配置
        if total_distance < best_distance:
            best_distance = total_distance
            best_config = key
            best_value = dict_search[key]

    return best_value, list(best_config)