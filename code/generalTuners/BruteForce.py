import itertools
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity


def run_tuners(file,  budget=20, seed=0):
    # 设置随机种子（保持接口一致性）
    # 读取数据集
    #file = get_data(filename)
    independent_set = file.independent_set  # 自变量的取值范围
    dict_search = file.dict_search  # 决策与目标值的映射字典

    # 初始化最优结果
    best_result = float('inf')
    best_config = None
    best_step = 0
    xs = []
    results = []

    # 新增：记录历史配置（用于去重）
    history_configs = set()  # 存储已评估的配置（元组形式，可哈希）

    # 生成所有可能的配置组合
    all_configs = itertools.product(*independent_set)

    # 开始暴力搜索（确保预算仅消耗在全新配置上）
    step = 0  # 有效步骤计数（仅统计新配置）
    for config in all_configs:
        if step >= budget:
            break  # 达到预算后停止

        current_config = list(config)
        config_tuple = tuple(current_config)  # 转换为元组用于去重判断

        # 检查是否为重复配置
        if config_tuple in history_configs:
            continue  # 重复配置不消耗预算，跳过

        # 新配置：评估并记录
        history_configs.add(config_tuple)
        score, _ = get_objective_score_with_similarity(dict_search, current_config)

        # 记录配置和性能结果
        xs.append(current_config)
        results.append(score)

        # 更新最优结果
        if score < best_result:
            best_result = score
            best_config = current_config
            best_step = step + 1  # 记录最优结果出现的有效步骤

        # 输出每一步的详细信息
        print(f"有效步骤: {step + 1}/{budget}")
        print(f"当前配置: {current_config}")
        print(f"当前得分: {score}")
        print(f"当前最优得分: {best_result}")
        print(f"当前最优配置: {best_config}")
        print("-" * 50)

        step += 1  # 仅在处理新配置后递增有效步骤计数

    # 返回结果（确保返回的轮次范围与有效步骤一致）
    return xs, results, range(1, len(results) + 1), best_result, best_step, step