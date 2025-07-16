import random
import numpy as np
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity


def iterative_first_improvement(current_config, current_score, independent_set, dict_search,
                                history_configs, budget_used, budget, xs, results, best_result, best_config):
    """
    迭代首次改进局部搜索（优化版）
    核心改进：直接传入当前配置得分，减少重复计算；补充邻域评估的过程记录与输出
    对应论文3.1节"one-exchange neighbourhood"和3.2节局部搜索逻辑
    """
    while True:
        improved = False
        # 生成单交换邻域（仅改变一个参数，论文中核心邻域定义）
        neighbors = []
        for i in range(len(current_config)):
            for value in independent_set[i]:
                if value != current_config[i]:
                    neighbor = current_config.copy()
                    neighbor[i] = value
                    neighbors.append(neighbor)

        random.shuffle(neighbors)  # 随机打乱邻域顺序，避免参数偏见

        for neighbor in neighbors:
            config_tuple = tuple(neighbor)
            # 检查是否已评估（论文中未明确提及，但为预算控制必要实现）
            if config_tuple in history_configs:
                score, _ = get_objective_score_with_similarity(dict_search, neighbor)
            else:
                if budget_used >= budget:
                    break  # 预算耗尽，停止搜索
                # 评估新配置（消耗预算）
                score, _ = get_objective_score_with_similarity(dict_search, neighbor)
                budget_used += 1
                history_configs.add(config_tuple)
                # 记录配置与得分（补充过程追踪）
                xs.append(neighbor)
                results.append(score)
                # 输出邻域评估信息（符合用户需求的过程记录）
                print(f"轮次: {budget_used}/{budget} (局部搜索)")
                print(f"配置: {neighbor}")
                print(f"得分: {score}")
                print(f"当前最优得分: {best_result}")
                print(f"当前最优配置: {best_config}\n{'-' * 50}")

            # 改进判断（论文中"first improvement"准则）
            if score < current_score:
                current_config = neighbor
                current_score = score  # 更新当前得分，避免重复计算
                improved = True
                # 更新全局最优
                if score < best_result:
                    best_result = score
                    best_config = neighbor
                break  # 首次改进后重启局部搜索

        if not improved or budget_used >= budget:
            break  # 无改进或预算耗尽时退出

    return current_config, current_score, budget_used, history_configs, xs, results, best_result, best_config


def run_tuners(file, budget=20, seed=0, r=10, s=3, p_restart=0.01):
    """
    ParamILS调优器主函数（优化版）
    对应论文3.1节ParamILS框架与Algorithm 1伪代码
    """
    random.seed(seed)
    np.random.seed(seed)

    # 读取数据集（论文5.2节配置场景数据加载逻辑）
    #file = get_data(filename)
    independent_set = file.independent_set
    dict_search = file.dict_search

    # 初始化变量（论文中"incumbent configuration"等核心变量）
    best_result = float('inf')
    best_config = None
    xs = []  # 所有评估过的配置
    results = []  # 对应得分
    history_configs = set()
    budget_used = 0

    # 1. 初始配置搜索（论文中r个随机初始化配置）
    initial_configs = [
        [random.choice(values) for values in independent_set]
        for _ in range(r)
    ]

    # 评估初始配置（论文中初始化阶段的评估逻辑）
    for config in initial_configs:
        config_tuple = tuple(config)
        if config_tuple in history_configs or budget_used >= budget:
            continue
        # 计算初始得分
        score, _ = get_objective_score_with_similarity(dict_search, config)
        budget_used += 1
        history_configs.add(config_tuple)
        xs.append(config)
        results.append(score)
        # 更新全局最优与输出
        if score < best_result:
            best_result = score
            best_config = config
        print(f"轮次: {budget_used}/{budget} (初始配置)")
        print(f"配置: {config}")
        print(f"得分: {score}")
        print(f"全局最优得分: {best_result}")
        print(f"全局最优配置: {best_config}\n{'-' * 50}")

    # 选择初始最优配置（论文中初始 incumbent 选择）
    if xs:
        initial_best_idx = np.argmin(results)
        current_config = xs[initial_best_idx]
        current_score = results[initial_best_idx]  # 预计算得分，避免重复计算
    else:
        current_config = [random.choice(values) for values in independent_set]
        current_score, _ = get_objective_score_with_similarity(dict_search, current_config)

    # 2. 迭代局部搜索主循环（论文Algorithm 1的核心循环）
    while budget_used < budget:
        # 局部搜索（传入当前得分，减少重复计算）
        current_config, current_score, budget_used, history_configs, xs, results, best_result, best_config = \
            iterative_first_improvement(
                current_config, current_score, independent_set, dict_search,
                history_configs, budget_used, budget, xs, results, best_result, best_config
            )

        if budget_used >= budget:
            break

        # 扰动操作（论文中"s个随机移动"的扰动策略）
        perturbed_config = current_config.copy()
        for _ in range(s):
            idx = random.randint(0, len(perturbed_config) - 1)
            possible_values = [v for v in independent_set[idx] if v != perturbed_config[idx]]
            if possible_values:
                perturbed_config[idx] = random.choice(possible_values)

        # 扰动后局部搜索（论文中"subsidiary local search"步骤）
        perturbed_score, _ = get_objective_score_with_similarity(dict_search, perturbed_config)
        perturbed_config, perturbed_score, budget_used, history_configs, xs, results, best_result, best_config = \
            iterative_first_improvement(
                perturbed_config, perturbed_score, independent_set, dict_search,
                history_configs, budget_used, budget, xs, results, best_result, best_config
            )

        # 接受准则（论文中"accept better or equally-good configurations"）
        if perturbed_score < current_score:
            current_config = perturbed_config
            current_score = perturbed_score

        # 随机重启（论文中"prestart"概率重启机制）
        if random.random() < p_restart:
            current_config = [random.choice(values) for values in independent_set]
            current_score, _ = get_objective_score_with_similarity(dict_search, current_config)
            config_tuple = tuple(current_config)
            if config_tuple not in history_configs and budget_used < budget:
                budget_used += 1
                history_configs.add(config_tuple)
                xs.append(current_config)
                results.append(current_score)
                if current_score < best_result:
                    best_result = current_score
                    best_config = current_config
                print(f"轮次: {budget_used}/{budget} (随机重启)")
                print(f"配置: {current_config}")
                print(f"得分: {current_score}")
                print(f"全局最优得分: {best_result}")
                print(f"全局最优配置: {best_config}\n{'-' * 50}")

    final_round = len(xs)
    return xs, results, range(1, final_round + 1), best_result, np.argmin(results) + 1, final_round