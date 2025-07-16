import random
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity


def run_tuners(file,   budget=20, seed=0, tabu_tenure=5):
    # 设置随机种子，确保结果可复现
    random.seed(seed)

    # 自变量的取值范围
    independent_set = file.independent_set
    print(f"搜索空间维度: {len(independent_set)}")

    # 估算邻域大小上限
    max_neighbors = sum(len(values) - 1 for values in independent_set)
    print(f"最大可能邻域大小: {max_neighbors}")

    # 决策与目标值的映射字典
    dict_search = file.dict_search

    # 初始化最优结果
    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []

    # 初始化禁忌表和历史配置
    tabu_list = []
    history_configs = set()

    # 随机生成初始配置
    current_config = [random.choice(values) for values in independent_set]
    config_tuple = tuple(current_config)
    history_configs.add(config_tuple)
    current_score, _ = get_objective_score_with_similarity(dict_search, current_config)

    # 记录初始配置和性能结果
    xs.append(current_config)
    results.append(current_score)
    best_result, best_config, best_loop = current_score, current_config, 1

    # 输出初始信息
    print(f"有效步骤: 1/{budget} | 当前得分: {current_score} | 最优得分: {best_result}")
    print("-" * 50)

    step = 1  # 有效步骤计数
    stagnation_count = 0  # 连续未改进的步数
    total_evaluations = 1  # 总评估次数（包括重复配置）

    # 新增：无新配置计数器
    no_new_config_count = 0
    max_no_new_config = budget * 100  # 最大无新配置步数（可调整为预算的2倍）

    while step < budget:
        # 检查是否达到无新配置阈值
        if no_new_config_count >= max_no_new_config:
            print(f"警告: 已连续 {no_new_config_count} 步未发现新配置，达到阈值 {max_no_new_config}，提前终止搜索")
            break

        # 生成邻域解
        neighbors = []
        for i in range(len(current_config)):
            values = independent_set[i]
            for value in values:
                if value != current_config[i]:
                    neighbor = current_config.copy()
                    neighbor[i] = value
                    neighbors.append(neighbor)

        print(f"步骤 {step}: 生成了 {len(neighbors)} 个邻居解")

        # 筛选非禁忌解
        non_tabu_neighbors = []
        for neighbor in neighbors:
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple not in tabu_list:
                non_tabu_neighbors.append(neighbor)

        print(f"步骤 {step}: 有 {len(non_tabu_neighbors)} 个非禁忌解")

        if not non_tabu_neighbors:
            # 如果没有非禁忌解，解禁最早的解
            if tabu_list:
                tabu_list.pop(0)
                print(f"步骤 {step}: 禁忌表已满，解禁最早的解")
            else:
                print(f"步骤 {step}: 警告: 没有非禁忌解且禁忌表为空！")
            no_new_config_count += 1  # 无新配置计数+1
            continue

        # 评估非禁忌解
        best_neighbor = None
        best_neighbor_score = float('inf')
        new_config_found = False  # 是否找到新配置
        evaluated_count = 0
        reused_count = 0

        for neighbor in non_tabu_neighbors:
            neighbor_tuple = tuple(neighbor)
            total_evaluations += 1

            if neighbor_tuple in history_configs:
                # 重复配置不消耗预算
                score, _ = get_objective_score_with_similarity(dict_search, neighbor)
                reused_count += 1
            else:
                # 新配置消耗预算
                score, _ = get_objective_score_with_similarity(dict_search, neighbor)
                history_configs.add(neighbor_tuple)
                step += 1
                evaluated_count += 1
                new_config_found = True  # 关键：找到新配置
                no_new_config_count = 0  # 重置无新配置计数

                # 记录配置和性能结果
                xs.append(neighbor)
                results.append(score)

                # 更新最优结果
                if score < best_result:
                    best_result = score
                    best_config = neighbor
                    best_loop = step
                    stagnation_count = 0
                    print(f"步骤 {step}: 找到新的全局最优解! 得分: {best_result}")
                else:
                    stagnation_count += 1

                # 输出本轮信息（每10步或找到更好解时）
                if score < best_result or step % 10 == 0:
                    print(f"有效步骤: {step}/{budget}")
                    print(f"本轮配置: {neighbor}")
                    print(f"本轮得分: {score}")
                    print(f"当前最优得分: {best_result}")
                    print(f"当前最优配置: {best_config}")
                    print(f"禁忌表大小: {len(tabu_list)}/{tabu_tenure}")
                    print(f"历史配置数: {len(history_configs)}")
                    print(f"停滞步数: {stagnation_count}")
                    print(f"总评估次数: {total_evaluations} (新: {evaluated_count}, 重复: {reused_count})")
                    print(f"连续无新配置步数: {no_new_config_count}")
                    print("-" * 50)

                if step >= budget:
                    break

            if score < best_neighbor_score:
                best_neighbor_score = score
                best_neighbor = neighbor

        # 如果没有找到新配置，增加无新配置计数
        if not new_config_found:
            no_new_config_count += len(non_tabu_neighbors)  # 所有邻居都是重复的
            print(f"步骤 {step}: 无新配置，连续无新配置计数增加到 {no_new_config_count}")

        # 检查是否有改进
        if best_neighbor_score < current_score:
            print(f"步骤 {step}: 选择了改进解，得分从 {current_score} 提升到 {best_neighbor_score}")
        else:
            print(f"步骤 {step}: 选择了非改进解，得分 {best_neighbor_score} (当前: {current_score})")

        if step >= budget:
            break

        # 更新当前配置
        current_config = best_neighbor
        current_score = best_neighbor_score

        # 将当前配置加入禁忌表
        current_config_tuple = tuple(current_config)
        tabu_list.append(current_config_tuple)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # 检查停滞情况
        if stagnation_count >= 20:
            print(f"警告: 算法已经停滞 {stagnation_count} 步，可能陷入局部最优")
            print(f"当前最优得分: {best_result}")
            print(f"当前配置: {current_config}")
            print(f"剩余预算: {budget - step}")
            print(f"连续无新配置步数: {no_new_config_count}")
            print("-" * 50)

    # 返回结果
    print(f"优化完成! 总有效步骤: {step}/{budget}")
    print(f"总评估次数: {total_evaluations}")
    print(f"找到的最优解得分: {best_result}")
    print(f"最优配置: {best_config}")
    return xs, results, range(1, len(results) + 1), best_result, best_loop, step