import random
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity

def run_tuners(file,  budget=20, seed=0):
    # 设置随机种子
    random.seed(seed)
    # 读取数据集
    #file = get_data(filename)
    # 自变量的取值范围
    independent_set = file.independent_set
    # 决策与目标值的映射字典
    dict_search = file.dict_search

    # 初始化最优结果
    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []

    # 记录历史配置（用于去重）
    history_configs = set()  # 存储已生成的配置（元组形式，可哈希）

    # 初始化当前配置
    current_config = []
    for values in independent_set:
        current_config.append(random.choice(values))
    config_tuple = tuple(current_config)
    history_configs.add(config_tuple)

    # 评估当前配置
    current_score, _ = get_objective_score_with_similarity(dict_search, current_config)

    step = 0  # 有效步骤计数（仅统计新配置）
    loop = 0  # 总循环次数
    repeat_count = 0  # 重复配置计数
    max_repeats = 100 * budget  # 最大重复次数阈值

    while step < budget:
        # 检查是否达到最大重复次数
        if repeat_count >= max_repeats:
            print(f"警告: 已生成 {repeat_count} 次重复配置，达到阈值 {max_repeats}，提前终止搜索")
            break

        # 生成邻域解
        neighbor_config = current_config.copy()
        index = random.randint(0, len(neighbor_config) - 1)
        neighbor_config[index] = random.choice(independent_set[index])
        neighbor_config_values = tuple(neighbor_config)

        # 检查是否为重复配置
        if neighbor_config_values in history_configs:
            print(f"步骤 {loop + 1} 生成重复配置 {neighbor_config_values}，重新生成...")
            loop += 1
            repeat_count += 1  # 增加重复计数
            continue  # 重复配置不消耗预算，重新生成

        # 重置重复计数（找到新配置）
        repeat_count = 0

        # 新配置：加入历史记录
        history_configs.add(neighbor_config_values)

        # 评估邻域解
        neighbor_score, _ = get_objective_score_with_similarity(dict_search, neighbor_config)

        # 如果邻域解更优，则更新当前配置
        if neighbor_score < current_score:
            current_config = neighbor_config
            current_score = neighbor_score

        # 更新最优结果
        if current_score < best_result:
            best_result = current_score
            best_config = current_config
            best_loop = step + 1

        # 记录配置和性能结果
        xs.append(current_config)
        results.append(current_score)

        # 输出每一轮的详细信息
        print(f"有效步骤: {step + 1}/{budget}")
        print(f"当前配置: {current_config}")
        print(f"当前得分: {current_score}")
        print(f"当前最优得分: {best_result}")
        print(f"当前最优配置: {best_config}")
        print("-" * 50)

        step += 1
        loop += 1

    # 返回结果
    print(step)
    return xs, results, range(1, len(results) + 1), best_result, best_loop, step