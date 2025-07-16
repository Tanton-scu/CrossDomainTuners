import random
import math
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity


def run_tuners(file ,  budget=200, seed=0):  # 修改默认预算为200
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

    # 初始化当前配置
    current_config = []
    for values in independent_set:
        current_config.append(random.choice(values))

    # 评估当前配置
    current_score, _ = get_objective_score_with_similarity(dict_search, current_config)

    # 模拟退火参数
    initial_temperature = 200  # 提高初始温度
    final_temperature = 0.01  # 降低终止温度
    # 注意：不再使用固定冷却率，改用动态冷却策略

    # 新增：记录历史配置（用于去重）
    history_configs = set()
    history_configs.add(tuple(current_config))

    step = 0  # 有效步骤计数（仅统计新配置）
    loop = 0  # 总循环次数
    repeat_count = 0  # 重复配置计数
    max_repeats = 100 * budget  # 最大重复次数阈值

    while step < budget:  # 仅受预算限制，不考虑温度下限
        # 检查是否达到最大重复次数
        if repeat_count >= max_repeats:
            print(f"警告: 已生成 {repeat_count} 次重复配置，达到阈值 {max_repeats}，提前终止搜索")
            break

        # 动态计算当前温度（基于迭代进度）
        temperature = initial_temperature * (1 - (step / budget)) ** 2

        # 生成邻域解
        neighbor_config = current_config.copy()
        index = random.randint(0, len(neighbor_config) - 1)
        neighbor_config[index] = random.choice(independent_set[index])
        neighbor_config_tuple = tuple(neighbor_config)

        # 检查是否为重复配置
        if neighbor_config_tuple in history_configs:
            print(f"步骤 {loop + 1} 生成重复配置 {neighbor_config_tuple}，重新生成...")
            loop += 1
            repeat_count += 1  # 增加重复计数
            continue  # 重复配置不消耗预算，重新生成

        # 重置重复计数（找到新配置）
        repeat_count = 0

        # 新配置：加入历史记录
        history_configs.add(neighbor_config_tuple)

        # 评估邻域解
        neighbor_score, _ = get_objective_score_with_similarity(dict_search, neighbor_config)

        # 判断是否接受邻域解
        if neighbor_score < current_score or random.random() < math.exp((current_score - neighbor_score) / temperature):
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

        # 输出每一轮的详细信息（可根据需要调整输出频率）
        if (step + 1) % 10 == 0:
            print(f"有效步骤: {step + 1}/{budget}")
            print(f"当前温度: {temperature:.4f}")
            print(f"当前配置: {current_config}")
            print(f"当前得分: {current_score}")
            print(f"当前最优得分: {best_result}")
            print(f"当前最优配置: {best_config}")
            print("-" * 50)

        step += 1
        loop += 1

    # 返回结果
    return xs, results, range(1, len(results) + 1), best_result, best_loop, step