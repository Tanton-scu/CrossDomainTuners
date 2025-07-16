import random
import numpy as np
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity


def run_tuners(file ,  budget=20, seed=0):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 读取数据集
    #file = get_data(filename)
    independent_set = file.independent_set
    dict_search = file.dict_search

    # 初始化最优结果
    best_result = float('inf')
    best_config = None
    best_loop = 0  # 记录最优结果的实际步骤
    xs = []
    results = []

    # 用字典存储历史配置及对应的分数
    history_configs = {}  # 格式：{config_tuple: score, ...}
    # 记录score的最大/最小值（用于归一化）
    score_min = float('inf')
    score_max = -float('inf')

    # 遗传算法参数
    population_size = 10  # 种群大小
    #generations = budget // population_size  # 迭代代数
    generations=budget#给重复配置留出容错
    mutation_rate = 0.3  # 变异率

    # 初始化种群
    population = []
    for _ in range(population_size):
        config = [random.choice(values) for values in independent_set]
        population.append(config)

    # 主循环：迭代多代进行进化
    global_step = 0  # 全局有效步骤计数（仅统计新配置）
    for generation in range(generations):
        print(f"\n=== 开始第 {generation + 1} 代 ===")

        # 评估种群中每个个体的适应度
        fitness_scores = []
        for config in population:
            config_tuple = tuple(config)

            # 检查是否为重复配置
            if config_tuple in history_configs:
                # 重复配置：直接使用历史分数
                score = history_configs[config_tuple]
                fitness_scores.append(score)
                print(f"发现重复配置 {config_tuple}，使用历史分数 {score}")
                continue  # 不消耗预算，继续评估下一个配置

            # 新配置：计算分数并加入历史记录
            score, _ = get_objective_score_with_similarity(dict_search, config)
            history_configs[config_tuple] = score
            fitness_scores.append(score)

            # 更新全局步骤计数
            global_step += 1

            # 动态更新score的最大/最小值
            if score < score_min:
                score_min = score
            if score > score_max:
                score_max = score

            # 奖励归一化
            if score_max == score_min:
                reward = 0.5
            else:
                reward = (score_max - score) / (score_max - score_min)

            # 当前轮次（全局有效步骤）
            current_loop = global_step

            # 更新最优结果
            if score < best_result:
                best_result = score
                best_config = config
                best_loop = current_loop
                print(f"新最优结果！步骤: {best_loop}, 得分: {best_result}, 配置: {best_config}")

            # 收集所有评估结果
            xs.append(config)
            results.append(score)

            # 打印每轮详细信息
            print(f"有效步骤: {global_step}/{budget}")
            print(f"当前配置: {config}")
            print(f"当前得分: {score}")
            print(f"当前归一化奖励: {reward}")
            print(f"当前最优得分: {best_result}")
            print(f"当前最优配置: {best_config}")
            print("-" * 50)

            # 检查是否达到预算上限
            if global_step >= budget:
                break

        # 如果达到预算上限，立即终止所有循环
        if global_step >= budget:
            break

        # 选择操作：轮盘赌选择（此时fitness_scores长度必为population_size）
        # 1. 计算适应度（处理可能的0值和负数值）
        fitness_values = []
        for score in fitness_scores:
            adjusted_score = max(score, 1e-10)  # 避免除零错误
            fitness_values.append(1 / adjusted_score)  # 分数越小，适应度越高

        # 2. 计算总适应度并归一化
        total_fitness = sum(fitness_values)
        if total_fitness <= 0:
            selection_probs = [1.0 / population_size for _ in range(population_size)]
        else:
            selection_probs = [f / total_fitness for f in fitness_values]

        # 3. 强制保证概率和为1（解决浮点数精度问题）
        epsilon = 1e-10
        prob_sum = sum(selection_probs)
        if abs(prob_sum - 1.0) > epsilon:
            selection_probs[-1] = 1.0 - sum(selection_probs[:-1])

        # 4. 执行选择（此时a和p长度均为population_size）
        selected_indices = np.random.choice(
            range(population_size),
            size=population_size,
            p=selection_probs
        )
        new_population = [population[i] for i in selected_indices]

        # 交叉操作
        next_generation = []
        for i in range(0, population_size, 2):
            if i + 1 >= population_size:  # 处理种群大小为奇数的情况
                next_generation.append(new_population[i])
                break
            parent1 = new_population[i]
            parent2 = new_population[i + 1]
            crossover_point = random.randint(1, len(independent_set) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            next_generation.extend([child1, child2])

        # 变异操作
        for config in next_generation:
            for i in range(len(config)):
                if random.random() < mutation_rate:
                    config[i] = random.choice(independent_set[i])

        # 更新种群为下一代
        population = next_generation

    return xs, results, range(1, len(results) + 1), best_result, best_loop, budget