import random
import numpy as np
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity

def run_tuners(file,   budget=20, seed=0):
    random.seed(seed)
    independent_set = file.independent_set
    dict_search = file.dict_search

    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []

    generated_configs = set()
    pop_size = 10
    F = 0.5
    CR = 0.7

    # 新增：重复配置计数器
    repeat_count = 0
    max_repeats = 100 * budget  # 最大重复次数阈值

    # 初始化种群
    population = []
    init_scores = []
    for _ in range(pop_size):
        individual = [random.choice(values) for values in independent_set]
        score, _ = get_objective_score_with_similarity(dict_search, individual)
        population.append(individual)
        init_scores.append(score)
        xs.append(individual)
        results.append(score)
        generated_configs.add(tuple(individual))

        if score < best_result:
            best_result = score
            best_config = individual
            best_loop = len(xs)

    remaining_budget = budget - pop_size
    if remaining_budget <= 0:
        return xs[:budget], results[:budget], range(1, budget + 1), best_result, best_loop, budget

    loop = 0
    while remaining_budget > 0:
        # 检查重复配置阈值
        if repeat_count >= max_repeats:
            print(f"警告: 已生成 {repeat_count} 次重复配置，达到阈值 {max_repeats}，提前终止搜索")
            break

        for i in range(pop_size):
            if remaining_budget <= 0:
                break

            indices = [j for j in range(pop_size) if j != i]
            a, b, c = random.sample(indices, 3)

            # 变异操作
            mutant = []
            for j in range(len(population[i])):
                val_a, val_b, val_c = population[a][j], population[b][j], population[c][j]
                if isinstance(val_a, (str, bool)):
                    possible = [v for v in independent_set[j] if v != population[i][j]]
                    v = random.choice(possible) if possible else population[i][j]
                else:
                    v = val_a + F * (val_b - val_c)
                    if v not in independent_set[j]:
                        v = random.choice(independent_set[j])
                mutant.append(v)

            # 交叉操作
            trial = []
            for j in range(len(population[i])):
                if random.random() < CR or j == random.randint(0, len(population[i]) - 1):
                    trial.append(mutant[j])
                else:
                    trial.append(population[i][j])

            config_tuple = tuple(trial)
            if config_tuple in generated_configs:
                # 配置已存在，不消耗预算
                repeat_count += 1  # 增加重复计数
                print(f"检测到重复配置，当前重复计数: {repeat_count}/{max_repeats}")
                continue  # 跳过后续处理，不消耗预算
            else:
                # 评估新个体（消耗1次预算）
                trial_score, _ = get_objective_score_with_similarity(dict_search, trial)
                remaining_budget -= 1
                generated_configs.add(config_tuple)
                repeat_count = 0  # 找到新配置时重置计数器

            xs.append(trial)
            results.append(trial_score)

            # 选择操作
            current_score = init_scores[i]
            if trial_score < current_score:
                population[i] = trial
                init_scores[i] = trial_score

            # 更新全局最优
            if trial_score < best_result:
                best_result = trial_score
                best_config = trial
                best_loop = len(xs)

            # 打印信息
            print(f"评估次数: {len(xs)}, 重复计数: {repeat_count}/{max_repeats}")
            print(f"当前配置: {trial}")
            print(f"当前得分: {trial_score}")
            print(f"当前最优得分: {best_result}")
            print(f"当前最优配置: {best_config}")
            print("-" * 50)

        loop += 1

    return xs[:budget], results[:budget], range(1, budget + 1), best_result, best_loop, len(xs)