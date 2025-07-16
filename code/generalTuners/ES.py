import random
import numpy as np
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity
import time
from threading import Thread
from queue import Queue, Empty


def run_tuners(file ,  budget=20, seed=0):
    """
    该函数使用进化策略（ES）算法对给定数据集进行调优，寻找最优配置。

    参数:
    filename (str): 数据集文件的名称，用于读取调优所需的数据。
    budget (int, 可选): 调优过程的最大轮数，默认为 20。
    seed (int, 可选): 随机数生成器的种子，用于确保结果的可重复性，默认为 0。

    返回:
    tuple: 包含以下元素的元组
        - xs (list): 所有评估过的配置列表。
        - results (list): 对应配置的性能结果列表。
        - range(1, len(results) + 1): 轮次范围。
        - best_result (float): 调优过程中找到的最优目标值。
        - best_config (list): 达到最优目标值的配置。
        - best_loop (int): 达到最优目标值的轮次。
        - budget (int): 调优的总轮数。
    """
    print(f"\n=== 开始运行 tuner: 预算={budget}, 种子={seed} ===")

    # 设置随机种子，确保结果的可重复性
    random.seed(seed)
    np.random.seed(seed)

    # 读取数据集
    #print(f"读取数据集: {filename}")
    #file = get_data(filename)
    print(f"数据集读取完成，自变量维度: {len(file.independent_set)}")

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
    evaluated_configs = set()

    # 记录配置生成尝试次数
    config_generation_attempts = 0
    max_attempts_per_eval = 100  # 每个评估最多尝试生成配置的次数

    # 初始化种群
    total_evaluated = 0
    population_size = 10
    population = []

    print(f"初始化种群 (大小={population_size})...")
    for _ in range(population_size):
        individual = [random.choice(values) for values in independent_set]
        population.append(individual)
    print(f"种群初始化完成")

    # 初始化自适应步长
    step_size = 1.0
    step_decay = 0.95

    # 记录运行时间
    start_time = time.time()

    # 目标函数调用超时设置（秒）
    objective_timeout = 300  # 5分钟超时

    while total_evaluated < budget:
        elapsed_time = time.time() - start_time
        print(f"\n=== 总评估次数: {total_evaluated}/{budget}, 已运行时间: {elapsed_time:.2f}s ===")

        # 计算本轮可评估的最大个体数
        remaining = budget - total_evaluated
        current_eval_count = min(population_size, remaining)
        current_eval_count = min(current_eval_count, len(population))

        print(f"本轮计划评估 {current_eval_count} 个个体")
        print(f"当前种群大小: {len(population)}")
        print(f"当前步长: {step_size:.4f}")
        print(f"已评估配置数: {len(evaluated_configs)}")

        # 仅评估前 current_eval_count 个个体
        fitness_scores = []
        eval_success_count = 0

        for i in range(current_eval_count):
            individual = population[i]
            individual_tuple = tuple(individual)

            # 检查是否已评估
            if individual_tuple in evaluated_configs:
                print(f"配置 {i + 1}/{current_eval_count} 已评估，跳过: {individual_tuple}")
                continue

            # 尝试生成有效配置（避免无限循环）
            config_generation_attempts = 0
            valid_config = False

            while not valid_config and config_generation_attempts < max_attempts_per_eval:
                config_generation_attempts += 1

                # 检查配置是否有效（例如是否所有值都在范围内）
                valid = True
                for j, value in enumerate(individual):
                    value_range = independent_set[j]
                    if isinstance(value, (int, float)):
                        if value < min(value_range) or value > max(value_range):
                            valid = False
                            break
                    else:
                        if value not in value_range:
                            valid = False
                            break

                if valid:
                    valid_config = True
                else:
                    # 配置无效，尝试修复或重新生成
                    print(f"警告: 检测到无效配置 {individual}，尝试修复...")
                    for j, value in enumerate(individual):
                        value_range = independent_set[j]
                        if isinstance(value, (int, float)):
                            if value < min(value_range):
                                individual[j] = min(value_range)
                            elif value > max(value_range):
                                individual[j] = max(value_range)
                        else:
                            if value not in value_range:
                                individual[j] = random.choice(value_range)

                    # 再次检查
                    individual_tuple = tuple(individual)
                    if individual_tuple not in evaluated_configs:
                        valid_config = True

            if not valid_config:
                print(f"错误: 无法生成有效配置，已尝试 {config_generation_attempts} 次")
                continue

            # 使用队列和线程实现超时控制
            q = Queue()

            def worker():
                try:
                    score, _ = get_objective_score_with_similarity(dict_search, individual)
                    q.put(score)
                except Exception as e:
                    print(f"错误: 目标函数调用异常: {e}")
                    q.put(float('inf'))

            t = Thread(target=worker)
            t.daemon = True
            t.start()

            try:
                print(f"评估配置 {i + 1}/{current_eval_count}: {individual} (尝试 {config_generation_attempts} 次)")
                start_time_obj = time.time()
                score = q.get(timeout=objective_timeout)
                elapsed_obj = time.time() - start_time_obj

                if elapsed_obj > 10:  # 长时间运行警告
                    print(f"警告: 目标函数调用耗时较长 ({elapsed_obj:.2f}s)")

                fitness_scores.append(score)
                evaluated_configs.add(individual_tuple)
                eval_success_count += 1

                # 更新最优结果和记录
                if score < best_result:
                    best_result = score
                    best_config = individual
                    best_loop = total_evaluated + 1
                    print(f"🎉 找到新的最优解! 得分: {best_result}, 配置: {best_config}")

                xs.append(individual)
                results.append(score)

                # 打印信息
                print(f"评估次数: {total_evaluated + 1}/{budget}")
                print(f"当前配置: {individual}")
                print(f"当前得分: {score}")
                print(f"当前最优得分: {best_result}")
                print("-" * 50)

                total_evaluated += 1
                if total_evaluated >= budget:
                    break

            except Empty:
                print(f"错误: 目标函数调用超时 ({objective_timeout}s)，配置: {individual}")
                # 标记该配置为已评估（避免重复尝试）
                evaluated_configs.add(individual_tuple)
            except Exception as e:
                print(f"错误: 评估过程中发生异常: {e}")

        print(f"本轮评估完成: 成功={eval_success_count}, 跳过={current_eval_count - eval_success_count}")

        if total_evaluated >= budget:
            break

        # 确保有足够的父代进行选择
        if len(fitness_scores) < 2:
            print("警告: 有效评估数量不足，无法进行选择操作")
            # 重新随机生成种群
            population = []
            for _ in range(population_size):
                individual = [random.choice(values) for values in independent_set]
                population.append(individual)
            continue

        # 选择父代
        print(f"选择父代 (当前适应度: {fitness_scores})")
        parents_indices = np.argsort(fitness_scores)[:int(len(fitness_scores) / 2)]
        parents = [population[i] for i in parents_indices]

        print(f"选择了 {len(parents)} 个父代个体")

        # 变异生成子代
        print("开始变异生成子代...")
        offspring = []

        for parent in parents:
            child = []
            for i, value in enumerate(parent):
                if random.random() < 0.2:
                    value_range = independent_set[i]
                    if all(isinstance(v, (int, float)) for v in value_range):
                        range_span = max(value_range) - min(value_range)
                        offset = np.random.normal(0, step_size * range_span)
                        new_value = value + offset
                        new_value = max(min(new_value, max(value_range)), min(value_range))
                    else:
                        new_value = random.choice(value_range)
                    child.append(new_value)
                else:
                    child.append(value)

            # 确保子代配置有效
            for j, value in enumerate(child):
                value_range = independent_set[j]
                if isinstance(value, (int, float)):
                    if value < min(value_range):
                        child[j] = min(value_range)
                    elif value > max(value_range):
                        child[j] = max(value_range)
                else:
                    if value not in value_range:
                        child[j] = random.choice(value_range)

            offspring.append(child)

        print(f"生成了 {len(offspring)} 个子代个体")

        # 更新种群
        population = parents + offspring
        print(f"更新后的种群大小: {len(population)}")

        # 更新自适应步长
        step_size *= step_decay
        print(f"更新步长: {step_size:.4f}")

    total_time = time.time() - start_time
    print(f"\n=== 算法执行完成 ===")
    print(f"总评估次数: {total_evaluated}/{budget}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均每次评估耗时: {total_time / max(1, total_evaluated):.2f}s")
    print(f"最优得分: {best_result}")
    print(f"最优配置: {best_config}")
    print(f"在第 {best_loop} 步找到最优解")

    return xs, results, range(1, len(results) + 1), best_result, best_loop, budget