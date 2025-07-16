import random
import numpy as np
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity

def run_tuners(file ,  budget=20, seed=0, num_particles=10):
    # 设置随机种子，确保结果可复现
    random.seed(seed)
    np.random.seed(seed)

    # 读取数据集
    #file = get_data(filename)
    # 计算最大迭代次数，确保总评估次数不超过预算
    max_iter = budget // num_particles
    if max_iter == 0:
        max_iter = 1  # 至少进行一次迭代
    # 自变量的取值范围，包含数值型和字符型参数
    independent_set = file.independent_set

    # 决策与目标值的映射字典
    dict_search = file.dict_search

    # 初始化最优结果
    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []

    # 初始化粒子的位置和速度
    particles_position = []
    particles_velocity = []
    particles_best_position = []
    particles_best_score = []

    # 有效步骤计数，从粒子初始化评估得分时开始计算
    step = 0

    # 为每个粒子初始化位置、速度、个体最优位置和个体最优得分
    for _ in range(num_particles):
        if step >= budget:
            break
        step += 1
        print(f"初始化粒子时 step: {step}")  # 添加调试信息
        # 随机生成粒子的初始位置，考虑字符型参数
        position = [random.choice(values) for values in independent_set]
        particles_position.append(position)

        # 初始化粒子的速度为0
        velocity = [0] * len(position)
        particles_velocity.append(velocity)

        # 初始个体最优位置为当前位置
        particles_best_position.append(position)

        # 评估当前位置的得分
        score, _ = get_objective_score_with_similarity(dict_search, position)
        particles_best_score.append(score)

        # 记录配置和性能结果
        xs.append(position)
        results.append(score)

        # 更新全局最优结果
        if score < best_result:
            best_result = score
            best_config = position
            best_loop = step

        # 输出每一轮的详细信息
        print(f"有效步骤: {step}/{budget}")
        print(f"当前粒子: {len(particles_position)}/{num_particles}")
        print(f"当前配置: {position}")
        print(f"当前得分: {score}")
        print(f"当前最优得分: {best_result}")
        print(f"当前最优配置: {best_config}")
        print("-" * 50)

    # 迭代更新粒子的位置和速度
    for iter_num in range(1, max_iter + 1):
        if step >= budget:
            break
        for i in range(num_particles):
            if step >= budget:  # 修正：使用 >=
                break
            step += 1
            print(f"迭代更新粒子时 step: {step}")  # 添加调试信息
            # 计算新的速度
            r1, r2 = random.random(), random.random()
            w = 0.7  # 惯性权重
            c1, c2 = 1.4, 1.4  # 学习因子
            for j in range(len(particles_position[i])):
                if isinstance(particles_position[i][j], (int, float, np.number)):
                    # 数值型参数更新速度
                    particles_velocity[i][j] = (
                            w * particles_velocity[i][j] +
                            c1 * r1 * (particles_best_position[i][j] - particles_position[i][j]) +
                            c2 * r2 * (best_config[j] - particles_position[i][j])
                    )
                    # 更新粒子的位置
                    new_position = particles_position[i][j] + particles_velocity[i][j]
                    # 确保新位置在自变量的取值范围内
                    values = independent_set[j]
                    min_val, max_val = min(values), max(values)
                    new_position = max(min_val, min(max_val, new_position))
                    particles_position[i][j] = new_position
                else:
                    # 字符型参数速度保持为0，位置保持不变
                    particles_velocity[i][j] = 0

            # 评估新位置的得分
            score, _ = get_objective_score_with_similarity(dict_search, particles_position[i])

            # 记录配置和性能结果
            xs.append(particles_position[i])
            results.append(score)

            # 更新个体最优结果
            if score < particles_best_score[i]:
                particles_best_score[i] = score
                particles_best_position[i] = particles_position[i]

            # 更新全局最优结果
            if score < best_result:
                best_result = score
                best_config = particles_position[i]
                best_loop = step

            # 输出每一轮的详细信息
            print(f"有效步骤: {step}/{budget}")
            print(f"当前粒子: {i + 1}/{num_particles}")
            print(f"当前配置: {particles_position[i]}")
            print(f"当前得分: {score}")
            print(f"当前最优得分: {best_result}")
            print(f"当前最优配置: {best_config}")
            print("-" * 50)

    # 返回结果
    return xs, results, range(1, len(results) + 1), best_result, best_loop, step