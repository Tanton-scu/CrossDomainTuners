import random
import numpy as np
# 导入读取数据集的工具函数
from ReadDataset import get_data
# 导入配置评估工具（计算相似配置的目标值）
from QueryDataset import get_objective_score_with_similarity


def generate_random_config(independent_set):
    """生成随机配置（论文中初始化种群的方式）
    功能：从每个参数的可选值列表中随机选择一个值，组成完整配置
    independent_set：数据集的自变量集合，每个元素是某一参数的所有可选值列表
    """
    return [random.choice(values) for values in independent_set]


def crossover(parent_c, parent_n):
    """交叉操作（基于论文Algorithm 1和图2的简化实现）
    功能：模拟生物繁殖中的基因重组，结合两个父代的特征生成子代
    parent_c：竞争性别（C）父代配置
    parent_n：非竞争性别（N）父代配置
    """
    # 随机选择交叉点（确保至少交换1个参数）
    crossover_point = random.randint(1, len(parent_c) - 1)
    # 交叉点前取parent_c的参数，交叉点后取parent_n的参数
    return parent_c[:crossover_point] + parent_n[crossover_point:]


def mutation(config, independent_set, mutation_rate=0.1):
    """变异操作（修复字符型参数唯一值问题）
    论文中M=10%，数值型用高斯分布，字符型随机选择
    功能：随机改变配置中的部分参数，增加种群多样性
    config：待变异的配置
    independent_set：参数可选值集合，用于确定变异范围
    mutation_rate：变异概率（默认10%）
    """
    for i in range(len(config)):
        # 按概率决定是否变异当前参数
        if random.random() < mutation_rate:
            # 获取当前参数的所有可选值
            values = independent_set[i]
            # 字符型参数处理：若参数值不是数值（int/float），则视为字符型
            if not isinstance(config[i], (int, float)):
                # 筛选与当前值不同的可选值（避免无意义变异）
                other_values = [v for v in values if v != config[i]]
                if other_values:  # 若存在其他可选值则随机选择一个
                    config[i] = random.choice(other_values)
                # 若没有其他可选值，则不变异（保持当前值）
            # 数值型参数处理
            else:
                # 确定参数的取值范围
                min_val, max_val = min(values), max(values)
                domain = max_val - min_val  # 参数值域大小
                mean = config[i]  # 以当前值为均值
                std = domain * 0.05  # 论文中S=5%（值域的5%作为标准差）
                # 生成高斯分布的新值
                new_val = np.random.normal(mean, std)
                # 确保新值在[min_val, max_val]范围内（截断处理）
                config[i] = max(min(new_val, max_val), min_val)
                # 若原参数是整数类型，转换为整数
                if isinstance(config[i], int):
                    config[i] = int(config[i])
    return config


def run_tuners(file,budget=20, seed=0, population_size=100, generations=50):
    """GGA调优器主函数（遵循论文性别分离机制）
    功能：实现遗传全局算法（GGA）的完整调优流程，在预算范围内寻找最优配置
    filename：数据集路径
    budget：预算（最大评估次数）
    seed：随机种子（保证实验可复现）
    population_size：种群大小（每代的个体数量）
    generations：进化代数（迭代次数）
    """
    # 设置随机种子，确保实验可复现
    random.seed(seed)
    np.random.seed(seed)

    # 读取数据集：获取参数可选值、配置-目标值映射等信息
    #file = get_data(filename)
    independent_set = file.independent_set  # 每个参数的可选值列表
    dict_search = file.dict_search  # 配置到目标值的映射字典

    # 初始化变量：记录调优过程中的关键信息
    best_result = float('inf')  # 全局最优得分（初始化为无穷大）
    best_config = None  # 全局最优配置
    xs = []  # 记录所有评估过的配置
    results = []  # 记录所有配置的得分
    history_configs = set()  # 记录已评估的配置（去重，用元组哈希存储）

    # 初始化种群：分配性别（C/N）和年龄（遵循论文机制）
    population = []
    for _ in range(population_size):#先随机生成population_size个配置
        # 生成随机初始配置
        config = generate_random_config(independent_set)
        # 随机分配性别：C（竞争型）或N（非竞争型）
        gender = random.choice(['C', 'N'])
        # 随机分配年龄（论文中A=3，年龄范围0-3）
        age = random.randint(0, 3)
        # 将个体信息加入种群（包含配置、性别、年龄、得分（暂空））
        population.append({'config': config, 'gender': gender, 'age': age, 'score': None})

    # 进化迭代（每代执行选择、交叉、变异）
    for generation in range(generations):#这里是要进行generations次吗
        # 评估竞争性别（C）个体（非竞争性别N暂不评估）
        for ind in population:#对于初始化好的这population_size个配置，只要是竞争型的配置，就评估
            # 仅评估未打分的竞争型个体
            if ind['gender'] == 'C' and ind['score'] is None:
                config = ind['config']  # 当前个体的配置
                config_tuple = tuple(config)  # 转换为元组用于哈希存储

                # 检查配置是否已评估过
                if config_tuple in history_configs:
                    # 已评估：直接获取得分（不消耗预算）
                    score, _ = get_objective_score_with_similarity(dict_search, config)
                else:
                    # 新配置：评估得分并消耗预算
                    score, _ = get_objective_score_with_similarity(dict_search, config)
                    history_configs.add(config_tuple)  # 记录到历史配置
                    xs.append(config)  # 保存配置
                    results.append(score)  # 保存得分

                    # 打印本轮信息
                    current_round = len(xs)  # 当前已评估次数（即本轮编号）
                    print(f"轮次: {current_round}/{budget}")  # 显示进度
                    print(f"配置: {config}")  # 显示当前配置
                    print(f"得分: {score}")  # 显示当前得分

                    # 更新全局最优
                    if score < best_result:
                        best_result = score
                        best_config = config

                    # 显示全局最优信息
                    print(f"全局最优得分: {best_result}")
                    print(f"全局最优配置: {best_config}\n{'-'*50}")

                    # 检查是否达到预算上限，若达到则提前终止并返回结果
                    if current_round >= budget:
                        return xs, results, range(1, current_round+1), best_result, current_round, current_round

                # 记录当前个体的得分
                ind['score'] = score

        # 选择竞争性别（C）中的优秀个体作为父代（论文中的选择机制）
        # 筛选所有竞争型个体
        competitive = [ind for ind in population if ind['gender'] == 'C']
        # 按得分升序排序（假设得分越低越好）
        competitive.sort(key=lambda x: x['score'])
        # 选择前10%的个体作为交配父代（X=10%）
        num_mating = max(1, int(len(competitive)*0.1))  # 至少选择1个
        mating_c = competitive[:num_mating]  # 竞争型父代列表

        # 选择非竞争性别（N）个体作为交配母代
        non_competitive = [ind for ind in population if ind['gender'] == 'N']
        # 每个竞争型父代匹配的非竞争型母代数量（论文中200/A%，A=3时约67%）
        num_n_per_c = max(1, int(len(non_competitive)* (200/3)/100))
        # 随机配对（每个C父代与num_n_per_c个N母代配对）
        mating_pairs = [(c, n) for c in mating_c for n in random.sample(non_competitive, num_n_per_c)]

        # 产生子代：通过交叉和变异生成新个体
        offspring = []
        for c, n in mating_pairs:
            # 交叉：结合父代C和父代N的配置
            child_config = crossover(c['config'], n['config'])
            # 变异：引入随机变化
            child_config = mutation(child_config, independent_set)
            # 为子代分配随机性别、初始年龄（0），未评估状态（score=None）
            offspring.append({
                'config': child_config,
                'gender': random.choice(['C', 'N']),
                'age': 0,
                'score': None
            })

        # 年龄更新与淘汰机制（论文中的年龄限制）
        # 所有个体年龄+1
        for ind in population:
            ind['age'] += 1
        # 淘汰年龄超过3的个体（A=3）
        population = [ind for ind in population if ind['age'] <= 3]
        # 将新生成的子代加入种群
        population.extend(offspring)
        # 维持种群大小：若超过上限则随机保留部分个体
        if len(population) > population_size:
            population = random.sample(population, population_size)

    # 所有进化代数结束后，返回最终结果
    final_round = len(xs)
    return xs, results, range(1, final_round+1), best_result, final_round, final_round