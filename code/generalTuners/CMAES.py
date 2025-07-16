# 导入必要的库
import cma  # 导入CMA-ES库
# 设置随机种子
import random
from ReadDataset import get_data  # 从ReadDataset模块导入get_data函数，用于读取数据集
from QueryDataset import get_objective_score_with_similarity  # 从QueryDataset模块导入get_objective_score_with_similarity函数，用于获取目标分数


def run_tuners(file,   budget=20, seed=0):
    """
    运行CMA-ES调优器
    :param filename: 数据集文件的名称
    :param budget: 调优的预算，即允许的最大迭代次数，默认为20
    :param seed: 随机种子，用于保证结果的可重复性，默认为0
    :return: 调优过程中尝试的所有配置、对应的性能结果、迭代轮次、最优结果、最优结果出现的轮次、实际使用的预算
    """
    random.seed(seed)
    # 读取数据集
    independent_set = file.independent_set  # 获取自变量的取值范围
    dict_search = file.dict_search  # 获取决策与目标值的映射字典

    # 初始化最优结果
    best_result = float('inf')  # 初始化最优结果为正无穷大
    best_config = None  # 初始化最优配置为None
    best_loop = 0  # 初始化最优结果出现的轮次为0
    xs = []  # 用于存储调优过程中尝试的所有配置
    results = []  # 用于存储与每个配置对应的性能结果
    # 用于记录已经生成过的配置
    generated_configs = set()

    # 定义目标函数
    def objective_function(config):
        """
        目标函数，用于评估给定配置的性能
        :param config: 待评估的配置
        :return: 配置的性能得分
        """
        # 将配置中的值映射到自变量的取值范围内
        new_config = []
        for i, values in enumerate(independent_set):
            index = int(config[i] * len(values)) % len(values)  # 根据配置值计算对应的索引
            new_config.append(values[index])  # 将对应的自变量值添加到新配置中

        # 评估配置
        score, _ = get_objective_score_with_similarity(dict_search, new_config)  # 调用get_objective_score_with_similarity函数获取配置的性能得分
        return score, new_config  # 返回性能得分和映射后的配置

    # 初始化CMA-ES优化器
    initial_mean = [0.5] * len(independent_set)  # 初始化优化器的均值向量
    initial_sigma = 0.3  # 初始化优化器的步长
    pop_size = 10  # 设置种群规模
    es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, {'seed': seed, 'maxfevals': budget, 'popsize': pop_size})  # 创建CMA-ES优化器实例

    # 开始优化过程
    loop = 0  # 已使用的预算计数
    while not es.stop() and loop < budget:
        solutions = es.ask()  # 从优化器中获取一组新的解
        valid_solutions = []  # 存储不重复的新配置
        mapped_configs = []   # 存储映射后的新配置（用于后续记录）

        for sol in solutions:
            # 将配置中的值映射到自变量的取值范围内
            new_config = []
            for i, values in enumerate(independent_set):
                index = int(sol[i] * len(values)) % len(values)
                new_config.append(values[index])

            config_tuple = tuple(new_config)
            if config_tuple not in generated_configs:
                # 仅保留不重复的配置
                valid_solutions.append(sol)
                mapped_configs.append(new_config)
                generated_configs.add(config_tuple)

        # 确保有足够的有效解传递给CMA-ES（至少需要2个）
        if len(valid_solutions) < 2:
            print(f"警告: 有效解数量不足({len(valid_solutions)})，提前终止优化")
            break

        # 检查是否会超过预算，裁剪有效解数量
        remaining_budget = budget - loop
        if len(valid_solutions) > remaining_budget:
            if remaining_budget >= 2:
                valid_solutions = valid_solutions[:remaining_budget]
                mapped_configs = mapped_configs[:remaining_budget]
            else:
                print(f"警告: 剩余预算不足({remaining_budget})，提前终止优化")
                break

        # 评估新配置（消耗预算）
        fitnesses = []
        for sol in valid_solutions:
            fit, _ = objective_function(sol)
            fitnesses.append(fit)

        # 反馈给CMA-ES优化器
        es.tell(valid_solutions, fitnesses)
        # 更新预算计数（仅新配置消耗预算）
        loop += len(valid_solutions)

        # 记录配置和性能结果
        for config, fit in zip(mapped_configs, fitnesses):
            xs.append(config)
            results.append(fit)

            # 更新最优结果
            if fit < best_result:
                best_result = fit
                best_config = config
                best_loop = len(xs)  # 记录当前轮次（xs的长度即总评估次数）

            # 输出每一轮的详细信息
            print(f"轮次: {len(xs)}")
            print(f"当前配置: {config}")
            print(f"当前得分: {fit}")
            print(f"当前最优得分: {best_result}")
            print(f"当前最优配置: {best_config}")
            print("-" * 50)

    # 返回结果
    print(loop)
    return xs, results, range(1, len(xs) + 1), best_result, best_loop, loop