import random
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity

def run_tuners(filename, budget=20, seed=0):
    # 设置随机种子
    random.seed(seed)
    # 读取数据集，这里可以指定训练集大小、是否打印详细信息
    file = get_data(filename)
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

    # 开始随机搜索
    for loop in range(budget):
        # 随机生成配置
        current_config = []
        for values in independent_set:
            current_config.append(random.choice(values))

        # 评估配置
        score, _ = get_objective_score_with_similarity(dict_search, current_config)

        # 记录配置和性能结果
        xs.append(current_config)
        results.append(score)

        # 更新最优结果
        if score < best_result:
            best_result = score
            best_config = current_config
            best_loop = loop + 1

        # 输出每一轮的详细信息
        print(f"轮次: {loop + 1}")
        print(f"当前配置: {current_config}")
        print(f"当前得分: {score}")
        print(f"当前最优得分: {best_result}")
        print(f"当前最优配置: {best_config}")
        print("-" * 50)

    # 返回结果
    #xs：这通常是一个列表或数组，存储了调优过程中尝试的所有自变量配置组合。每一个元素代表一次尝试时的自变量取值情况，比如在一个超参数调优问题中，xs 可能存储了不同超参数的取值组合。
    #results：这也是一个列表或数组，存储了与 xs 中每个自变量配置组合相对应的性能结果（即因变量的值）。results 中的元素顺序与 xs 中的元素顺序一一对应，也就是说 results[i] 是 xs[i] 对应的性能指标。
    #used_budget：表示调优过程中实际使用的预算。在调优任务中，通常会设置一个预算，例如最大迭代次数或者最大计算资源消耗，used_budget 记录了在调优结束时实际消耗的预算。
    return xs, results, range(1, budget + 1), best_result, best_loop, budget