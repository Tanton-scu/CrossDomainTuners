# generalTuners/BO.py
# 导入 random 模块，用于生成随机数
import random
# 导入 numpy 模块，用于数值计算和数组操作
import numpy as np
# 从 ReadDataset 模块中导入 get_data 函数，用于读取数据集
from ReadDataset import get_data
# 从 QueryDataset 模块中导入 get_objective_score_with_similarity 函数，用于获取目标分数
from QueryDataset import get_objective_score_with_similarity
# 从 sklearn.gaussian_process 模块中导入 GaussianProcessRegressor 类，用于高斯过程回归
from sklearn.gaussian_process import GaussianProcessRegressor
# 从 sklearn.gaussian_process.kernels 模块中导入 Matern 类，用于定义高斯过程的核函数
from sklearn.gaussian_process.kernels import Matern
# 从 scipy.stats 模块中导入 norm 类，用于计算正态分布的累积分布函数和概率密度函数
from scipy.stats import norm
# 导入 OneHotEncoder 用于编码分类变量
from sklearn.preprocessing import OneHotEncoder

def run_tuners(file,  budget=20, seed=0):
    # 设置随机种子，确保结果的可重复性
    random.seed(seed)
    np.random.seed(seed)
    # 读取数据集
    # 自变量的取值范围
    # 从数据集对象中获取自变量的取值范围
    independent_set = file.independent_set
    # 决策与目标值的映射字典
    # 从数据集对象中获取决策与目标值的映射字典
    dict_search = file.dict_search

    # 初始化最优结果
    # 初始化最优结果为正无穷大
    best_result = float('inf')
    # 初始化最优配置为 None
    best_config = None
    # 初始化达到最优结果的轮次为 0
    best_loop = 0
    # 用于存储所有评估过的配置
    xs = []
    # 用于存储所有评估过的配置对应的性能结果
    results = []

    # 初始化高斯过程回归模型
    # 定义 Matern 核函数，nu=2.5 是核函数的参数
    kernel = Matern(nu=2.5)
    # 创建高斯过程回归模型对象，使用定义的核函数，n_restarts_optimizer=20 表示优化核函数超参数的重启次数
    #GaussianProcessRegressor 的 n_restarts_optimizer 参数从 10 提升到 20，这样优化器就有更多的迭代次数去寻找最优解。
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

    # 初始化编码器并拟合所有可能的类别
    encoder = OneHotEncoder(handle_unknown='ignore')
    all_possible_configs = [list(product) for product in zip(*independent_set)]
    encoder.fit(all_possible_configs)

    # 新增：记录历史配置（用于去重）
    history_configs = set()  # 存储已生成的配置（元组形式，可哈希）

    # 随机初始化一些样本
    # 确定初始样本的数量，取 5 和预算的最小值
    initial_samples = min(5, budget)
    step = 0  # 有效步骤计数（仅统计新配置）
    while step < initial_samples:
        # 随机生成一个配置
        # 从每个自变量的取值范围中随机选择一个值，组成一个配置
        current_config = [random.choice(values) for values in independent_set]
        current_config_tuple = tuple(current_config)  # 转换为元组便于哈希存储

        # 检查是否为重复配置
        if current_config_tuple in history_configs:
            print(f"步骤 {step + 1} 生成重复配置 {current_config_tuple}，重新生成...")
            continue  # 重复配置不消耗预算，重新生成

        # 新配置：加入历史记录
        history_configs.add(current_config_tuple)

        # 评估当前配置的性能
        # 调用 get_objective_score_with_similarity 函数，传入映射字典和当前配置，获取当前配置的得分
        current_score, _ = get_objective_score_with_similarity(dict_search, current_config)

        # 更新最优结果
        # 如果当前得分小于最优结果
        if current_score < best_result:
            # 更新最优结果为当前得分
            best_result = current_score
            # 更新最优配置为当前配置
            best_config = current_config
            # 更新达到最优结果的轮次为当前轮次
            best_loop = step + 1

        # 记录配置和性能结果
        # 将当前配置添加到 xs 列表中
        xs.append(current_config)
        # 将当前得分添加到 results 列表中
        results.append(current_score)

        # 输出每一轮的详细信息
        print(f"有效步骤: {step + 1}")
        print(f"当前配置: {current_config}")
        print(f"当前得分: {current_score}")
        print(f"当前最优得分: {best_result}")
        print(f"当前最优配置: {best_config}")
        print("-" * 50)

        step += 1

    # 主循环：贝叶斯优化
    # 从初始样本数量开始，循环到预算次数
    while step < budget:
        # 训练高斯过程回归模型
        # 将 xs 列表转换为 numpy 数组
        X = np.array(xs)
        # 对 X 进行编码
        X_encoded = encoder.transform(X).toarray()
        # 将 results 列表转换为 numpy 数组
        y = np.array(results)
        # 使用 X 和 y 训练高斯过程回归模型
        gp.fit(X_encoded, y)

        # 采集函数：期望改进（EI）
        # EI 函数计算每个候选配置的期望改进值，即选择该配置可能带来的性能提升的期望
        def expected_improvement(X, X_sample, y_sample, gp, xi=0.01):
            # 对 X 进行编码
            X_encoded = encoder.transform(X).toarray()
            # 使用高斯过程回归模型预测 X 的均值和标准差
            mu, std = gp.predict(X_encoded, return_std=True)
            # 对 X_sample 进行编码
            X_sample_encoded = encoder.transform(X_sample).toarray()
            # 使用高斯过程回归模型预测 X_sample 的均值
            mu_sample = gp.predict(X_sample_encoded)

            # 获取 X_sample 预测均值中的最小值
            mu_sample_opt = np.min(mu_sample)
            # 计算改进值
            # mu：是高斯过程回归模型对候选配置的预测均值，表示模型认为该候选配置可能达到的目标值
            # mu_sample_opt：是当前已评估样本中目标值的最小值，代表目前找到的最优结果
            # xi：是一个超参数，通常设置为一个小的正数（如 0.01），用于控制探索和利用的平衡。较大的 xi 值会鼓励更多的探索，而较小的 xi 值会更倾向于利用已有的信息
            # 在目标是使目标值越小越好的优化场景下，应该是 imp 为负的时候，候选配置可能更好
            imp = mu - mu_sample_opt - xi
            # 计算 Z 值
            # std：是高斯过程回归模型对候选配置的预测标准差，表示模型对该候选配置预测的不确定性。
            # Z 值是改进值 imp 与预测标准差 std 的比值，可以看作是标准化后的改进值
            Z = imp / std
            # 计算期望改进值
            # imp * norm.cdf(Z)：表示在候选配置有可能改进的情况下，改进值的期望贡献
            # std * norm.pdf(Z)：表示由于预测的不确定性，可能带来的额外改进的期望贡献
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            # 将标准差为 0 的期望改进值设为 0
            # 当预测标准差 std 为 0 时，说明模型对该候选配置的预测是确定的，没有不确定性。在这种情况下，候选配置的改进值是已知的，不会带来额外的信息，因此将期望改进值 ei 设为 0
            ei[std == 0.0] = 0.0
            return ei

        # 生成候选配置
        # 定义候选配置的数量
        num_candidates = 100
        # 用于存储候选配置的列表
        candidates = []
        # 循环生成候选配置
        for _ in range(num_candidates):
            # 随机生成一个候选配置
            candidate = [random.choice(values) for values in independent_set]
            # 将候选配置添加到 candidates 列表中
            candidates.append(candidate)
        # 将 candidates 列表转换为 numpy 数组
        candidates = np.array(candidates)

        # 计算期望改进
        # 调用 expected_improvement 函数，计算每个候选配置的期望改进值
        ei_values = expected_improvement(candidates, X, y, gp)

        # 选择具有最大期望改进的配置
        # 获取期望改进值最大的候选配置的索引
        next_index = np.argmax(ei_values)
        # 根据索引获取下一个要评估的配置
        next_config = candidates[next_index]
        next_config_tuple = tuple(next_config)  # 转换为元组便于哈希存储

        # 检查是否为重复配置
        if next_config_tuple in history_configs:
            print(f"步骤 {step + 1} 生成重复配置 {next_config_tuple}，重新生成...")
            continue  # 重复配置不消耗预算，重新生成

        # 新配置：加入历史记录
        history_configs.add(next_config_tuple)

        # 评估下一个配置
        # 调用 get_objective_score_with_similarity 函数，传入映射字典和下一个配置，获取下一个配置的得分
        next_score, _ = get_objective_score_with_similarity(dict_search, next_config)

        # 更新最优结果
        # 如果下一个配置的得分小于最优结果
        if next_score < best_result:
            # 更新最优结果为下一个配置的得分
            best_result = next_score
            # 更新最优配置为下一个配置
            best_config = next_config
            # 更新达到最优结果的轮次为当前轮次
            best_loop = step + 1

        # 记录配置和性能结果
        # 将下一个配置添加到 xs 列表中
        xs.append(next_config)
        # 将下一个配置的得分添加到 results 列表中
        results.append(next_score)

        # 输出每一轮的详细信息
        print(f"有效步骤: {step + 1}")
        print(f"当前配置: {next_config}")
        print(f"当前得分: {next_score}")
        print(f"当前最优得分: {best_result}")
        print(f"当前最优配置: {best_config}")
        print("-" * 50)

        step += 1

    # 返回所有评估过的配置、对应的性能结果、轮次范围、最优结果、达到最优结果的轮次和预算次数
    return xs, results, range(1, len(results) + 1), best_result, best_loop, budget