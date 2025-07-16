import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity


class SMACTuner:
    def __init__(self, param_space, n_trees=10, ei_weight=0.5):
        """
        初始化SMAC调优器
        :param param_space: 参数空间，格式为[{参数1可能值}, {参数2可能值}, ...]
        :param n_trees: 随机森林树的数量（论文中B=10）
        :param ei_weight: 探索与利用平衡权重（影响EI计算）
        """
        self.param_space = param_space  # 参数空间
        self.n_trees = n_trees  # 随机森林树数量
        self.ei_weight = ei_weight  # EI权重
        self.model = None  # 随机森林模型
        self.X = []  # 已评估配置（特征）
        self.y = []  # 已评估配置的性能（目标值）
        self.incumbent = None  # 当前最优配置
        self.incumbent_score = float('inf')  # 当前最优得分

    def _convert_config_to_features(self, config):
        """
        将配置转换为模型可接受的特征（处理字符型参数）
        字符型参数转换为独热编码索引，数值型保持原值
        """
        features = []
        for i, val in enumerate(config):
            param_values = self.param_space[i]
            # 字符型参数：转换为索引
            if isinstance(val, str):
                features.append(param_values.index(val))
            # 数值型参数：直接使用值
            else:
                features.append(val)
        return features

    def fit_model(self):
        """
        拟合随机森林模型（论文4.1节模型构建逻辑）
        基于历史评估数据训练模型，用于预测新配置性能
        """
        if len(self.X) < 2:  # 至少需要2个样本才能训练模型
            return

        # 处理目标值，确保对数计算的稳定性
        y_values = np.array(self.y)

        # 1. 确保所有值都大于某个极小值，避免接近零的问题
        min_value = -1e-5  # 调整为更大的最小值，避免过于接近零
        y_non_negative = np.maximum(y_values, min_value) - min_value + 1e-5

        # 2. 对处理后的值进行对数转换
        # 使用np.log1p（计算log(1+x)）更适合处理接近零的值
        # 这里先将值缩放，使得log1p更稳定
        scale_factor = 1e5
        y_scaled = y_non_negative * scale_factor
        y_log = np.log1p(y_scaled)  # log(1 + x)，对小x更精确

        # 3. 构建随机森林回归模型
        self.model = RandomForestRegressor(
            n_estimators=self.n_trees,
            random_state=42,
            n_jobs=-1
        )

        # 4. 训练模型
        self.model.fit(self.X, y_log)

    def _expected_improvement(self, config):
        """
        计算预期改进值（EI），用于选择下一个配置（论文4.3节）
        EI = 平衡探索（高不确定性）和利用（高预测性能）
        """
        if self.model is None:  # 模型未训练时返回随机值
            return random.random()

        # 转换配置为特征并预测
        features = self._convert_config_to_features(config)
        # 随机森林预测：获取所有树的预测结果以计算方差
        y_preds = [tree.predict([features])[0] for tree in self.model.estimators_]
        mu = np.mean(y_preds)  # 预测均值
        sigma = np.std(y_preds)  # 预测方差（不确定性）

        # 计算EI（基于log转换的目标值，论文公式1）
        # 处理当前最优得分，确保与训练数据处理方式一致
        incumbent_score_non_negative = max(self.incumbent_score, -1e-5) - (-1e-5) + 1e-5
        incumbent_score_scaled = incumbent_score_non_negative * 1e5
        f_min = np.log1p(incumbent_score_scaled)  # 使用log1p保持一致性

        if sigma < 1e-6:  # 无不确定性时EI为0
            return 0.0
        z = (f_min - mu) / sigma
        # 标准正态分布的CDF和PDF
        cdf = 0.5 * (1 + np.math.erf(z / np.sqrt(2)))
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        ei = (f_min - mu) * cdf + sigma * pdf
        return ei  # 越大越值得探索

    def select_next_config(self, num_candidates=1000):
        """
        选择下一个最有前景的配置（论文4.3节选择策略）
        生成候选配置并通过EI排序选择最优
        """
        candidates = []
        # 生成候选配置（混合随机采样和局部搜索）
        for _ in range(num_candidates):
            # 随机生成配置
            config = [random.choice(vals) for vals in self.param_space]
            candidates.append(config)

        # 对候选配置按EI排序，选择最高的
        candidates.sort(key=lambda x: self._expected_improvement(x), reverse=True)
        return candidates[0]

    def intensify(self, config, dict_search, history_configs):
        """
        强化评估：比较新配置与当前最优（论文3.1节Intensify逻辑）
        确保新配置确实优于当前最优才更新
        """
        config_tuple = tuple(config)
        # 评估配置（若已评估则直接获取结果）
        if config_tuple in history_configs:
            score, _ = get_objective_score_with_similarity(dict_search, config)
        else:
            score, _ = get_objective_score_with_similarity(dict_search, config)

        # 更新当前最优
        if score < self.incumbent_score:
            self.incumbent = config
            self.incumbent_score = score
        return score


def run_tuners(file,   budget=20, seed=0, num_candidates=1000):
    """
    SMAC调优器主函数
    :param filename: 数据集路径
    :param budget: 最大评估次数（预算）
    :param seed: 随机种子（确保可复现）
    :param num_candidates: 每次选择的候选配置数量
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 读取数据集（论文5.2节数据加载逻辑）
    # file = get_data(filename)
    param_space = file.independent_set  # 参数空间：每个参数的可能值
    dict_search = file.dict_search  # 配置-性能映射字典

    # 初始化变量
    best_result = float('inf')  # 全局最优得分
    best_config = None  # 全局最优配置
    xs = []  # 记录所有评估过的配置
    results = []  # 记录对应得分
    history_configs = set()  # 记录已评估配置（去重）
    budget_used = 0  # 已使用预算

    # 初始化SMAC调优器
    tuner = SMACTuner(param_space)

    # 初始配置：随机生成1个作为初始 incumbent（论文3.2节初始化）
    initial_config = [random.choice(vals) for vals in param_space]
    initial_score, _ = get_objective_score_with_similarity(dict_search, initial_config)
    # 消耗预算记录初始配置
    budget_used += 1
    history_configs.add(tuple(initial_config))
    xs.append(initial_config)
    results.append(initial_score)
    # 更新初始最优
    tuner.incumbent = initial_config
    tuner.incumbent_score = initial_score
    best_result = initial_score
    best_config = initial_config
    # 记录初始数据用于模型训练
    tuner.X.append(tuner._convert_config_to_features(initial_config))
    tuner.y.append(initial_score)

    # 输出初始配置信息
    print(f"轮次: 1/{budget} (初始配置)")
    print(f"配置: {initial_config}")
    print(f"得分: {initial_score}")
    print(f"全局最优得分: {best_result}")
    print(f"全局最优配置: {best_config}\n{'-' * 50}")

    # 主循环：在预算内迭代优化
    while budget_used < budget:
        # 1. 拟合模型（论文4.1节随机森林训练）
        tuner.fit_model()

        # 2. 选择下一个配置（论文4.3节EI选择策略）
        next_config = tuner.select_next_config(num_candidates)
        config_tuple = tuple(next_config)

        # 3. 评估配置（若已评估则跳过，不消耗预算）
        if config_tuple in history_configs:
            continue

        # 4. 消耗预算评估新配置
        budget_used += 1
        score, _ = get_objective_score_with_similarity(dict_search, next_config)
        # 记录配置和得分
        history_configs.add(config_tuple)
        xs.append(next_config)
        results.append(score)
        # 更新模型训练数据
        tuner.X.append(tuner._convert_config_to_features(next_config))
        tuner.y.append(score)

        # 5. 强化评估并更新全局最优（论文3.1节Intensify）
        tuner.intensify(next_config, dict_search, history_configs)
        # 更新全局最优
        if score < best_result:
            best_result = score
            best_config = next_config

        # 6. 输出本轮信息
        print(f"轮次: {budget_used}/{budget}")
        print(f"配置: {next_config}")
        print(f"得分: {score}")
        print(f"全局最优得分: {best_result}")
        print(f"全局最优配置: {best_config}\n{'-' * 50}")

    # 返回调优结果（符合SaveToCSV.py的输出格式）
    final_round = len(xs)
    return xs, results, range(1, final_round + 1), best_result, np.argmin(results) + 1, final_round