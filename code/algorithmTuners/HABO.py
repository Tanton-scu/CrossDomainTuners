import numpy as np
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity


class HABOTuner:
    def __init__(self, hyperparameters: dict, gamma: float = 0.1):
        """
        初始化HABO调优器
        :param hyperparameters: 超参数配置字典，键为超参数名称（超级臂），值为可能的取值列表（子臂）
        :param gamma: 探索参数，控制探索程度（0 < gamma <= 1），论文中用于平衡探索与利用
        """
        self.hyperparameters = hyperparameters  # 超级臂：{超参数名: 取值列表}
        self.gamma = gamma
        self.super_arms = list(hyperparameters.keys())
        self.k = len(self.super_arms)  # 超级臂数量

        # 初始化超级臂权重：初始均为1
        self.super_weights = {arm: 1.0 for arm in self.super_arms}

        # 初始化子臂权重：每个超级臂下的子臂初始权重均为1
        self.sub_weights = {
            super_arm: {sub_arm: 1.0 for sub_arm in hyperparameters[super_arm]}
            for super_arm in self.super_arms
        }

        # 记录当前配置（用于逐步构建完整参数组合）
        self.current_config = {super_arm: None for super_arm in self.super_arms}

    def _select_super_arm(self) -> str:
        """选择超级臂（超参数），基于EXP3的概率分布"""
        # 计算超级臂的总权重
        total_weight = sum(self.super_weights.values())
        # 计算每个超级臂的选择概率：(1-γ)*权重占比 + γ/总数量
        probabilities = {
            arm: (1 - self.gamma) * (weight / total_weight) + self.gamma / self.k
            for arm, weight in self.super_weights.items()
        }
        # 基于概率选择超级臂
        #首先将超级臂的名称和对应的选择概率分别提取为列表arms和probs；
        arms = list(probabilities.keys())
        probs = list(probabilities.values())
        #然后使用np.random.choice(arms, p=probs)按照probs中定义的概率分布从arms中随机选择一个超级臂。
        selected_arm = np.random.choice(arms, p=probs)
        return selected_arm

    def _select_sub_arm(self, super_arm: str) -> any:
        """选择子臂（超参数取值），基于当前超级臂下子臂的权重"""
        sub_arms = self.hyperparameters[super_arm]
        sub_weights = self.sub_weights[super_arm]
        total_weight = sum(sub_weights.values())
        # 计算子臂的选择概率：权重占比（无额外探索项，因超级臂选择已包含探索）
        probabilities = {
            sub_arm: weight / total_weight
            for sub_arm, weight in sub_weights.items()
        }
        # 基于概率选择子臂
        sub_arms_list = list(probabilities.keys())
        sub_probs = list(probabilities.values())
        selected_sub_arm = np.random.choice(sub_arms_list, p=sub_probs)
        return selected_sub_arm

    def generate_config(self) -> dict:
        """生成新的超参数配置：选择一个超级臂并更新其对应的子臂"""
        # 选择超级臂（超参数）
        super_arm = self._select_super_arm()
        # 选择子臂（超参数取值）
        sub_arm = self._select_sub_arm(super_arm)
        # 更新当前配置
        self.current_config[super_arm] = sub_arm
        # 返回当前完整配置（未更新的超参数保持上一轮取值，初始为None时需填充默认值）
        # 注：若为首次调用，需为未选择的超参数填充初始值（如第一个取值）
        for arm in self.super_arms:
            if self.current_config[arm] is None:
                self.current_config[arm] = self.hyperparameters[arm][0]
        # 返回当前配置、本轮选中的超级臂和子臂
        return self.current_config.copy(), super_arm, sub_arm

    def update_weights(self, selected_super_arm: str, selected_sub_arm: any, reward: float) -> None:
        """
        根据奖励更新超级臂和子臂的权重
        :param selected_super_arm: 本轮选择的超级臂
        :param selected_sub_arm: 本轮选择的子臂
        :param reward: 该配置对应的性能指标（如准确率、R²等，需为正值）
        """
        # 计算超级臂的选择概率（用于权重更新）
        total_super_weight = sum(self.super_weights.values())
        super_prob = (1 - self.gamma) * (
                self.super_weights[selected_super_arm] / total_super_weight) + self.gamma / self.k

        # 更新超级臂权重：w ← w * exp(γ * reward / (k * P))
        self.super_weights[selected_super_arm] *= np.exp(self.gamma * reward / (self.k * super_prob))

        # 计算子臂的选择概率（用于权重更新）
        total_sub_weight = sum(self.sub_weights[selected_super_arm].values())
        sub_prob = self.sub_weights[selected_super_arm][selected_sub_arm] / total_sub_weight

        # 更新子臂权重：w ← w * exp(γ * reward / Q)
        self.sub_weights[selected_super_arm][selected_sub_arm] *= np.exp(self.gamma * reward / sub_prob)

    def reset(self) -> None:
        """重置调优器状态（用于新的调优任务）"""
        self.super_weights = {arm: 1.0 for arm in self.super_arms}
        self.sub_weights = {
            super_arm: {sub_arm: 1.0 for sub_arm in self.hyperparameters[super_arm]}
            for super_arm in self.super_arms
        }
        self.current_config = {super_arm: None for super_arm in self.super_arms}


def run_tuners(filename, budget=20, seed=0):
    # 设置随机种子
    np.random.seed(seed)
    # 读取数据集，这里可以指定训练集大小、是否打印详细信息
    file = get_data(filename)
    # 自变量的取值范围
    independent_set = file.independent_set
    # 决策与目标值的映射字典
    dict_search = file.dict_search
    # 将自变量取值范围转换为HABO调优器所需的超参数配置格式
    #将数据集的自变量取值范围转换为 HABO 框架中 “超级臂 - 子臂” 的结构。其中，每个自变量对应 HABO 中的一个 “超级臂”（超参数），用param_0、param_1等命名；自变量的所有可能取值则构成该超级臂下的 “子臂”（超参数的可能配置）
    hyperparameters = {f'param_{i}': values for i, values in enumerate(independent_set)}
    # 初始化HABO调优器
    tuner = HABOTuner(hyperparameters)
    # 初始化最优结果
    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []
    results = []

    # 新增：记录score的最大值和最小值（用于归一化）
    score_min = float('inf')  # 最小分数（最优性能）
    score_max = -float('inf')  # 最大分数（最差性能）
    # 开始HABO调优
    for loop in range(budget):
        # 生成新配置，并直接获取本轮选中的超级臂和子臂
        current_config, selected_super_arm, selected_sub_arm = tuner.generate_config()
        # 将字典转换为列表，确保元素顺序与 independent_set 一致
        current_config_values = [current_config[f'param_{i}'] for i in range(len(independent_set))]

        # 评估配置
        score, _ = get_objective_score_with_similarity(dict_search, current_config_values )

        # 动态更新score的最大/最小值
        if score < score_min:
            score_min = score
        if score > score_max:
            score_max = score

        # 奖励归一化：将score映射到 [0, 1] 区间，性能越好（score越小），奖励越高
        # 处理极端情况（所有score相同）
        if score_max == score_min:
            reward = 0.5  # 若所有分数相同，奖励设为中间值
        else:
            # 公式：reward = (score_max - score) / (score_max - score_min)
            # 逻辑：score越小 → (score_max - score) 越大 → reward越接近1（最高奖励）
            reward = (score_max - score) / (score_max - score_min)

        # 更新调优器的权重（使用本轮明确选中的超级臂和子臂）
        #score是奖励，性能越好，奖励越高。而我这个数据集表示性能越好，数值越低，因此给它取个相反数
        tuner.update_weights(selected_super_arm, selected_sub_arm, reward)

        # 记录配置和性能结果
        xs.append(current_config_values)  # 存储为列表
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
    return xs, results, range(1, budget + 1), best_result, best_loop, budget