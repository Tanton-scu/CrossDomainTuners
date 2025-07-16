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
        arms = list(probabilities.keys())
        probs = list(probabilities.values())
        selected_arm = np.random.choice(arms, p=probs)
        return selected_arm

    def _select_sub_arm(self, super_arm: str) -> any:
        """选择子臂（超参数取值），基于当前超级臂下子臂的权重"""
        sub_arms = self.hyperparameters[super_arm]
        sub_weights = self.sub_weights[super_arm]
        total_weight = sum(sub_weights.values())

        # 处理除零错误
        if total_weight == 0:
            num_sub_arms = len(sub_arms)
            probabilities = {sub_arm: 1 / num_sub_arms for sub_arm in sub_arms}
        else:
            # 计算子臂的选择概率：权重占比
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

        # 避免指数运算溢出
        exp_super = self.gamma * reward / (self.k * super_prob)
        if exp_super > 709:  # np.exp(709) 接近最大值
            exp_super = 709
        # 更新超级臂权重：w ← w * exp(γ * reward / (k * P))
        self.super_weights[selected_super_arm] *= np.exp(exp_super)

        # 计算子臂的选择概率（用于权重更新）
        total_sub_weight = sum(self.sub_weights[selected_super_arm].values())
        sub_prob = self.sub_weights[selected_super_arm][selected_sub_arm] / total_sub_weight

        # 避免指数运算溢出
        exp_sub = self.gamma * reward / sub_prob
        if exp_sub > 709:
            exp_sub = 709
        # 更新子臂权重：w ← w * exp(γ * reward / Q)
        self.sub_weights[selected_super_arm][selected_sub_arm] *= np.exp(exp_sub)

    def reset(self) -> None:
        """重置调优器状态（用于新的调优任务）"""
        self.super_weights = {arm: 1.0 for arm in self.super_arms}
        self.sub_weights = {
            super_arm: {sub_arm: 1.0 for sub_arm in self.hyperparameters[super_arm]}
            for super_arm in self.super_arms
        }
        self.current_config = {super_arm: None for super_arm in self.super_arms}


def run_tuners(file, budget=20, seed=0):
    # 设置随机种子
    np.random.seed(seed)
    # 读取数据集
    # file = get_data(filename)
    independent_set = file.independent_set
    dict_search = file.dict_search
    # 转换为HABO调优器所需的超参数格式
    hyperparameters = {f'param_{i}': values for i, values in enumerate(independent_set)}
    # 初始化HABO调优器
    tuner = HABOTuner(hyperparameters)
    # 初始化最优结果
    best_result = float('inf')
    best_config = None
    best_step = 0
    xs = []
    results = []

    # 新增：记录历史配置（用于去重）
    history_configs = set()  # 存储已生成的配置（元组形式，可哈希）
    # 新增：记录score的最大/最小值（用于归一化）
    score_min = float('inf')
    score_max = -float('inf')

    # 新增：重复配置计数器
    repeat_count = 0
    # 新增：最大重复次数阈值
    max_repeats = 100 * budget

    # 调整循环逻辑：直到消耗完预算且所有配置均为新配置
    step = 0  # 有效步骤计数（仅统计新配置）
    while step < budget:
        # 检查是否达到最大重复次数
        if repeat_count >= max_repeats:
            print(f"警告: 已生成 {repeat_count} 次重复配置，达到阈值 {max_repeats}，提前终止搜索")
            break

        # 生成配置（可能重复）
        current_config, selected_super_arm, selected_sub_arm = tuner.generate_config()
        current_config_values = [current_config[f'param_{i}'] for i in range(len(independent_set))]
        config_tuple = tuple(current_config_values)  # 转换为元组便于哈希存储

        # 检查是否为重复配置
        if config_tuple in history_configs:
            print(f"步骤 {step + 1} 生成重复配置 {config_tuple}，重新生成...")
            repeat_count += 1  # 增加重复计数
            continue  # 重复配置不消耗预算，重新生成

        # 重置重复计数（找到新配置）
        repeat_count = 0

        # 新配置：加入历史记录
        history_configs.add(config_tuple)

        # 评估配置
        score, _ = get_objective_score_with_similarity(dict_search, current_config_values)

        # 动态更新score的最大/最小值
        if score < score_min:
            score_min = score
        if score > score_max:
            score_max = score

        # 奖励归一化：性能越好（score越小），奖励越高
        if score_max == score_min:
            reward = 0.5  # 所有分数相同时的默认奖励
        else:
            reward = (score_max - score) / (score_max - score_min)

        # 更新权重（仅对新配置）
        tuner.update_weights(selected_super_arm, selected_sub_arm, reward)

        # 记录结果
        xs.append(current_config_values)
        results.append(score)

        # 更新最优结果
        if score < best_result:
            best_result = score
            best_config = current_config
            best_step = step + 1  # 记录最优结果出现的有效步骤

        # 输出有效步骤信息
        print(f"有效步骤: {step + 1}/{budget}")
        print(f"当前配置: {current_config}")
        print(f"当前得分: {score}")
        print(f"当前最优得分: {best_result}")
        print(f"当前最优配置: {best_config}")
        print("-" * 50)

        # 有效步骤计数+1
        step += 1

    return xs, results, range(1, budget + 1), best_result, best_step, step