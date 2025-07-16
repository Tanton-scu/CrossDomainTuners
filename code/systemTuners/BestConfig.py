import random
import numpy as np
from ReadDataset import get_data
from QueryDataset import get_objective_score_with_similarity


def enum_to_numeric(enum_values):
    """将枚举值映射为连续数值（如["low", "medium", "high"]→[0,1,2]）
    论文4.1节：枚举参数通过数值映射转为连续范围，确保DDS可划分区间
    """
    # 去重并保持原始顺序（避免重复值影响映射一致性）
    unique_enums = list(dict.fromkeys(enum_values))
    enum_map = {enum: i for i, enum in enumerate(unique_enums)}
    numeric_values = [enum_map[enum] for enum in enum_values]
    return numeric_values, enum_map, unique_enums


def numeric_to_enum(numeric_value, unique_enums):
    """将数值反向映射为枚举值（如1→"medium"）
    论文4.1节：采样生成的数值需转回枚举值以匹配系统实际配置
    """
    idx = int(round(numeric_value))  # 处理采样时的浮点值
    idx = max(0, min(idx, len(unique_enums) - 1))  # 边界保护
    return unique_enums[idx]


def divide_parameter_ranges(param_values, k):
    """将参数取值范围划分为k个区间（支持数值型和枚举型）
    论文4.1节：DDS核心逻辑，确保每个区间在样本中至少出现一次
    """
    numeric_values, enum_map, unique_enums = enum_to_numeric(param_values)
    sorted_values = sorted(numeric_values)
    param_min, param_max = sorted_values[0], sorted_values[-1]

    # 若参数可能值数量≤k：每个值作为独立区间（离散场景）
    if len(sorted_values) <= k:
        return [[enum] for enum in unique_enums]  # 枚举值保留原始类型

    # 连续值场景：按参数理论范围均匀划分
    step = (param_max - param_min) / k  # 区间步长
    ranges = []
    for i in range(k):
        start = param_min + i * step
        end = start + step if i < k - 1 else param_max  # 最后一个区间含最大值
        ranges.append((start, end))
    return ranges


def generate_dds_samples(independent_set, num_samples):
    """生成DDS采样样本（高维空间均匀覆盖）
    论文4.1节：通过"划分-分散"策略确保样本在参数空间均匀分布
    """
    k = num_samples  # 样本数量=区间划分粒度k
    param_ranges = []  # 每个参数的区间划分结果
    param_enum_info = []  # 枚举参数的映射信息（(enum_map, unique_enums)）

    # 预处理每个参数：划分区间并记录类型信息
    for values in independent_set:
        numeric_values, enum_map, unique_enums = enum_to_numeric(values)
        param_enum_info.append((enum_map, unique_enums))
        ranges = divide_parameter_ranges(values, k)
        param_ranges.append(ranges)

    # 生成样本：第i个样本选择第i个区间（分散策略）
    samples = []
    for i in range(k):
        sample = []
        for p_idx, ranges in enumerate(param_ranges):
            range_idx = i % len(ranges)  # 循环选择区间，确保每个区间被覆盖
            current_range = ranges[range_idx]

            # 解析区间边界（处理连续/离散/枚举类型）
            if isinstance(current_range, tuple):
                r_start, r_end = current_range  # 连续值区间
            elif isinstance(current_range, list):
                r_start = r_end = current_range[0]  # 离散/枚举单值区间
            else:
                r_start = r_end = current_range  # 异常处理

            enum_map, unique_enums = param_enum_info[p_idx]

            # 确保边界是数值类型
            if isinstance(r_start, str) or isinstance(r_end, str):
                # 如果边界是字符串，转换为数值
                r_start = enum_map.get(r_start, 0)
                r_end = enum_map.get(r_end, len(unique_enums) - 1)
            # 在数值区间内采样
            val = random.uniform(float(r_start), float(r_end))
            # 将采样值映射回枚举值
            sample.append(numeric_to_enum(val, unique_enums))

        samples.append(sample)
    return samples


def get_bounded_space(current_best, independent_set, other_samples, expand_factor=1):
    """确定RBS有界搜索空间（聚焦当前最优配置周围）
    论文4.2节：通过左右边界限制搜索范围，平衡局部探索与全局搜索
    expand_factor：边界扩展系数（用于无改进时扩大搜索范围）
    """
    bounded_space = []
    for p_idx, (current_val, param_values) in enumerate(zip(current_best, independent_set)):
        _, enum_map, unique_enums = enum_to_numeric(param_values)
        current_numeric = enum_map[current_val]
        other_numerics = [enum_map[s[p_idx]] for s in other_samples if s[p_idx] in enum_map]
        sorted_others = sorted(other_numerics)
        global_min, global_max = 0, len(unique_enums) - 1

        # 计算左右边界（论文4.2节定义）
        left = -float('inf')
        right = float('inf')
        for v in sorted_others:
            if current_numeric > v > left:
                left = v  # 左边界：小于当前值的最大值
            if current_numeric < v < right:
                right = v  # 右边界：大于当前值的最小值

        # 边界有效性处理（若无参考样本，使用全局范围）
        left = left if left != -float('inf') else global_min
        right = right if right != float('inf') else global_max

        # 边界扩展（论文4.2节递归逻辑：无改进时扩大范围）
        left = left - (current_numeric - left) * expand_factor if left != global_min else global_min
        right = right + (right - current_numeric) * expand_factor if right != global_max else global_max

        # 生成边界内的候选值
        candidates = [enum for enum in unique_enums if left < enum_map[enum] < right]
        candidates = candidates if candidates else unique_enums  # 兜底
        bounded_space.append(candidates)
    return bounded_space


def run_tuners(file ,  budget=20, seed=0):
    """BestConfig主函数（论文3.2节闭环流程）"""
    total_rounds = 3
    random.seed(seed)
    np.random.seed(seed)

    # 读取数据集（论文3.1节：输入为配置-性能映射表）
    #file = get_data(filename)
    independent_set = file.independent_set  # 参数取值范围
    dict_search = file.dict_search  # 配置-性能映射

    # 初始化调优变量
    best_result = float('inf')
    best_config = None
    best_loop = 0
    xs = []  # 所有尝试的配置
    results = []  # 对应性能
    used_budget = 0
    expand_factor = 1  # RBS边界扩展系数（初始为1，无改进时增大）

    # 修正后：确保总和等于budget
    sum_ratio = sum(range(1, total_rounds + 1))  # 3轮时为6
    base_samples = []
    remaining = budget  # 剩余预算，用于分配误差

    # 先按比例分配整数部分
    for i in range(total_rounds):
        ratio = (total_rounds - i) / sum_ratio
        base = int(budget * ratio)
        base = max(1, base)  # 每轮至少1个样本
        base_samples.append(base)
        remaining -= base

    # 将剩余预算分配给最后一轮（或按比例分配）
    if remaining > 0:
        base_samples[-1] += remaining  # 最后一轮多分配剩余部分

    # 确保总采样数不超过预算（兜底）
    samples_per_round = base_samples

    # 多轮迭代：采样-评估-优化
    for round_idx in range(total_rounds):
        improved_in_round = False
        if used_budget >= budget:
            break
        current_samples_num = samples_per_round[round_idx]
        if current_samples_num <= 0:
            continue

        # 确定采样空间
        if round_idx == 0:
            # 第一轮：全局DDS采样（覆盖整个空间）
            current_samples = generate_dds_samples(independent_set, current_samples_num)
        else:
            # 后续轮次：RBS有界采样（聚焦当前最优）
            if best_config is None:
                current_samples = generate_dds_samples(independent_set, current_samples_num)
            else:
                # 基于当前扩展系数生成有界空间
                bounded_space = get_bounded_space(best_config, independent_set, xs, expand_factor)
                current_samples = generate_dds_samples(bounded_space, current_samples_num)

        # 评估当前轮次样本
        for config in current_samples:
            if used_budget >= budget:
                break
            score, _ = get_objective_score_with_similarity(dict_search, config)
            xs.append(config)
            results.append(score)
            used_budget += 1

            # 更新最优配置
            if score < best_result:
                best_result = score
                best_config = config
                best_loop = used_budget
                expand_factor = 1  # 找到更优配置时重置扩展系数
                improved_in_round = True
            # 输出过程信息
            print(f"预算消耗: {used_budget}/{budget} | 本轮最优: {score:.4f} | 全局最优: {best_result:.4f}")

        # 替换原有的边界扩展判断
        if not improved_in_round:
            expand_factor *= 2
            print(f"第{round_idx + 1}轮无改进，扩展搜索边界（系数={expand_factor}）")
        else:
            # 有改进则保持扩展系数不变
            print(f"第{round_idx + 1}轮找到更优结果，重置搜索边界（系数={expand_factor}）")
    return xs, results, range(1, used_budget + 1), best_result, best_loop, used_budget