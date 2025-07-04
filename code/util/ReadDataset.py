"""
该函数读取 CSV 文件，
将数据划分为独立变量（配置组合）和依赖变量（性能指标），
并进行去重、排序等处理，
最终返回一个包含训练集、测试集等信息的file_data对象。
"""
import pandas as pd
import random
#在机器学习或优化问题中，每一个样本（或配置）可以看作是对问题的一个尝试性解决方案，这个类的作用就是封装这些解决方案的相关信息。
class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions#“decisions” 表示决策变量。在优化问题中，决策变量是可以控制和调整的参数，通过改变这些参数的值，可以得到不同的解决方案。在代码里，decisions 存储的是自变量的值，这些自变量的值就是我们为了找到最优解而可以做出的决策。
        self.objective = objective#“objective” 表示目标变量。在优化问题中，目标变量是我们希望优化的指标，通常是一个需要最小化或最大化的值。在代码里，objective 存储的是因变量的值，这些因变量的值就是我们优化的目标。
        self.rank = rank

class file_data:
    def __init__(self, name, training_set, testing_set, all_set, independent_set, features, dict_search):
        self.name = name#数据文件的名称，用于标识数据来源。
        self.training_set = training_set#训练集数据，包含了用于模型训练的样本。
        self.testing_set = testing_set#测试集数据，包含了用于评估模型性能的样本。
        self.all_set = all_set#所有数据样本的集合，包含了训练集和测试集的所有样本。
        self.independent_set = independent_set#自变量的值，存储了数据集中所有自变量的取值情况。
        self.features = features#特征列表，存储了数据集中所有特征的名称。其实就是所有自变量列的名称呗
        self.dict_search = dict_search#一个字典，用于存储决策变量（自变量）和目标值（因变量的最后一个值）之间的映射关系，方便根据决策变量的值快速查找对应的目标值。

def sort_and_deduplicate_columns(pdcontent, indepcolumns):
    """
    对自变量列进行排序和去重
    """
    tmp_sortindepcolumns = []#初始化一个空列表 tmp_sortindepcolumns：用于存储每个自变量列去重和排序后的结果。
    for col in indepcolumns:#循环遍历 indepcolumns 列表中的每个自变量列名。
        tmp_sortindepcolumns.append(sorted(list(set(pdcontent[col]))))
    return tmp_sortindepcolumns

def create_ranks(sortpdcontent, depcolumns):
    """
    创建目标值的排名字典
    """
    ranks = {}
    for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):
        ranks[item] = i
    return ranks

def create_content(sortpdcontent, indepcolumns, depcolumns, ranks):
    """
    创建 solution_holder 对象列表
    """
    content = []
    for c in range(len(sortpdcontent)):
        content.append(solution_holder(
            c,#c 作为样本的唯一标识符，即当前行的索引
            sortpdcontent.iloc[c][indepcolumns].tolist(),#提取当前行中所有自变量列的值，并将其转换为列表
            sortpdcontent.iloc[c][depcolumns].tolist(),#提取当前行中所有因变量列的值，并将其转换为列表
            ranks[sortpdcontent.iloc[c][depcolumns].tolist()[-1]]#根据当前行最后一个因变量的值，从 ranks 字典中获取对应的排名
        ))
    return content

def get_data(filename, initial_size=5, verbose=False):
    """
    :param filename: 文件名，用于指定数据来
    :param initial_size: 初始训练集大小，该参数用于划分训练集和测试集
    :param verbose: 是否打印详细信息，如果为 True，则会打印去重排序结果和读取数据完成的提示；如果为 False，则不打印这些信息
    :return: file
    """
    try:
        # 读取文件，使用 pandas 库的 read_csv 函数读取指定的 CSV 文件，并将其存储在 pdcontent 变量中。pdcontent 是一个 DataFrame 对象，用于存储表格数据。
        pdcontent = pd.read_csv(filename)

        # 找出以 +$< 开头的列
        columns_to_invert = [col for col in pdcontent.columns if col.startswith('+$<')]
        # 对这些列的数值取反
        for col in columns_to_invert:
            pdcontent[col] = -pdcontent[col]
        # 区分自变量和因变量
        indepcolumns = [col for col in pdcontent.columns if "$<" not in col]
        depcolumns = [col for col in pdcontent.columns if "$<" in col]

        # 对自变量列进行排序和去重，函数的作用是去除自变量列中的重复值，并对结果进行排序，方便查看参数的取值范围
        tmp_sortindepcolumns = sort_and_deduplicate_columns(pdcontent, indepcolumns)
        if verbose:
            print("去重排序：", tmp_sortindepcolumns)

        # 按目标从小到大排序，对 pdcontent 中的数据按照最后一个因变量列的值进行升序排序。排序后的结果存储在 sortpdcontent 中。
        sortpdcontent = pdcontent.sort_values(by=depcolumns[-1])
        # 创建目标值的排名字典，为每个不同的目标值创建一个排名。排名信息存储在 ranks 字典中，键为目标值，值为对应的排名。
        ranks = create_ranks(sortpdcontent, depcolumns)
        # 主要作用是将经过排序处理后的 DataFrame 对象 sortpdcontent 转换为一个 solution_holder 对象列表
        # 创建 solution_holder 对象列表，每个 solution_holder 对象包含一个样本的 ID、自变量值、因变量值和目标值的排名。
        content = create_content(sortpdcontent, indepcolumns, depcolumns, ranks)

        # 创建决策和目标值的映射字典
        #该字典用于建立决策变量（自变量）和目标值（因变量的最后一个值）之间的映射关系
        #[tuple(i.decision) for i in content]用于从 content 列表中的每个 solution_holder 对象里提取决策变量（自变量）的值，并将其转换为元组。
        # [i.objective[-1] for i in content]用于从 content 列表中的每个 solution_holder 对象里提取目标值（因变量的最后一个值）。
        #zip() 函数将这两个列表中的元素一一对应地组合成元组，例如，第一个决策变量的元组和第一个目标值组成一个元组，第二个决策变量的元组和第二个目标值组成一个元组，以此类推。
        dict_search = dict(zip([tuple(i.decision) for i in content], [i.objective[-1] for i in content]))

        # 打乱数据
        #使用 random 模块的 shuffle 函数对 content 列表进行随机打乱，以确保训练集和测试集的划分是随机的，避免数据的顺序对模型训练产生影响。
        random.shuffle(content)

        # 划分训练集和测试集
        indexes = range(len(content))
        #indexes[:initial_size] 会截取 indexes 序列中从开头到 initial_size - 1 的部分，这些索引对应的样本将构成训练集。
        #indexes[initial_size:] 会截取 indexes 序列中从 initial_size 到末尾的部分，这些索引对应的样本将构成测试集。
        train_indexes, test_indexes = indexes[:initial_size], indexes[initial_size:]
        #检查划分的正确性
        assert len(train_indexes) + len(test_indexes) == len(indexes), "Something is wrong"
        #[content[i] for i in train_indexes] 会遍历 train_indexes 中的每个索引 i，并将 content[i] 添加到 train_set 列表中。
        train_set = [content[i] for i in train_indexes]
        test_set = [content[i] for i in test_indexes]

        # 创建 file_data 对象
        file = file_data(filename, train_set, test_set,
                         content, tmp_sortindepcolumns, indepcolumns, dict_search)
        if verbose:
            print("完成读取数据")
        return file
    except FileNotFoundError:
        print(f"文件 {filename} 未找到，请检查文件路径。")
    except Exception as e:
        print(f"读取数据时发生错误: {e}")