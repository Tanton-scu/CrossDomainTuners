import pandas as pd
import random

class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank

class file_data:
    def __init__(self, name, training_set, testing_set, all_set, independent_set, features, dict_search):
        self.name = name
        self.training_set = training_set
        self.testing_set = testing_set
        self.all_set = all_set
        self.independent_set = independent_set
        self.features = features
        self.dict_search = dict_search

def sort_and_deduplicate_columns(pdcontent, indepcolumns):
    tmp_sortindepcolumns = []
    for col in indepcolumns:
        tmp_sortindepcolumns.append(sorted(list(set(pdcontent[col]))))
    return tmp_sortindepcolumns

def create_ranks(sortpdcontent, depcolumns):
    ranks = {}
    for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):
        ranks[item] = i
    return ranks

def create_content(sortpdcontent, indepcolumns, depcolumns, ranks):
    content = []
    for c in range(len(sortpdcontent)):
        content.append(solution_holder(
            c,
            sortpdcontent.iloc[c][indepcolumns].tolist(),
            sortpdcontent.iloc[c][depcolumns].tolist(),
            ranks[sortpdcontent.iloc[c][depcolumns].tolist()[-1]]
        ))
    return content

def get_data(filename, initial_size=5, verbose=False):
    try:
        # 读取文件
        pdcontent = pd.read_csv(filename)

        # 关键改动：仅保留前2000行（不足则全部保留）
        #pdcontent = pdcontent.head(2000)  # 添加此行，截取前2000行

        # 找出以 +$< 开头的列并取反
        columns_to_invert = [col for col in pdcontent.columns if col.startswith('+$<')]
        for col in columns_to_invert:
            pdcontent[col] = -pdcontent[col]

        # 区分自变量和因变量
        indepcolumns = [col for col in pdcontent.columns if "$<" not in col]
        depcolumns = [col for col in pdcontent.columns if "$<" in col]

        # 对自变量列进行排序和去重
        tmp_sortindepcolumns = sort_and_deduplicate_columns(pdcontent, indepcolumns)
        if verbose:
            print("去重排序：", tmp_sortindepcolumns)

        # 按目标从小到大排序
        sortpdcontent = pdcontent.sort_values(by=depcolumns[-1])
        # 创建目标值的排名字典
        ranks = create_ranks(sortpdcontent, depcolumns)
        # 创建 solution_holder 对象列表
        content = create_content(sortpdcontent, indepcolumns, depcolumns, ranks)

        # 创建决策和目标值的映射字典
        dict_search = dict(zip([tuple(i.decision) for i in content], [i.objective[-1] for i in content]))

        # 打乱数据
        random.shuffle(content)

        # 划分训练集和测试集
        indexes = range(len(content))
        train_indexes, test_indexes = indexes[:initial_size], indexes[initial_size:]
        assert len(train_indexes) + len(test_indexes) == len(indexes), "划分错误"
        train_set = [content[i] for i in train_indexes]
        test_set = [content[i] for i in test_indexes]

        # 创建 file_data 对象
        file = file_data(filename, train_set, test_set,
                         content, tmp_sortindepcolumns, indepcolumns, dict_search)
        if verbose:
            print(f"完成读取数据（共读取 {len(content)} 行）")
        return file
    except FileNotFoundError:
        print(f"文件 {filename} 未找到，请检查路径。")
    except Exception as e:
        print(f"读取数据时出错: {e}")