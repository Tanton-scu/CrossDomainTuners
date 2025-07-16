from util.SaveToCSV import run_with_budget
from ReadDataset import get_data
from generalTuners import RandomSearch,BruteForce,SA,HillClimbing,GA,BO,ES,CMAES,DE,PSO,TabuSearch
from systemTuners import BestConfig
from algorithmTuners import HABO,GGA,ParamILS,SMAC,DEHB,HB
if __name__ == "__main__":
    kwargs = {
        "budget": 200,
        "seed": 1,
        "filename": "D:/MyProj/PythonProj/CrossDomainTuners/code/algorithmDatasets/CNN_cifar10.csv",
        #"filename": "D:/MyProj/PythonProj/CrossDomainTuners/code/systemDatasets/7z.csv",
        #"filename": "D:/MyProj/PythonProj/CrossDomainTuners/code/systemDatasets/PostgreSQL.csv",
    }
    #D:/MyProj/PythonProj/CrossDomainTuners/code/systemDatasets/7z.csv
    #D:/MyProj/PythonProj/CrossDomainTuners/code/algorithmDatasets/CNN_cifar10.csv
    # 在 main 函数中读取数据
    file = get_data(kwargs["filename"])
    kwargs["file"] = file  # 将读取的数据添加到 kwargs 中
    result = run_with_budget(HB.run_hyperband, **kwargs)
    print(f"调优结果: {result}")