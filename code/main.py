from util.SaveToCSV import run_with_budget
from generalTuners import RandomSearch
from systemTuners import BestConfig
from algorithmTuners import HABO
if __name__ == "__main__":
    kwargs = {
        "budget": 30,
        "seed": 1,
        "filename": "D:/MyProj/PythonProj/CrossDomainTuners/code/algorithmDatasets/cifar10_modified.csv",
    }
    result = run_with_budget(HABO.run_tuners, **kwargs)
    print(f"调优结果: {result}")