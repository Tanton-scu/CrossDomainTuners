CrossDomainTuners: 跨领域参数调优框架
项目简介
CrossDomainTuners 是一个用于跨领域参数优化的 Python 框架，集成了多种先进的调优算法，支持从不同领域的数据集自动学习最优参数配置。
多算法支持：实现了 HABO、Random Search、BestConfig 等多种调优算法
跨领域适配：可处理连续型、离散型、布尔型、分类型等多种参数类型
自动参数类型推断：基于数据特征自动识别参数类型并优化搜索策略
相似度查询：高效查找与给定配置最相似的已知配置
结果持久化：自动将调优结果保存为 CSV 格式，便于后续分析
目录结构
plaintext
CrossDomainTuners/
├── code/                      # 核心代码
│   ├── algorithmDatasets/     # 算法调优数据集
│   ├── algorithmTuners/       # 算法调优器实现
│   ├── generalTuners/         # 通用调优器
│   ├── systemDatasets/        # 系统调优数据集
│   ├── systemTuners/          # 系统调优器实现
│   ├── result/                # 调优结果存储
│   ├── util/                  # 工具函数
│   └── main.py                # 程序入口
├── .idea/                     # IDE配置文件
└── .gitignore                 # Git忽略规则
安装依赖
bash
pip install numpy pandas matplotlib
使用方法
准备数据集：将数据集放置在code/algorithmDatasets或code/systemDatasets目录下
配置调优参数：在main.py中设置预算、随机种子等参数
选择调优器：默认使用 HABO 调优器，也可切换至其他调优器
运行调优：
运行main.py
查看结果：调优结果将保存在code/result目录下的 CSV 文件中
调优器说明
HABO (HyperArm Bandit Optimization)
基于层级化多臂老虎机的调优算法，通过 EXP3 机制自适应选择超参数及其取值，平衡探索与利用，特别适合高维参数空间优化。
Random Search
简单高效的随机搜索算法，作为基线方法用于对比评估。
BestConfig
基于相似度匹配的配置推荐算法，通过查找历史最优配置进行参数推荐。
