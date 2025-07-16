import csv
import os
import inspect

def run_with_budget(tuner_function, **kwargs):
    """
    通用函数，用于运行调优器并在达到预算时结束
    :param tuner_function: 调优器函数
    :param kwargs: 调优器函数的参数
    :return: 调优器函数的返回结果
    """
    budget = kwargs.get('budget', 20)  # 默认预算为20
    seed = kwargs.get('seed', 0)  # 默认种子为0
    filename = kwargs.get('filename', '')
    file=kwargs.get('file', '')
    result = tuner_function(budget,seed,file)
    # 处理结果并保存到CSV文件
    if isinstance(result, tuple):
        if len(result) == 3:  # 例如 ConEx, sampling, random, GA, best_config, GGA, E_search, paramILS
            xs, results, used_budget = result
            x_axis = range(len(results) + 1)[1:]
            best_result = None
            best_loop = None
        elif len(result) == 6:  # 例如 boca, atconf, flash, ottertune, restune, robotune, smac, tuneful
            xs, results, x_axis, best_result, best_loop, used_budget = result

        # 获取调优器函数所在的文件名
        func_file = inspect.getfile(tuner_function)
        func_filename = os.path.splitext(os.path.basename(func_file))[0]
        # 修改保存路径
        save_dir = r'D:\MyProj\PythonProj\CrossDomainTuners\code\result'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 仅提取文件名，去掉后缀
        file_name_only = os.path.splitext(os.path.basename(filename))[0]
        csv_file_path = os.path.join(save_dir, f'{func_filename}_{file_name_only}_seed{seed}.csv')

    with open(csv_file_path, 'w', newline="") as f:
        csv_writer = csv.writer(f)
        # 写入表头（如果需要）
        #csv_writer.writerow(['轮次'] + ['配置' + str(i) for i in range(len(xs[0]))] + ['性能'])
        #目前就是轮次+配置+性能；最后一行复现最佳性能
        for i, (config, score) in enumerate(zip(xs, results)):
            #这两行代码把 config 列表里的所有元素以及 score 都转换为字符串类型。
            config_str = [str(c) for c in config]
            score_str = str(score)
            row = [str(i + 1)] + config_str + [score_str]
            csv_writer.writerow(row)

        # 在 SaveToCSV.py 文件中
        if best_loop is not None:
            if best_loop - 1 < len(xs):
                best_config = xs[best_loop - 1]
                best_score = results[best_loop - 1]
                # 将 best_loop、best_config 中的元素和 best_score 都转换为字符串类型
                best_loop_str = str(best_loop)
                best_config_str = [str(c) for c in best_config]
                best_score_str = str(best_score)
                last_row = [best_loop_str] + best_config_str + [best_score_str]
                csv_writer.writerow(last_row)
            else:
                print(f"Warning: best_loop ({best_loop}) is out of range of xs ({len(xs)}). Skipping best result row.")

    return result