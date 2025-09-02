#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HybridNER 组合器功能示例脚本
这个脚本演示了如何使用HybridNER的组合器功能来组合多个NER模型的结果
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """运行命令并打印结果"""
    print(f"\n=== {description} ===")
    print(f"执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("执行成功!")
        if result.stdout:
            print("输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {e}")
        if e.stdout:
            print("标准输出:", e.stdout)
        if e.stderr:
            print("错误输出:", e.stderr)


def main():
    print("=== HybridNER 组合器功能示例 ===")

    # 设置基本参数
    dataname = "conll03"
    data_dir = "data/conll03"

    # 示例1: 仅运行组合器功能（不训练模型）
    print("\n示例1: 仅运行组合器功能")
    cmd1 = [
        "python", "main.py",
        "--dataname", dataname,
        "--data_dir", data_dir,
        "--use_combiner", "True",
        "--comb_model_results",
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt",
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt",
        "conll03_CbertWnon_lstmCrf_1_test_9246.txt",
        "conll03_CflairWglove_lstmCrf_1_test_9302.txt",
        "--comb_strategy", "all",
        "--comb_result_dir", "comb_result"
    ]
    run_command(cmd1, "仅运行组合器功能")

    # 示例2: 训练模型后自动运行组合器
    print("\n示例2: 训练模型后自动运行组合器")
    cmd2 = [
        "python", "main.py",
        "--dataname", dataname,
        "--data_dir", data_dir,
        "--state", "train",
        "--iteration", "5",
        "--batch_size", "32",
        "--use_combiner", "True",
        "--comb_model_results",
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt",
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt",
        "--comb_strategy", "majority",
        "--auto_combine_after_train", "True"
    ]
    run_command(cmd2, "训练模型后自动运行组合器")

    # 示例3: 推理后自动运行组合器
    print("\n示例3: 推理后自动运行组合器")
    cmd3 = [
        "python", "main.py",
        "--dataname", dataname,
        "--data_dir", data_dir,
        "--state", "inference",
        "--inference_model", "results/edl123123_model.pkl",
        "--use_combiner", "True",
        "--comb_model_results",
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt",
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt",
        "conll03_CbertWnon_lstmCrf_1_test_9246.txt",
        "--comb_strategy", "weighted_f1",
        "--auto_combine_after_inference", "True"
    ]
    run_command(cmd3, "推理后自动运行组合器")

    # 示例4: 使用不同的组合策略
    print("\n示例4: 使用不同的组合策略")

    strategies = [
        ("majority", "多数投票策略"),
        ("weighted_f1", "整体F1加权策略"),
        ("weighted_cat", "类别F1加权策略"),
        ("span_score", "Span预测分数策略")
    ]

    for strategy, description in strategies:
        cmd4 = [
            "python", "main.py",
            "--dataname", dataname,
            "--data_dir", data_dir,
            "--use_combiner", "True",
            "--comb_model_results",
            "conll03_CflairWnon_lstmCrf_1_test_9241.txt",
            "conll03_CbertWglove_lstmCrf_1_test_9201.txt",
            "--comb_strategy", strategy
        ]
        run_command(cmd4, f"4.{strategies.index((strategy, description)) + 1} {description}")

    print("\n=== 组合器功能使用说明 ===")
    print("1. 确保模型结果文件存在于指定的目录中")
    print("2. 文件名格式建议为: model_name_f1score.txt (如: model_9201.txt)")
    print("3. 系统会自动从文件名中提取F1分数")
    print("4. 组合结果会保存到 comb_result/ 目录")
    print("5. 支持多种组合策略: majority, weighted_f1, weighted_cat, span_score, all")
    print("6. 可以在训练或推理后自动运行组合器功能")


if __name__ == "__main__":
    main()
