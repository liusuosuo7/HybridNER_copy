#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试HybridNER组合器功能的脚本
"""

import os
import sys
import tempfile
import shutil


def create_test_data():
    """创建测试用的模型结果文件"""
    test_dir = "test_combiner_data"
    os.makedirs(test_dir, exist_ok=True)

    # 创建测试用的模型结果文件
    test_files = {
        "model1_9201.txt": """EU    B-ORG    B-ORG
rejects    O    O
German    B-MISC    B-MISC
call    O    O
to    O    O
boycott    O    O
British    B-MISC    B-MISC
lamb    O    O
.    O    O

Peter    B-PER    B-PER
Blackburn    I-PER    I-PER
""",
        "model2_9246.txt": """EU    B-ORG    B-ORG
rejects    O    O
German    B-MISC    B-MISC
call    O    O
to    O    O
boycott    O    O
British    B-MISC    B-MISC
lamb    O    O
.    O    O

Peter    B-PER    B-PER
Blackburn    I-PER    I-PER
""",
        "model3_9302.txt": """EU    B-ORG    B-ORG
rejects    O    O
German    B-MISC    B-MISC
call    O    O
to    O    O
boycott    O    O
British    B-MISC    B-MISC
lamb    O    O
.    O    O

Peter    B-PER    B-PER
Blackburn    I-PER    I-PER
"""
    }

    for filename, content in test_files.items():
        with open(os.path.join(test_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)

    return test_dir


def test_combiner():
    """测试组合器功能"""
    print("=== 测试HybridNER组合器功能 ===")

    # 创建测试数据
    test_dir = create_test_data()
    print(f"创建测试数据目录: {test_dir}")

    # 测试组合器功能
    try:
        from models.comb_voting import CombByVoting

        dataname = "conll03"
        file_dir = test_dir
        fmodels = ["model1_9201.txt", "model2_9246.txt", "model3_9302.txt"]
        f1s = [0.9201, 0.9246, 0.9302]
        cmodelname = "test_combiner"
        classes = ["ORG", "PER", "LOC", "MISC"]
        fn_stand_res = "model1_9201.txt"
        fn_prob = None

        print("创建组合器实例...")
        combiner = CombByVoting(dataname, file_dir, fmodels, f1s, cmodelname, classes, fn_stand_res, fn_prob)

        print("测试多数投票策略...")
        result = combiner.voting_majority()
        print(f"多数投票结果: F1={result[0]:.4f}, P={result[1]:.4f}, R={result[2]:.4f}")

        print("测试整体F1加权策略...")
        result = combiner.voting_weightByOverallF1()
        print(f"整体F1加权结果: F1={result[0]:.4f}, P={result[1]:.4f}, R={result[2]:.4f}")

        print("测试类别F1加权策略...")
        result = combiner.voting_weightByCategotyF1()
        print(f"类别F1加权结果: F1={result[0]:.4f}, P={result[1]:.4f}, R={result[2]:.4f}")

        print("测试Span预测分数策略...")
        result = combiner.voting_spanPred_onlyScore()
        print(f"Span预测分数结果: F1={result[0]:.4f}, P={result[1]:.4f}, R={result[2]:.4f}")

        print("✅ 组合器功能测试通过!")

    except Exception as e:
        print(f"❌ 组合器功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试数据
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"清理测试数据目录: {test_dir}")

    return True


def test_framework_integration():
    """测试框架集成"""
    print("\n=== 测试框架集成 ===")

    try:
        # 模拟参数
        class MockArgs:
            def __init__(self):
                self.dataname = "conll03"
                self.data_dir = "test_combiner_data"
                self.comb_result_dir = "test_comb_result"

        # 创建测试数据
        test_dir = create_test_data()

        # 模拟框架
        from src.framework import FewShotNERFramework
        import logging

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # 创建模拟框架实例
        args = MockArgs()
        framework = FewShotNERFramework(
            args, logger, None, None, None, None, None, 123, 5
        )

        # 测试组合器方法
        model_results = ["model1_9201.txt", "model2_9246.txt", "model3_9302.txt"]
        results = framework.run_combiner(model_results, strategy='majority')

        if results and 'majority' in results:
            print(f"✅ 框架集成测试通过! 结果: {results['majority'][0]:.4f}")
        else:
            print("❌ 框架集成测试失败: 没有返回结果")
            return False

    except Exception as e:
        print(f"❌ 框架集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试数据
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        if os.path.exists("test_comb_result"):
            shutil.rmtree("test_comb_result")

    return True


def main():
    """主函数"""
    print("开始测试HybridNER组合器功能...")

    # 测试组合器核心功能
    if not test_combiner():
        print("组合器核心功能测试失败!")
        return False

    # 测试框架集成
    if not test_framework_integration():
        print("框架集成测试失败!")
        return False

    print("\n🎉 所有测试通过! HybridNER组合器功能正常工作!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
