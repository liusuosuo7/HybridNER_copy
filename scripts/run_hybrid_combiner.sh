#!/bin/bash

# HybridNER 组合器功能示例脚本
# 这个脚本演示了如何使用HybridNER的组合器功能来组合多个NER模型的结果

echo "=== HybridNER 组合器功能示例 ==="

# 设置基本参数
DATANAME="conll03"
DATA_DIR="data/conll03"
MODEL_RESULTS_DIR="results/conll03"

# 示例1: 仅运行组合器功能（不训练模型）
echo "示例1: 仅运行组合器功能"
python main.py \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --use_combiner True \
    --comb_model_results \
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt" \
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt" \
        "conll03_CbertWnon_lstmCrf_1_test_9246.txt" \
        "conll03_CflairWglove_lstmCrf_1_test_9302.txt" \
    --comb_strategy all \
    --comb_result_dir comb_result

echo ""

# 示例2: 训练模型后自动运行组合器
echo "示例2: 训练模型后自动运行组合器"
python main.py \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --state train \
    --iteration 5 \
    --batch_size 32 \
    --use_combiner True \
    --comb_model_results \
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt" \
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt" \
    --comb_strategy majority \
    --auto_combine_after_train True

echo ""

# 示例3: 推理后自动运行组合器
echo "示例3: 推理后自动运行组合器"
python main.py \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --state inference \
    --inference_model results/edl123123_model.pkl \
    --use_combiner True \
    --comb_model_results \
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt" \
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt" \
        "conll03_CbertWnon_lstmCrf_1_test_9246.txt" \
    --comb_strategy weighted_f1 \
    --auto_combine_after_inference True

echo ""

# 示例4: 使用不同的组合策略
echo "示例4: 使用不同的组合策略"

echo "4.1 多数投票策略"
python main.py \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --use_combiner True \
    --comb_model_results \
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt" \
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt" \
    --comb_strategy majority

echo "4.2 整体F1加权策略"
python main.py \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --use_combiner True \
    --comb_model_results \
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt" \
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt" \
    --comb_strategy weighted_f1

echo "4.3 类别F1加权策略"
python main.py \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --use_combiner True \
    --comb_model_results \
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt" \
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt" \
    --comb_strategy weighted_cat

echo "4.4 Span预测分数策略"
python main.py \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --use_combiner True \
    --comb_model_results \
        "conll03_CflairWnon_lstmCrf_1_test_9241.txt" \
        "conll03_CbertWglove_lstmCrf_1_test_9201.txt" \
    --comb_strategy span_score

echo ""
echo "=== 组合器功能使用说明 ==="
echo "1. 确保模型结果文件存在于指定的目录中"
echo "2. 文件名格式建议为: model_name_f1score.txt (如: model_9201.txt)"
echo "3. 系统会自动从文件名中提取F1分数"
echo "4. 组合结果会保存到 comb_result/ 目录"
echo "5. 支持多种组合策略: majority, weighted_f1, weighted_cat, span_score, all"
echo "6. 可以在训练或推理后自动运行组合器功能"
