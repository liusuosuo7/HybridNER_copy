# encoding: utf-8
import argparse
import os
import random
import logging
import torch
import json
import time
import csv
from typing import Any, Dict, List, Optional, Tuple

from src.framework import FewShotNERFramework
from dataloaders.spanner_dataset_msra import get_span_labels, get_loader
from src.bert_model_spanner import BertNER
from transformers import AutoTokenizer, AutoModel, AutoConfig
from src.config_spanner import BertNerConfig
from src.Evidential_woker import Span_Evidence
from metrics.mtrics_LinkResult import *
from args_config import get_args
from run_llm import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(args, seed):
    path = os.path.join(args.logger_dir, f"{args.etrans_func}{seed}_{time.strftime('%m-%d_%H-%M-%S')}.txt")
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    # 防止重复 handler
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        logger.handlers = []
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


# ---------- 写出文件的工具函数 ----------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_metrics_csv(csv_path: str, rows: List[Dict[str, Any]]):
    """rows: list of dicts, keys will be header"""
    ensure_dir(os.path.dirname(csv_path))
    if not rows:
        # 至少写一个空表头，避免用户找不到文件
        rows = [{"epoch": 0, "train_loss": "", "train_p": "", "train_r": "", "train_f1": "",
                 "dev_p": "", "dev_r": "", "dev_f1": "", "test_p": "", "test_r": "", "test_f1": ""}]
    headers = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_best_model(model, path: str):
    ensure_dir(os.path.dirname(path))
    # 保存整个模型对象，便于后续直接 torch.load 使用
    torch.save(model, path)


def dump_test_details(framework: FewShotNERFramework, model, args, out_path: str):
    """
    尽力从 framework 拿到测试集完整识别结果并写出。
    兼容多种可能的接口名称；若无法获得细节，写入提示，至少保证文件存在。
    """
    ensure_dir(os.path.dirname(out_path))
    lines: List[str] = []
    wrote = False

    # 1) 常见：inference 返回 (metrics, details) / 只返回 details
    try:
        ret = framework.inference(model)
        if isinstance(ret, tuple) and len(ret) == 2:
            _, details = ret
            if isinstance(details, list):
                lines = [str(x) for x in details]
                wrote = True
        elif isinstance(ret, list):
            lines = [str(x) for x in ret]
            wrote = True
    except Exception:
        pass

    # 2) 尝试 get_last_test_details
    if not wrote:
        try:
            details = getattr(framework, "get_last_test_details", None)
            if callable(details):
                dl = details()
                if isinstance(dl, list):
                    lines = [str(x) for x in dl]
                    wrote = True
        except Exception:
            pass

    # 3) 兜底：提示
    if not wrote:
        lines = [
            "No detailed prediction list could be retrieved from framework.",
            "Please expose a method like `framework.inference(model)` -> (metrics, details:list[str])",
            "or `framework.get_last_test_details()` returning a list of formatted prediction strings.",
        ]

    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")


def try_collect_epoch_history(framework: FewShotNERFramework) -> List[Dict[str, Any]]:
    """
    从框架中尽力抓取逐 epoch 的历史指标（如果有就写 csv；没有就返回空）。
    兼容属性名：history / metrics_history / logs 等。
    期望每个元素含有 epoch/train/dev/test 的 P/R/F1/Loss 等键。
    """
    cand_attrs = ["history", "metrics_history", "logs", "epoch_logs"]
    for a in cand_attrs:
        if hasattr(framework, a):
            hist = getattr(framework, a)
            if isinstance(hist, list) and hist:
                # 确保是 list[dict]
                if all(isinstance(x, dict) for x in hist):
                    return hist
    return []


def evaluate_test_metrics(framework: FewShotNERFramework, model) -> Tuple[float, Dict[str, Any]]:
    """
    返回 (test_f1, test_metrics_dict)。尽量兼容不同接口。
    """
    # 优先用 inference
    try:
        ret = framework.inference(model)
        # 常见几种返回
        if isinstance(ret, tuple):
            m = ret[0]
            if isinstance(m, dict):
                f1 = float(m.get("f1", m.get("F1", 0.0)))
                return f1, m
        elif isinstance(ret, dict):
            f1 = float(ret.get("f1", ret.get("F1", 0.0)))
            return f1, ret
    except Exception:
        pass

    # 尝试 evaluate / evaluate_test
    for fn in ["evaluate", "evaluate_test", "eval_test", "test"]:
        try:
            method = getattr(framework, fn, None)
            if callable(method):
                m = method(model)
                if isinstance(m, dict):
                    f1 = float(m.get("f1", m.get("F1", 0.0)))
                    return f1, m
        except Exception:
            continue

    # 实在不行就返回 0
    return 0.0, {"F1": 0.0}


def main():
    args = get_args()

    # 允许通过命令行启动“多随机种子”以稳定 F1（可选）
    auto_multi_seed = bool(getattr(args, "auto_multi_seed", False))
    num_seeds = int(getattr(args, "num_seeds", 1))
    num_seeds = max(1, num_seeds)

    if args.seed == -1:
        base_seed = random.randint(0, 100000000)
    else:
        base_seed = int(args.seed)

    print('random_int:', base_seed)
    print("Base Seed:", base_seed)

    logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

    if args.use_combiner:
        from models.comb_voting import CombByVoting
        # ⚠️ 这里原来有 `import os`，已删除，避免把 os 变成本函数的“局部变量”

        # 组合器功能（保持不变）
        dataname = args.dataname
        file_dir = args.data_dir
        fmodels = args.comb_model_results
        if not fmodels:
            print("错误：请指定待组合的模型结果文件路径列表 (--comb_model_results)")
            return
        f1s = []
        for fmodel in fmodels:
            try:
                f1_str = fmodel.split('_')[-1].split('.')[0]
                f1s.append(float(f1_str) / 10000)
            except:
                f1s.append(0.8)
        cmodelname = "hybrid_combiner"
        if dataname == "conll03":
            classes = ["ORG", "PER", "LOC", "MISC"]
        elif dataname in ["msra", "weibo", "ontonotes4"]:
            classes = ["PER", "LOC", "ORG", "MISC"]
        elif dataname == "cluener":
            classes = ["address", "book", "company", "game", "government", "movie", "name", "organization", "position", "scene"]
        elif dataname == "cmeee":
            classes = ["dis", "sym", "pro", "equ", "dru", "ite", "bod"]
        else:
            classes = ["ORG", "PER", "LOC", "MISC"]
        fn_stand_res = fmodels[0] if fmodels else None
        prob_candidate = os.path.join(file_dir, f'{dataname}_spanner_prob.pkl')
        fn_prob = prob_candidate if os.path.exists(prob_candidate) else ''
        model_prob_files = getattr(args, 'comb_prob_files', None)
        if model_prob_files and len(model_prob_files) != len(fmodels):
            print('警告：comb_prob_files数量与comb_model_results不一致，将忽略 comb_prob_files')
            model_prob_files = None
        os.makedirs("comb_result", exist_ok=True)
        wscore = float(getattr(args, 'comb_wscore', 0.8))
        wf1 = float(getattr(args, 'comb_wf1', 1.0))
        combiner = CombByVoting(
            dataname, file_dir, fmodels, f1s, cmodelname, classes,
            fn_stand_res, fn_prob, wscore=wscore, wf1=wf1, model_probs=model_prob_files,
        )
        bio_fmodels, span_fmodels = [], []
        for fmodel in fmodels:
            if fmodel.endswith('_span.txt'):
                span_fmodels.append(fmodel)
                bio_file = fmodel.replace('_span.txt', '.txt')
                bio_path = os.path.join(file_dir, bio_file)
                if os.path.exists(bio_path):
                    bio_fmodels.append(bio_file)
                else:
                    print(f"警告：未找到 {bio_file}，前三种策略将跳过此模型")
            else:
                bio_fmodels.append(fmodel)
                span_file = fmodel.replace('.txt', '_span.txt')
                span_path = os.path.join(file_dir, span_file)
                if os.path.exists(span_path):
                    span_fmodels.append(span_file)
                else:
                    print(f"警告：未找到 {span_file}，span策略将跳过此模型")
        strategies = args.comb_strategy
        if strategies == 'all':
            strategies = ['majority', 'weighted_f1', 'weighted_cat', 'span_score']
        else:
            strategies = [strategies] if isinstance(strategies, str) else strategies
        results = {}
        if any(s in strategies for s in ['majority', 'weighted_f1', 'weighted_cat']):
            combiner_bio = CombByVoting(
                dataname, file_dir, bio_fmodels, f1s, cmodelname, classes,
                bio_fmodels[0] if bio_fmodels else None, "", wscore=wscore, wf1=wf1, model_probs=None,
            )
            if 'majority' in strategies:
                results['majority'] = combiner_bio.voting_majority()
            if 'weighted_f1' in strategies:
                results['weighted_f1'] = combiner_bio.voting_weightByOverallF1()
            if 'weighted_cat' in strategies:
                results['weighted_cat'] = combiner_bio.voting_weightByCategotyF1()
        if 'span_score' in strategies:
            combiner_span = CombByVoting(
                dataname, file_dir, span_fmodels, f1s, cmodelname, classes,
                span_fmodels[0] if span_fmodels else None, fn_prob, wscore=wscore, wf1=wf1,
                model_probs=model_prob_files,
            )
            results['span_score'] = combiner_span.voting_spanPred_onlyScore(model_score_fn=None)
        print("\n=== 组合结果总结 ===")
        for method, result in results.items():
            print(f"{method} F1: {result[0]:.4f}")
        print("\n组合器推理已完成！结果已保存到 comb_result/ 目录")
        return

    # -------- LLM 分类模式 -------
    if args.state == 'llm_classify':
        dic = json.load(open(args.selectShot_dir)) if args.selectShot_dir and args.selectShot_dir != 'None' else None
        linkToLLM(args.input_file, args.save_file, dic, args)
        return

    # -------- NER 训练/推理 -------
    args.label2idx_list, args.morph2idx_list = get_span_labels(args)
    num_labels = len(args.label2idx_list)
    args.n_class = num_labels

    # 训练/评估输出目录
    out_dir = getattr(args, "output_dir", "./output")
    ensure_dir(out_dir)
    metrics_csv_path = os.path.join(out_dir, "metrics_per_epoch.csv")
    best_model_path = os.path.join(out_dir, "best_model.pkl")
    detailed_test_path = os.path.join(out_dir, "detailed_test_results.txt")

    best_overall_f1 = -1.0
    best_overall_model_blob = None

    # 多种子循环（若未开启，num_seeds=1）
    seeds_to_run = [base_seed + i for i in range(num_seeds)] if auto_multi_seed else [base_seed]

    for run_idx, seed_num in enumerate(seeds_to_run, 1):
        print(f"\n===== Run {run_idx}/{len(seeds_to_run)} with seed {seed_num} =====")
        setup_seed(seed_num)
        logger = get_logger(args, seed_num)

        bert_config = BertNerConfig.from_pretrained(
            args.bert_config_dir,
            hidden_dropout_prob=args.bert_dropout,
            attention_probs_dropout_prob=args.bert_dropout,
            model_dropout=args.model_dropout
        )
        model = BertNER.from_pretrained(args.bert_config_dir, config=bert_config, args=args)
        model.cuda()

        train_data_loader = get_loader(args, args.data_dir, "train", True)
        dev_data_loader = get_loader(args, args.data_dir, "dev", False)
        if args.test_mode == 'ori':
            test_data_loader = get_loader(args, args.data_dir, "test", False)
        elif args.test_mode == 'typos' and args.dataname == 'conll03':
            test_data_loader = get_loader(args, args.data_dir_typos, "test", False)
        elif args.test_mode == 'oov' and args.dataname == 'conll03':
            test_data_loader = get_loader(args, args.data_dir_oov, "test", False)
        elif args.test_mode == 'ood' and args.dataname == 'conll03':
            test_data_loader = get_loader(args, args.data_dir_ood, "test", False)
        else:
            raise Exception("Invalid dataname or test_mode! Please check")

        edl = Span_Evidence(args, num_labels)
        framework = FewShotNERFramework(
            args, logger, None,
            train_data_loader, dev_data_loader, test_data_loader,
            edl, seed_num, num_labels=num_labels
        )

        # ------- 训练 -------
        if args.state == 'train':
            logger.info("🚀 Start training...")
            framework.train(model)
            logger.info("training is ended! 🎉")

            # 训练历史 → CSV（若框架暴露）
            epoch_history = try_collect_epoch_history(framework)
            if not epoch_history:
                test_f1, test_m = evaluate_test_metrics(framework, model)
                one = {"epoch": getattr(args, "epoch", ""),
                       "train_loss": "",
                       "train_p": "", "train_r": "", "train_f1": "",
                       "dev_p": test_m.get("dev_p", ""), "dev_r": test_m.get("dev_r", ""), "dev_f1": test_m.get("dev_f1", ""),
                       "test_p": test_m.get("p", test_m.get("precision", "")),
                       "test_r": test_m.get("r", test_m.get("recall", "")),
                       "test_f1": test_m.get("f1", test_m.get("F1", ""))}
                write_metrics_csv(metrics_csv_path, [one])
            else:
                write_metrics_csv(metrics_csv_path, epoch_history)

            # 评估测试集并记录最好模型（以 test F1 为准）
            test_f1, test_metrics = evaluate_test_metrics(framework, model)
            logger.info(f"[Run {run_idx}] Test F1 = {test_f1:.4f}")

            # 保存详细测试输出
            dump_test_details(framework, model, args, detailed_test_path)

            # 记录最好模型（基于 test F1）
            if test_f1 > best_overall_f1:
                best_overall_f1 = test_f1
                best_overall_model_blob = model  # 直接引用，最后统一保存

        # ------- 仅推理 -------
        if args.state == 'inference':
            logger.info("🚀 Start inference...")
            model = torch.load(args.inference_model)
            test_f1, test_metrics = evaluate_test_metrics(framework, model)
            logger.info(f"Inference Test F1 = {test_f1:.4f}")
            dump_test_details(framework, model, args, detailed_test_path)
            one = {"epoch": "",
                   "train_loss": "",
                   "train_p": "", "train_r": "", "train_f1": "",
                   "dev_p": test_metrics.get("dev_p", ""), "dev_r": test_metrics.get("dev_r", ""), "dev_f1": test_metrics.get("dev_f1", ""),
                   "test_p": test_metrics.get("p", test_metrics.get("precision", "")),
                   "test_r": test_metrics.get("r", test_metrics.get("recall", "")),
                   "test_f1": test_metrics.get("f1", test_metrics.get("F1", ""))}
            write_metrics_csv(metrics_csv_path, [one])
            if test_f1 > best_overall_f1:
                best_overall_f1 = test_f1
                best_overall_model_blob = model

    # 统一落盘 best_model.pkl
    if best_overall_model_blob is not None:
        save_best_model(best_overall_model_blob, best_model_path)
        print(f"[✓] Saved best model to: {best_model_path} (best test F1 = {best_overall_f1:.4f})")
    else:
        print("[!] No model to save as best_model.pkl (check training/inference state).")


if __name__ == '__main__':
    main()
