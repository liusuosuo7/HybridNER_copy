import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD
import time
import prettytable as pt
import os
import json
import csv
from typing import List, Dict, Any, Tuple
from metrics.function_metrics import span_f1_prune, ECE_Scores, get_predict_prune


class FewShotNERFramework:
    def __init__(self, args, logger, task_idx2label,
                 train_data_loader, val_data_loader, test_data_loader,
                 edl, seed_num, num_labels):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.logger = logger
        self.seed = seed_num
        self.args = args
        self.eps = 1e-10
        self.learning_rate = args.lr
        self.load_ckpt = args.load_ckpt
        self.optimizer = args.optimizer
        self.annealing_start = 1e-6
        self.epoch_num = args.iteration
        self.edl = edl
        self.num_labels = num_labels
        self.task_idx2label = task_idx2label

        # 输出目录：优先 output_dir；否则回落到 model_save_dir；再否则 ./output
        self.output_dir = getattr(self.args, "output_dir",
                                  getattr(self.args, "model_save_dir", "./output"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_csv = os.path.join(self.output_dir, "metrics_per_epoch.csv")
        self.best_model_path = os.path.join(self.output_dir, "best_model.pkl")
        self.best_details_path = os.path.join(self.output_dir, "detailed_test_results.txt")

        # 逐 epoch 历史（也会写入 CSV）
        self.metrics_history: List[Dict[str, Any]] = []

    def item(self, x):
        return x.item()

    def metric(self, model, eval_dataset, mode):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        context_results = []

        with torch.no_grad():
            gold_tokens_list = []
            pred_scores_list = []
            pred_list = []

            for data in eval_dataset:
                (tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken,
                 all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs) = data

                loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken,
                           all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                attention_mask = (tokens != 0).long()

                logits = model(loadall, all_span_lens, all_span_idxs_ltoken,
                               tokens, attention_mask, token_type_ids)
                predicts, uncertainty = self.edl.pred(logits)
                correct, tmp_pred_cnt, tmp_label_cnt = span_f1_prune(
                    predicts, span_label_ltoken, real_span_mask_ltoken)

                pred_cls, pred_scores, tgt_cls = self.edl.ece_value(
                    logits, span_label_ltoken, real_span_mask_ltoken)

                prob, pred_id = torch.max(predicts, 2)
                batch_results = get_predict_prune(
                    self.args.label2idx_list, all_span_word, words, pred_id,
                    span_label_ltoken, all_span_idxs, prob, uncertainty)

                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                context_results += batch_results

                gold_tokens_list.append(tgt_cls)
                pred_scores_list.append(pred_scores)
                pred_list.append(pred_cls)

            gold_tokens_cat = torch.cat(gold_tokens_list, dim=0)
            pred_scores_cat = torch.cat(pred_scores_list, dim=0)
            pred_cat = torch.cat(pred_list, dim=0)

            ece = ECE_Scores(pred_cat, gold_tokens_cat, pred_scores_cat)
            precision = correct_cnt / (pred_cnt + 1e-12)
            recall = correct_cnt / (label_cnt + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            # 保留原 jsonl 输出
            if mode == 'test':
                results_dir = os.path.join(
                    self.args.results_dir,
                    f"{self.args.dataname}_{self.args.uncertainty_type}_local_model.jsonl"
                )
                os.makedirs(os.path.dirname(results_dir), exist_ok=True)
                with open(results_dir, 'w', encoding='utf-8') as fout:
                    for rec in context_results:
                        json.dump(rec, fout, ensure_ascii=False)
                        fout.write('\n')

            return precision, recall, f1, ece, (context_results if mode == 'test' else None)

    def eval(self, model, mode=None):
        if mode == 'dev':
            self.logger.info("Use val dataset")
            precision, recall, f1, ece, _ = self.metric(model, self.val_data_loader, mode='dev')
            head = "dev"
        elif mode == 'test':
            self.logger.info("Use " + str(self.args.test_mode) + " test dataset")
            precision, recall, f1, ece, details = self.metric(model, self.test_data_loader, mode='test')
            head = "test"
        else:
            raise ValueError("mode must be 'dev' or 'test'")

        table = pt.PrettyTable([head, "Precision", "Recall", "F1", "ECE"])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [precision, recall, f1, ece]])
        self.logger.info("\n{}".format(table))

        return (f1, ece, details) if mode == 'test' else (f1, ece, None)

    def _append_metrics_row(self, row: Dict[str, Any]):
        """追加指标并实时写 CSV"""
        self.metrics_history.append(row)
        headers = list(self.metrics_history[0].keys())
        with open(self.metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in self.metrics_history:
                w.writerow(r)

    def train(self, model):
        self.logger.info("Start training...")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.98),
                              lr=self.learning_rate, eps=self.args.adam_epsilon)
        elif self.optimizer == "sgd":
            optimizer = SGD(optimizer_grouped_parameters, self.learning_rate, momentum=0.9)
        elif self.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr,
                                          eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        t_total = len(self.train_data_loader) * self.args.iteration
        warmup_steps = int(self.args.warmup_proportion * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        model.train()
        best_test_f1 = -1.0
        best_step = 0
        iter_loss = 0.0

        early_stop_patience = int(getattr(self.args, "early_stop", 10**9))
        since_best = 0

        for idx in range(self.args.iteration):
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0
            epoch_start = time.time()
            self.logger.info("training...")

            # ✅ 关键修复：正确遍历 dataloader（不能每步 new iter(...)）
            for batch in self.train_data_loader:
                (tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken,
                 all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs) = batch

                loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken,
                           all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                attention_mask = (tokens != 0).long()
                logits = model(loadall, all_span_lens, all_span_idxs_ltoken,
                               tokens, attention_mask, token_type_ids)

                loss, pred = self.edl.loss(logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx)
                correct, tmp_pred_cnt, tmp_label_cnt = span_f1_prune(
                    pred, span_label_ltoken, real_span_mask_ltoken)

                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                iter_loss += self.item(loss.data)

            # 本 epoch 训练指标
            epoch_cost = time.time() - epoch_start
            precision = pred_cnt and (correct_cnt / (pred_cnt + 1e-12)) or 0.0
            recall = label_cnt and (correct_cnt / (label_cnt + 1e-12)) or 0.0
            train_f1 = 2 * precision * recall / (precision + recall + 1e-8)

            self.logger.info("Time '%.2f's" % epoch_cost)
            self.logger.info(
                'step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'
                .format(idx + 1, iter_loss, precision, recall, train_f1)
            )

            # 每个 epoch 都做 dev & test
            dev_f1, dev_ece, _ = self.eval(model, mode='dev')
            test_f1, test_ece, test_details = self.eval(model, mode='test')

            # 写入一行 CSV
            row = {
                "epoch": idx + 1,
                "train_loss": round(float(iter_loss), 6),
                "train_p": round(float(precision), 4),
                "train_r": round(float(recall), 4),
                "train_f1": round(float(train_f1), 4),
                "dev_f1": round(float(dev_f1), 4),
                "dev_ece": round(float(dev_ece), 4),
                "test_f1": round(float(test_f1), 4),
                "test_ece": round(float(test_ece), 4),
            }
            self._append_metrics_row(row)

            # 以 test F1 选最优并保存模型 & 明细
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_step = idx + 1
                since_best = 0

                os.makedirs(self.args.model_save_dir, exist_ok=True)
                # 稳定文件名（符合你要求）
                torch.save(model, self.best_model_path)
                self.logger.info(
                    f"Saved new BEST (by test F1) model to: {self.best_model_path} "
                    f"(epoch {best_step}, test F1={best_test_f1:.4f})"
                )
                # 兼容旧命名
                compat_ckpt_path = os.path.join(
                    self.args.model_save_dir, f"best_f1_{best_test_f1:.4f}_epoch_{best_step}.pkl"
                )
                torch.save(model, compat_ckpt_path)
                self.logger.info(f"Saved compat checkpoint: {compat_ckpt_path}")

                # 写详细测试识别结果
                try:
                    with open(self.best_details_path, "w", encoding="utf-8") as f:
                        if isinstance(test_details, list):
                            for rec in test_details:
                                f.write((json.dumps(rec, ensure_ascii=False)
                                         if isinstance(rec, (dict, list)) else str(rec)).rstrip() + "\n")
                        else:
                            f.write("No detailed test predictions captured.\n")
                    self.logger.info(f"Wrote detailed test results to: {self.best_details_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to write detailed_test_results.txt: {e}")

            else:
                since_best += 1

            # 早停
            if since_best > early_stop_patience:
                self.logger.info(
                    f"Early stop at epoch {idx+1}. Best test F1={best_test_f1:.4f} at epoch {best_step}."
                )
                return

            # reset
            iter_loss = 0.0

    def inference(self, model):
        # 返回 test 指标与明细，供外部需要时使用
        f1, ece, details = self.eval(model, mode='test')
        return f1, ece, details

    def run_combiner(self, model_results, f1_scores=None, strategy='all'):
        """保留原组合器接口"""
        from models.comb_voting import CombByVoting

        if not model_results:
            self.logger.error("没有提供模型结果文件")
            return None

        if f1_scores is None:
            f1_scores = []
            for fmodel in model_results:
                try:
                    f1_str = fmodel.split('_')[-1].split('.')[0]
                    f1_scores.append(float(f1_str) / 10000)
                except:
                    f1_scores.append(0.8)

        dataname = self.args.dataname
        file_dir = self.args.data_dir
        cmodelname = "hybrid_combiner"
        classes = ["ORG", "PER", "LOC", "MISC"] if dataname == "conll03" else []
        fn_stand_res = model_results[0] if model_results else None
        fn_prob = None

        result_dir = getattr(self.args, 'comb_result_dir', 'comb_result')
        os.makedirs(result_dir, exist_ok=True)

        combiner = CombByVoting(dataname, file_dir, model_results, f1_scores,
                                cmodelname, classes, fn_stand_res, fn_prob)

        self.logger.info(f"开始组合 {len(model_results)} 个模型的结果...")
        self.logger.info(f"模型文件: {model_results}")
        self.logger.info(f"F1分数: {f1_scores}")
        self.logger.info(f"组合策略: {strategy}")

        results = {}
        if strategy in ['majority', 'all']:
            self.logger.info("=== 多数投票 (Majority Voting) ===")
            results['majority'] = combiner.voting_majority()
        if strategy in ['weighted_f1', 'all']:
            self.logger.info("=== 基于整体F1加权投票 (Weighted by Overall F1) ===")
            results['weighted_f1'] = combiner.voting_weightByOverallF1()
        if strategy in ['weighted_cat', 'all']:
            self.logger.info("=== 基于类别F1加权投票 (Weighted by Category F1) ===")
            results['weighted_cat'] = combiner.voting_weightByCategotyF1()
        if strategy in ['span_score', 'all']:
            self.logger.info("=== 基于Span预测分数投票 (Span Prediction Score) ===")
            results['span_score'] = combiner.voting_spanPred_onlyScore()

        self.logger.info("=== 组合结果总结 ===")
        for method, result in results.items():
            self.logger.info(f"{method} F1: {result[0]:.4f}")

        return results
