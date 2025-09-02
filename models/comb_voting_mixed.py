# -*- coding: utf-8 -*

import numpy as np
import os
import pickle
import json
from .dataread import DataReader


def evaluate_chunk_level(pred_chunks, true_chunks):
    correct_preds, total_correct, total_preds = 0., 0., 0.
    correct_preds += len(set(true_chunks) & set(pred_chunks))
    total_preds += len(pred_chunks)
    total_correct += len(true_chunks)
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return f1, p, r, correct_preds, total_preds, total_correct


class CombByVotingMixed():
    """
    混合组合器：前三种策略用BIO格式文件，第四种策略用span格式文件
    """

    def __init__(self, dataname, file_dir, fmodels_bio, fmodels_span, f1s, cmodelname, classes, fn_stand_res, fn_prob,
                 wscore: float = 0.8, wf1: float = 1.0):
        self.dataname = dataname
        self.file_dir = file_dir
        self.fmodels_bio = fmodels_bio  # BIO格式文件列表，用于前三种策略
        self.fmodels_span = fmodels_span  # span格式文件列表，用于第四种策略
        self.f1s = f1s
        self.cmodelname = cmodelname
        self.fn_prob = fn_prob
        self.classes = classes

        # 为前三种策略创建DataReader（使用BIO文件）
        self.mres_bio = DataReader(dataname, file_dir, classes, fmodels_bio, fn_stand_res)

        # 为第四种策略创建DataReader（使用span文件）
        self.mres_span = DataReader(dataname, file_dir, classes, fmodels_span, fn_stand_res)

        # 组合权重：可通过外部传入调整
        self.wf1 = wf1
        self.wscore = wscore

    def get_unique_pchunk_labs_bio(self):
        """获取BIO文件的chunks（用于前三种策略）"""
        tchunks_models, \
            tchunks_unique, \
            pchunks_models, \
            tchunks_models_onedim, \
            pchunks_models_onedim, \
            pchunk2label_models, \
            tchunk2label_dic, \
            class2f1_models = self.mres_bio.get_allModels_pred()
        self.tchunks_unique = tchunks_unique
        self.class2f1_models = class2f1_models

        pchunk_plb_ms = []
        keep_pref_upchunks = []
        for ind_m, pchunk2label_model in enumerate(pchunk2label_models):
            pchunk_plb_m = []
            for ind_st, upchunk in enumerate(tchunks_unique):
                pref_upchunk = (upchunk[0], upchunk[1], upchunk[2])  # 包括sentid
                if pref_upchunk in pchunk2label_model:
                    pchunk_plb_m.append(pchunk2label_model[pref_upchunk])
                else:
                    pchunk_plb_m.append('O')
                if ind_m == 0:
                    keep_pref_upchunks.append(pref_upchunk)
            pchunk_plb_ms.append(pchunk_plb_m)

        assert len(pchunk_plb_ms) == len(self.fmodels_bio)
        return pchunk_plb_ms, keep_pref_upchunks

    def get_unique_pchunk_labs_span(self):
        """获取span文件的chunks（用于第四种策略）"""
        tchunks_models, \
            tchunks_unique, \
            pchunks_models, \
            tchunks_models_onedim, \
            pchunks_models_onedim, \
            pchunk2label_models, \
            tchunk2label_dic, \
            class2f1_models = self.mres_span.get_allModels_pred()

        pchunk_plb_ms = []
        keep_pref_upchunks = []
        for ind_m, pchunk2label_model in enumerate(pchunk2label_models):
            pchunk_plb_m = []
            for ind_st, upchunk in enumerate(tchunks_unique):
                pref_upchunk = (upchunk[0], upchunk[1], upchunk[2])  # 包括sentid
                if pref_upchunk in pchunk2label_model:
                    pchunk_plb_m.append(pchunk2label_model[pref_upchunk])
                else:
                    pchunk_plb_m.append('O')
                if ind_m == 0:
                    keep_pref_upchunks.append(pref_upchunk)
            pchunk_plb_ms.append(pchunk_plb_m)

        assert len(pchunk_plb_ms) == len(self.fmodels_span)
        return pchunk_plb_ms, keep_pref_upchunks

    def voting_majority(self):
        """多数投票（使用BIO文件）"""
        pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs_bio()
        assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

        comb_kchunks = []
        for pchunk_plb_m, pref_upchunk in zip(zip(*pchunk_plb_ms), keep_pref_upchunks):
            lb2num_dic = {}
            for plbm in pchunk_plb_m:
                if plbm not in lb2num_dic:
                    lb2num_dic[plbm] = 0
                lb2num_dic[plbm] += 1

            max_lb, max_num = '', 0
            for plb, pnum in lb2num_dic.items():
                if pnum > max_num:
                    max_lb, max_num = plb, pnum
            if max_lb != 'O':
                comb_kchunks.append(pref_upchunk + (max_lb,))

        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
        return f1, p, r, correct_preds, total_preds, total_correct

    def voting_weightByOverallF1(self):
        """基于整体F1加权投票（使用BIO文件）"""
        pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs_bio()
        assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

        comb_kchunks = []
        for pchunk_plb_m, pref_upchunk in zip(zip(*pchunk_plb_ms), keep_pref_upchunks):
            lb2num_dic = {}
            for i, plbm in enumerate(pchunk_plb_m):
                if plbm not in lb2num_dic:
                    lb2num_dic[plbm] = 0.0
                lb2num_dic[plbm] += self.f1s[i]  # 使用F1分数作为权重

            max_lb, max_num = '', 0.0
            for plb, pnum in lb2num_dic.items():
                if pnum > max_num:
                    max_lb, max_num = plb, pnum
            if max_lb != 'O':
                comb_kchunks.append(pref_upchunk + (max_lb,))

        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
        return f1, p, r, correct_preds, total_preds, total_correct

    def voting_weightByCateF1(self):
        """基于类别F1加权投票（使用BIO文件）"""
        pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs_bio()
        assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

        comb_kchunks = []
        for pchunk_plb_m, pref_upchunk in zip(zip(*pchunk_plb_ms), keep_pref_upchunks):
            lb2num_dic = {}
            for i, plbm in enumerate(pchunk_plb_m):
                if plbm not in lb2num_dic:
                    lb2num_dic[plbm] = 0.0

                # 使用该模型在该类别上的F1分数作为权重
                category_f1 = self.class2f1_models[i].get(plbm, 0.0)
                lb2num_dic[plbm] += category_f1

            max_lb, max_num = '', 0.0
            for plb, pnum in lb2num_dic.items():
                if pnum > max_num:
                    max_lb, max_num = plb, pnum
            if max_lb != 'O':
                comb_kchunks.append(pref_upchunk + (max_lb,))

        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
        return f1, p, r, correct_preds, total_preds, total_correct

    def voting_spanPred_onlyScore(self):
        """基于Span预测分数投票（使用span文件）"""
        wf1 = self.wf1
        wscore = self.wscore
        pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs_span()
        assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

        print('self.fn_prob: ', self.fn_prob)
        try:
            pchunk_labPrb_dic = self.mres_span.read_span_score(keep_pref_upchunks, self.fn_prob)
        except Exception as e:
            print(f"读取概率文件出错: {e}, 使用默认概率")
            pchunk_labPrb_dic = {}

        comb_kchunks = []
        for pchunk_plb_m, pref_upchunk in zip(zip(*pchunk_plb_ms), keep_pref_upchunks):
            lb2num_dic = {}
            for i, (plbm, f1) in enumerate(zip(pchunk_plb_m, self.f1s)):
                if plbm not in lb2num_dic:
                    lb2num_dic[plbm] = 0.0

                # 获取概率分数
                try:
                    prob_score = pchunk_labPrb_dic.get(pref_upchunk, {}).get(plbm, 0.5)
                except Exception:
                    prob_score = 0.5

                # 组合F1权重和概率分数
                combined_score = wf1 * f1 + wscore * prob_score
                lb2num_dic[plbm] += combined_score

            max_lb, max_num = '', 0.0
            for plb, pnum in lb2num_dic.items():
                if pnum > max_num:
                    max_lb, max_num = plb, pnum
            if max_lb != 'O':
                comb_kchunks.append(pref_upchunk + (max_lb,))

        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
        return f1, p, r, correct_preds, total_preds, total_correct

