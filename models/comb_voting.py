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


class CombByVoting():
    def __init__(self, dataname, file_dir, fmodels, f1s, cmodelname, classes, fn_stand_res, fn_prob, wscore: float = 0.8, wf1: float = 1.0, model_probs: list = None):
        self.dataname = dataname
        self.file_dir = file_dir
        self.fmodels = fmodels
        self.f1s = f1s
        self.cmodelname = cmodelname
        self.fn_prob = fn_prob
        # 可选：与各输入模型一一对应的概率文件列表（若提供，将在span_score策略中按模型索引使用）
        self.model_probs = model_probs or [None] * len(fmodels)
        self.classes = classes

        self.mres = DataReader(dataname, file_dir, classes, fmodels, fn_stand_res)

        # 组合权重：可通过外部传入调整
        self.wf1 = wf1
        self.wscore = wscore

    def get_unique_pchunk_labs(self):
        tchunks_models, \
        tchunks_unique, \
        pchunks_models, \
        tchunks_models_onedim, \
        pchunks_models_onedim, \
        pchunk2label_models, \
        tchunk2label_dic, \
        class2f1_models = self.mres.get_allModels_pred()
        self.tchunks_unique = tchunks_unique
        self.class2f1_models = class2f1_models
        self.tchunk2label_dic = tchunk2label_dic

        # the unique chunk that predict by the model..
        pchunks_unique = list(set(pchunks_models_onedim))

        # get the unique non-O chunk's label that are predicted by all the 10 models.
        keep_pref_upchunks = []
        pchunk_plb_ms = []
        for pchunk in pchunks_unique:
            lab, sid, eid, sentid = pchunk
            key1 = (sid, eid, sentid)
            if key1 not in keep_pref_upchunks:
                keep_pref_upchunks.append(key1)
                plb_ms = []  # the length is the num of the models
                # the first position is the pchunk
                for i in range(len(self.f1s)):
                    plb = 'O'
                    if key1 in pchunk2label_models[i]:
                        plb = pchunk2label_models[i][key1]
                    plb_ms.append(plb)
                pchunk_plb_ms.append(plb_ms)

        # get the non-O true chunk that are not be recognized..
        for tchunk in tchunks_unique:
            if tchunk not in pchunks_unique:  # it means that the tchunk are not been recognized by all the models
                plab, sid, eid, sentid = tchunk
                key1 = (sid, eid, sentid)
                if key1 not in keep_pref_upchunks:
                    continue

        return pchunk_plb_ms, keep_pref_upchunks

    def best_potential(self):
        pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
        assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

        comb_kchunks = []
        for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
            sid, eid, sentid = pref_upchunks
            key1 = (sid, eid, sentid)
            if key1 in self.tchunk2label_dic:
                klb = self.tchunk2label_dic[key1]
            elif 'O' in pchunk_plb_m:
                klb = 'O'
            else:
                klb = pchunk_plb_m[0]
            if klb != 'O':
                kchunk = (klb, sid, eid, sentid)
                comb_kchunks.append(kchunk)

        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
        print()
        print('best_potential results: ')
        print("f1, p, r, correct_preds, total_preds, total_correct:")
        print(f1, p, r, correct_preds, total_preds, total_correct)

        return [f1, p, r, correct_preds, total_preds, total_correct]

    def voting_majority(self):
        pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
        assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

        comb_kchunks = []
        for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
            lb2num_dic = {}
            for plbm in pchunk_plb_m:
                if plbm not in lb2num_dic:
                    lb2num_dic[plbm] = 0.0
                lb2num_dic[plbm] += 1

            klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
            if klb != 'O':
                sid, eid, sentid = pref_upchunks
                kchunk = (klb, sid, eid, sentid)
                comb_kchunks.append(kchunk)
        comb_kchunks = list(set(comb_kchunks))
        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
        print()
        print('majority_voting results: ')
        print("f1, p, r, correct_preds, total_preds, total_correct:")
        print(f1, p, r, correct_preds, total_preds, total_correct)

        kf1 = int(f1 * 10000)
        fn_save_comb_kchunks = 'comb_result/VM_combine_' + str(kf1) + '.pkl'

        pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

        return [f1, p, r, correct_preds, total_preds, total_correct]

    def voting_weightByOverallF1(self):
        pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
        assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

        comb_kchunks = []
        for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
            lb2num_dic = {}
            for plbm, f1 in zip(pchunk_plb_m, self.f1s):
                if plbm not in lb2num_dic:
                    lb2num_dic[plbm] = 0.0
                lb2num_dic[plbm] += f1
            klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
            if klb != 'O':
                sid, eid, sentid = pref_upchunks
                kchunk = (klb, sid, eid, sentid)
                comb_kchunks.append(kchunk)

        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
        print()
        print('voting_weightByOverallF1 results: ')
        print("f1, p, r, correct_preds, total_preds, total_correct:")
        print(f1, p, r, correct_preds, total_preds, total_correct)

        kf1 = int(f1 * 10000)
        fn_save_comb_kchunks = 'comb_result/VOF1_combine_' + str(kf1) + '.pkl'
        pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

        return [f1, p, r, correct_preds, total_preds, total_correct]

    def voting_weightByCategotyF1(self):
        pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
        assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

        comb_kchunks = []
        for pchunk_plb_m, pref_upchunks in zip(pchunk_plb_ms, keep_pref_upchunks):
            lb2num_dic = {}
            for plbm, f1, cf1_dic in zip(pchunk_plb_m, self.f1s, self.class2f1_models):
                if plbm not in lb2num_dic:
                    lb2num_dic[plbm] = 0.0
                if plbm == 'O':
                    lb2num_dic[plbm] += f1
                else:
                    lb2num_dic[plbm] += cf1_dic[plbm]
            klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
            if klb != 'O':
                sid, eid, sentid = pref_upchunks
                kchunk = (klb, sid, eid, sentid)
                comb_kchunks.append(kchunk)

        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
        print()
        print('voting_weightByCategotyF1 results: ')
        print("f1, p, r, correct_preds, total_preds, total_correct:")
        print(f1, p, r, correct_preds, total_preds, total_correct)

        kf1 = int(f1 * 10000)
        fn_save_comb_kchunks = 'comb_result/VCF1_combine_' + str(kf1) + '.pkl'
        pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

        return [f1, p, r, correct_preds, total_preds, total_correct]

    def voting_spanPred_onlyScore(self, model_score_fn=None):
        wf1 = self.wf1
        wscore = self.wscore
        pchunk_plb_ms, keep_pref_upchunks = self.get_unique_pchunk_labs()
        assert len(pchunk_plb_ms) == len(keep_pref_upchunks)

        print('self.fn_prob: ', self.fn_prob)
        if model_score_fn is not None:
            # 用模型对所有候选span打分，构建完整概率
            pchunk_labPrb_dic = self.mres.build_prob_from_model(keep_pref_upchunks, model_score_fn)
        else:
            pchunk_labPrb_dic = self.mres.read_span_score(keep_pref_upchunks, self.fn_prob)

        comb_kchunks = []
        for pchunk_plb_m, pref_upchunk in zip(pchunk_plb_ms, keep_pref_upchunks):
            lb2num_dic = {}
            for i, (plbm, f1) in enumerate(zip(pchunk_plb_m, self.f1s)):
                if plbm not in lb2num_dic:
                    lb2num_dic[plbm] = 0.0
                # 读取概率时做健壮性处理：若文件缺失/格式不符/键不存在，则回退到0.5
                try:
                    score = pchunk_labPrb_dic[pref_upchunk][plbm]
                except Exception:
                    score = 0.5

                # lb2num_dic[plbm] += score+0.5*f1 # best
                lb2num_dic[plbm] += wscore * score + wf1 * f1  # best

            klb = sorted(lb2num_dic, key=lambda x: lb2num_dic[x])[-1]
            if klb != 'O':
                sid, eid, sentid = pref_upchunk
                kchunk = (klb, sid, eid, sentid)
                comb_kchunks.append(kchunk)
        comb_kchunks = list(set(comb_kchunks))

        f1, p, r, correct_preds, total_preds, total_correct = evaluate_chunk_level(comb_kchunks, self.tchunks_unique)
        print()
        print('voting_spanPred_onlyScore results: ')
        print("f1, p, r, correct_preds, total_preds, total_correct:")
        print(f1, p, r, correct_preds, total_preds, total_correct)

        kf1 = int(f1 * 10000)
        fn_save_comb_kchunks = 'comb_result/SpanNER_combine_' + str(kf1) + '.pkl'
        pickle.dump([comb_kchunks, self.tchunks_unique], open(fn_save_comb_kchunks, "wb"))

        return [f1, p, r, correct_preds, total_preds, total_correct]