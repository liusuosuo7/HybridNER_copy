# -*- coding: utf-8 -*
import codecs
import os
import pickle
from .evaluate_metric import evaluate_ByCategory, get_chunks_onesent, evaluate_chunk_level


class DataReader():
    def __init__(self, dataname, file_dir, classes, fmodels, fn_stand_res):
        self.dataname = dataname
        self.file_dir = file_dir
        self.fmodels = fmodels
        self.classes = classes
        self.fn_stand_res = fn_stand_res

    def read_seqModel_data(self, fn, column_no=-1, delimiter=' '):
        # # read seq model's results
        word_sequences = list()
        tag_sequences = list()
        total_word_sequences = list()
        total_tag_sequences = list()
        with codecs.open(fn, 'r', 'utf-8') as f:
            lines = f.readlines()
        curr_words = list()
        curr_tags = list()
        for k in range(len(lines)):
            line = lines[k].strip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):  # new sentence or new document
                if len(curr_words) > 0:
                    word_sequences.append(curr_words)
                    tag_sequences.append(curr_tags)
                    curr_words = list()
                    curr_tags = list()
                continue

            strings = line.split(delimiter)
            word = strings[0].strip()
            tag = strings[column_no].strip()  # be default, we take the last tag
            if tag == 'work' or tag == 'creative-work':  # for wnut17
                tag = 'work'
            if self.dataname == 'ptb2':
                tag = 'B-' + tag
            curr_words.append(word)
            curr_tags.append(tag)
            total_word_sequences.append(word)
            total_tag_sequences.append(tag)
            if k == len(lines) - 1:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)

        return total_word_sequences, total_tag_sequences, word_sequences, tag_sequences

    # get the predict result of sequence model
    def get_seqModel_pred(self, fpath):
        test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, test_trueTag_sequences_sent = self.read_seqModel_data(
            fpath, column_no=1)
        _, test_predTag_sequences, _, test_predTag_sequences_sent = self.read_seqModel_data(fpath, column_no=2)

        pchunk2label_dic = {}
        tchunks = []
        pchunks = []
        for sentid, (true_tags, pred_tags) in enumerate(
                zip(test_trueTag_sequences_sent, test_predTag_sequences_sent)):
            tchunk = get_chunks_onesent(true_tags, sentid)
            pchunk = get_chunks_onesent(pred_tags, sentid)
            tchunks += tchunk
            pchunks += pchunk

            for pchunk1 in pchunk:
                label, sid, eid, sentid = pchunk1
                pchunk2label_dic[(sid, eid, sentid)] = label

        return tchunks, pchunks, pchunk2label_dic

    # get the span predict result of span prediction model,
    def get_spanModel_pred(self, fpath):
        fread = open(fpath, 'r')
        lines = fread.readlines()
        snum = -1
        tchunks = []
        pchunks = []
        pchunk2label_dic = {}

        for j, line in enumerate(lines):
            line = line.strip()
            if '-DOCSTART-' in line:
                continue
            else:
                snum += 1
                spans1 = line.split('\t')
                pchunk = []
                tchunk = []

                for i, span1 in enumerate(spans1):
                    if i == 0:  # skip the sentence content.
                        continue
                    else:
                        sp, seids, ttag, ptag = span1.split(":: ")
                        seid = seids.split(',')
                        sid = int(seid[0])
                        eid = int(seid[1])
                        if ttag != 'O':
                            tchunk.append((ttag, sid, eid, snum))
                        if ptag != 'O':
                            pchunk.append((ptag, sid, eid, snum))
                            pchunk2label_dic[(sid, eid, snum)] = ptag

                tchunks += tchunk
                pchunks += pchunk

        return tchunks, pchunks, pchunk2label_dic

    def get_tchunk2lab_dic(self, tchunks_models_onedim):
        tchunk2label_dic = {}
        for tchunk in tchunks_models_onedim:
            label, sid, eid, sentid = tchunk
            tchunk2label_dic[(sid, eid, sentid)] = label
        return tchunk2label_dic

    def get_allModels_pred(self, ):
        tchunks_models = []
        pchunks_models = []
        pchunk2label_models = []
        class2f1_models = []

        for fmodel in self.fmodels:
            fpath = os.path.join(self.file_dir, fmodel)
            # Heuristics to decide format:
            # - filenames containing 'spanNER' or ending with '_span.txt' -> span format
            # - otherwise, peek the first non-empty line: if contains '::' treat as span format
            is_span = False
            if 'spanNER' in fmodel or fmodel.endswith('_span.txt'):
                is_span = True
            else:
                try:
                    with open(fpath, 'r', encoding='utf-8') as _fin:
                        for _line in _fin:
                            s = _line.strip()
                            if not s or s.startswith('-DOCSTART-'):
                                continue
                            if '::' in s or '\t' in s:
                                is_span = True
                            break
                except Exception:
                    is_span = False

            if is_span:
                tchunks, pchunks, pchunk2label_dic = self.get_spanModel_pred(fpath)
            else:
                tchunks, pchunks, pchunk2label_dic = self.get_seqModel_pred(fpath)

            tchunks_models.append(tchunks)
            pchunks_models.append(pchunks)
            pchunk2label_models.append(pchunk2label_dic)

            # get the class2f1 for each model
            class2f1 = evaluate_ByCategory(tchunks, pchunks, self.classes)
            class2f1_models.append(class2f1)

        tchunks_models_onedim = []
        pchunks_models_onedim = []
        for tchunks in tchunks_models:
            tchunks_models_onedim += tchunks
        for pchunks in pchunks_models:
            pchunks_models_onedim += pchunks

        tchunks_unique = list(set(tchunks_models_onedim))
        tchunk2label_dic = self.get_tchunk2lab_dic(tchunks_unique)

        return tchunks_models, tchunks_unique, pchunks_models, tchunks_models_onedim, pchunks_models_onedim, pchunk2label_models, tchunk2label_dic, class2f1_models

    def get_sent_word(self):
        # choose CnnWglove_lstmCrf model as the standard test-set result-file.
        fpath = os.path.join(self.file_dir, self.fn_stand_res)
        test_word_sequences, test_trueTag_sequences, test_word_sequences_sent, test_trueTag_sequences_sent = self.read_seqModel_data(
            fpath, column_no=1)
        return test_word_sequences_sent

    def read_span_score(self, pref_upchunks, fn_prob):
        pchunk_labPrb_dic = {}
        if fn_prob == '':
            for pref_upchunk in pref_upchunks:
                pchunk_labPrb_dic[pref_upchunk] = {}
                for class_name in self.classes:
                    pchunk_labPrb_dic[pref_upchunk][class_name] = 0.5
                pchunk_labPrb_dic[pref_upchunk]['O'] = 0.5
        else:
            pchunk_labPrb_dic = pickle.load(open(fn_prob, "rb"))
        return pchunk_labPrb_dic

    def build_prob_from_model(self, pref_upchunks, model_score_fn):
        """
        使用回调函数 model_score_fn 对每个 (sid,eid,sentid) 计算各类别概率，
        返回与 read_span_score 相同结构的字典。
        model_score_fn 接口：
            probs = model_score_fn(sid, eid, sentid) -> dict[label] = prob
        """
        pchunk_labPrb_dic = {}
        for key in pref_upchunks:
            try:
                probs = model_score_fn(*key)
            except Exception:
                probs = None
            if not probs:
                # fallback to 0.5 for robustness
                probs = {c: 0.5 for c in self.classes}
                probs['O'] = 0.5
            pchunk_labPrb_dic[key] = probs
        return pchunk_labPrb_dic