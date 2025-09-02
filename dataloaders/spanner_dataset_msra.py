import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

import os
import json
import torch
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing, BertProcessing
from torch.utils.data import Dataset
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from dataloaders.collate_functions import collate_to_max_length
from dataloaders.truncate_dataset import TruncateDataset
from torch.utils.data import DataLoader
import random
import numpy as np


def get_span_labels(args):
    label2idx = {}
    if 'conll' in args.dataname:
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3, "MISC": 4}

    elif 'note' in args.dataname:
        label2idx = {'O': 0, 'PERSON': 1, 'ORG': 2, 'GPE': 3, 'DATE': 4, 'NORP': 5, 'CARDINAL': 6, 'TIME': 7,
                     'LOC': 8,
                     'FAC': 9, 'PRODUCT': 10, 'WORK_OF_ART': 11, 'MONEY': 12, 'ORDINAL': 13, 'QUANTITY': 14,
                     'EVENT': 15,
                     'PERCENT': 16, 'LAW': 17, 'LANGUAGE': 18}

    elif 'bio' in args.dataname:
        label2idx = {'O': 0, 'protein': 1, 'DNA': 2, 'cell_type': 3, 'cell_line': 4, 'RNA': 5}

    elif args.dataname == 'wnut17':
        label2idx = {'O': 0, 'location': 1, 'group': 2, 'corporation': 3, 'person': 4, 'creative-work': 5, 'product': 6}

    elif args.dataname == 'twitter':
        label2idx = {'O': 0, 'PER': 1, 'LOC': 2, 'OTHER': 3, 'ORG': 4}

    elif args.dataname == 'wiki':
        label2idx = {'O': 0, 'ORG': 1, 'PER': 2, 'LOC': 3, 'MISC': 4}

    # 中文数据集标签映射
    elif args.dataname in ['msra', 'weibo', 'ontonotes4']:
        label2idx = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3, "MISC": 4}
    elif args.dataname == 'cluener':
        # CLUENER数据集标签映射
        label2idx = {"O": 0, "address": 1, "book": 2, "company": 3, "game": 4, "government": 5,
                     "movie": 6, "name": 7, "organization": 8, "position": 9, "scene": 10}
    elif args.dataname == 'cmeee':
        # CMEEE数据集标签映射（医学实体）
        label2idx = {"O": 0, "dis": 1, "sym": 2, "pro": 3, "equ": 4, "dru": 5, "ite": 6, "bod": 7}
    elif args.dataname == 'navigation':
        # navigation：你的数据是“抽取所有实体”，统一成单一类别
        label2idx = {"O": 0, "Entity": 1}

    label2idx_list = []
    for lab, idx in label2idx.items():
        pair = (lab, idx)
        label2idx_list.append(pair)

    morph2idx_list = []
    # 根据数据集类型选择不同的特征映射
    if args.dataname in ['msra', 'weibo', 'ontonotes4', 'cluener', 'cmeee', 'navigation']:
        # 中文数据集特征映射
        morph2idx = {'isdigit': 0, 'ispunct': 1, 'other': 2}
    else:
        # 英文数据集特征映射
        morph2idx = {'isupper': 0, 'islower': 1, 'istitle': 2, 'isdigit': 3, 'other': 4}

    for morph, idx in morph2idx.items():
        pair = (morph, idx)
        morph2idx_list.append(pair)

    return label2idx_list, morph2idx_list


class BERTNERDataset(Dataset):
    """
	Args:
		json_path: path to spanner style json
		tokenizer: BertTokenizer
		max_length: int, max length of query+context
		possible_only: if True, only use possible samples that contain answer for the query/context
	"""

    def __init__(self, args, json_path, tokenizer, max_length: int = 128, possible_only=False,
                 pad_to_maxlen=False):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenzier = tokenizer
        self.max_length = max_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [
                x for x in self.all_data if x.get("start_position") is not None
            ]
        self.pad_to_maxlen = pad_to_maxlen

        self.args = args

        self.max_spanLen = self.args.max_spanLen
        minus = int((self.max_spanLen + 1) * self.max_spanLen / 2)
        self.max_num_span = self.max_length * self.max_spanLen - minus
        self.dataname = self.args.dataname
        self.spancase2idx_dic = {}

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
		Args:
			item: int, idx
		Returns:
			tokens: tokens of query + context, [seq_len]
			token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
			start_labels: start labels of NER in tokens, [seq_len]
			end_labels: end labels of NER in tokens, [seq_len]
			label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
			match_labels: match labels, [seq_len, seq_len]
			sample_idx: sample id
			label_idx: label id

		"""
        cls_tok = "[CLS]"
        sep_tok = "[SEP]"

        # begin{get the label2idx dictionary}
        label2idx = {}
        label2idx_list = self.args.label2idx_list
        for labidx in label2idx_list:
            lab, idx = labidx
            label2idx[lab] = int(idx)
        # end{get the label2idx dictionary}
        Tlabel2idx = {}

        # begin{get the morph2idx dictionary}
        morph2idx = {}
        morph2idx_list = self.args.morph2idx_list
        for morphidx in morph2idx_list:
            morph, idx = morphidx
            morph2idx[morph] = int(idx)
        # end{get the morph2idx dictionary}

        data = self.all_data[item]
        tokenizer = self.tokenzier

        # ---------- 兼容不同字段名：context / text ----------
        context = data.get("context", None)
        if context is None:
            context = data.get("text", None)
        if context is None:
            context = data.get("sentence", None)
        if context is None:
            raise KeyError("Neither 'context' nor 'text' found in data sample.")

        context = context.strip()
        if '\u200b' in context:
            context = context.replace('\u200b', '')
        elif '\ufeff' in context:
            context = context.replace('\ufeff', '')
        elif '  ' in context:
            context = context.replace('  ', ' ')

        # ---------- 兼容没有 span_posLabel 的数据：从 entities 构建 ----------
        span_idxLab = data.get("span_posLabel", None)
        if span_idxLab is None and "entities" in data:
            span_idxLab = {}
            ents = data.get("entities", [])
            for ent in ents:
                # 优先使用 start_offset / end_offset
                s = ent.get("start_offset", ent.get("start", ent.get("start_char", None)))
                e = ent.get("end_offset", ent.get("end", ent.get("end_char", None)))
                if s is None or e is None:
                    continue
                # 大多数数据 end_offset 为右开区间，转成“闭区间”的 eidx
                try:
                    s = int(s)
                    e = int(e)
                except Exception:
                    continue
                if e > s:
                    e = e - 1
                # 边界截断，避免越界
                s = max(0, min(s, max(0, len(context) - 1)))
                e = max(0, min(e, max(0, len(context) - 1)))
                if e < s:
                    continue
                span_idxLab[f"{s};{e}"] = "Entity"  # 统一类别
        if span_idxLab is None:
            # 仍然没有可用标注，给个空字典，后续全当负样本
            span_idxLab = {}

        # 解析正例 span
        sidxs = []
        eidxs = []
        for seidx, label in span_idxLab.items():
            try:
                sidx, eidx = seidx.split(';')
                sidxs.append(int(sidx))
                eidxs.append(int(eidx))
            except Exception:
                continue

        # ---------- 中文/导航：用字符级 ----------
        if self.dataname in ['msra', 'weibo', 'ontonotes4', 'cluener', 'cmeee', 'navigation']:
            # 不去掉空格，保证与 offset 对齐
            words = list(context)
        else:
            words = context.split()

        # convert the span position into the character index, space is also a position.
        pos_span_idxs = []
        for sidx, eidx in zip(sidxs, eidxs):
            pos_span_idxs.append((sidx, eidx))

        # all span (sidx, eidx)
        if self.dataname in ['msra', 'weibo', 'ontonotes4', 'cluener', 'cmeee', 'navigation']:
            all_span_idxs = enumerate_spans(words, offset=0, max_span_width=self.args.max_spanLen)
        else:
            all_span_idxs = enumerate_spans(context.split(), offset=0, max_span_width=self.args.max_spanLen)

        # begin{compute the span weight}
        # 关键修复：兼容右开/右闭 —— 把 (s,e) 与 (s,e-1) 都纳入正例集合
        pos_set = set()
        for s, e in pos_span_idxs:
            pos_set.add((s, e))
            if e > 0:
                pos_set.add((s, e - 1))
        all_span_weights = []
        for span_idx in all_span_idxs:
            weight = self.args.neg_span_weight
            if span_idx in pos_set:
                weight = 1.0
            all_span_weights.append(weight)
        # end{compute the span weight}

        all_span_lens = []
        for idxs in all_span_idxs:
            sid, eid = idxs
            slen = eid - sid + 1
            all_span_lens.append(slen)

        morph_idxs = self.case_feature_tokenLevel(morph2idx, all_span_idxs, words, self.args.max_spanLen)

        # 根据分词器类型进行不同的处理
        if hasattr(tokenizer, 'encode'):  # transformers tokenizer
            try:
                # 限制文本长度，避免序列过长
                max_context_length = self.max_length - 2  # 减去[CLS]和[SEP]
                if len(context) > max_context_length:
                    context = context[:max_context_length]

                encoded = tokenizer.encode_plus(context, add_special_tokens=True, return_offsets_mapping=True)
                tokens = encoded['input_ids']
                type_ids = encoded['token_type_ids']
                offsets = encoded['offset_mapping']
            except Exception:
                # 如果不支持return_offsets_mapping，则手动计算
                max_context_length = self.max_length - 2
                if len(context) > max_context_length:
                    context = context[:max_context_length]

                encoded = tokenizer.encode_plus(context, add_special_tokens=True)
                tokens = encoded['input_ids']
                type_ids = encoded.get('token_type_ids', [0] * len(tokens))

                # 手动计算offset mapping
                tokens_text = tokenizer.tokenize(context)
                offsets = []
                current_pos = 0

                # 为 [CLS] token 添加 offset
                offsets.append((0, 0))

                for token in tokens_text:
                    if token == '[CLS]' or token == '[SEP]':
                        offsets.append((0, 0))
                    else:
                        start = context.find(token, current_pos)
                        if start == -1:
                            start = current_pos
                        end = start + len(token)
                        offsets.append((start, end))
                        current_pos = end

                # 为 [SEP] token 添加 offset
                offsets.append((0, 0))
        else:  # tokenizers tokenizer
            context_tokens = tokenizer.encode(context, add_special_tokens=True)
            tokens = context_tokens.ids
            type_ids = context_tokens.type_ids
            offsets = context_tokens.offsets

        # ---- 关键：当 words 为字符级时，span_idxs 已经是字符索引，不再二次换算 ----
        all_span_idxs_ltoken, all_span_word, all_span_idxs_new_label = self.convert2tokenIdx(
            words, tokens, type_ids, offsets, all_span_idxs, span_idxLab
        )
        span_label_ltoken = []
        for seidx_str, label in all_span_idxs_new_label.items():
            # 对 navigation：label 统一为 'Entity'
            lab = label if label in label2idx else ('Entity' if 'Entity' in label2idx else 'O')
            span_label_ltoken.append(label2idx[lab])

        # return  tokens, type_ids, all_span_idxs_ltoken, pos_span_mask_ltoken
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        all_span_idxs_ltoken = all_span_idxs_ltoken[:self.max_num_span]
        span_label_ltoken = span_label_ltoken[:self.max_num_span]
        all_span_lens = all_span_lens[:self.max_num_span]
        morph_idxs = morph_idxs[:self.max_num_span]
        all_span_weights = all_span_weights[:self.max_num_span]

        # make sure last token is [SEP]
        if hasattr(tokenizer, 'convert_tokens_to_ids'):  # transformers tokenizer
            sep_token = tokenizer.convert_tokens_to_ids(sep_tok)
        else:  # tokenizers tokenizer
            sep_token = tokenizer.token_to_id(sep_tok)
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[:-1] + [sep_token]

        import numpy as np
        real_span_mask_ltoken = np.ones_like(span_label_ltoken)
        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            all_span_idxs_ltoken = self.pad(all_span_idxs_ltoken, value=(0, 0), max_length=self.max_num_span)
            real_span_mask_ltoken = self.pad(real_span_mask_ltoken, value=0, max_length=self.max_num_span)
            span_label_ltoken = self.pad(span_label_ltoken, value=0, max_length=self.max_num_span)
            all_span_lens = self.pad(all_span_lens, value=0, max_length=self.max_num_span)
            morph_idxs = self.pad(morph_idxs, value=0, max_length=self.max_num_span)
            all_span_weights = self.pad(all_span_weights, value=0, max_length=self.max_num_span)

        tokens = torch.LongTensor(tokens).cuda()
        type_ids = torch.LongTensor(type_ids).cuda()  # use to split the first and second sentence.
        all_span_idxs_ltoken = torch.LongTensor(all_span_idxs_ltoken).cuda()
        real_span_mask_ltoken = torch.LongTensor(real_span_mask_ltoken).cuda()
        span_label_ltoken = torch.LongTensor(span_label_ltoken).cuda()
        all_span_lens = torch.LongTensor(all_span_lens).cuda()
        morph_idxs = torch.LongTensor(morph_idxs).cuda()
        all_span_weights = torch.Tensor(all_span_weights).cuda()

        return [
            tokens,
            type_ids,  # use to split the first and second sentence.
            all_span_idxs_ltoken,
            morph_idxs,
            span_label_ltoken.cuda(),
            all_span_lens,
            all_span_weights,
            real_span_mask_ltoken.cuda(),
            words,
            all_span_word,
            all_span_idxs,
        ]

    def case_feature_tokenLevel(self, morph2idx, span_idxs, words, max_spanlen):
        '''
		this function use to characterize the capitalization feature.
		:return:
		'''
        caseidxs = []

        for idxs in span_idxs:
            sid, eid = idxs
            span_word = words[sid:eid + 1]
            caseidx1 = [0 for _ in range(max_spanlen)]
            for j, token in enumerate(span_word):
                tfeat = ''
                if self.dataname in ['msra', 'weibo', 'ontonotes4', 'cluener', 'cmeee', 'navigation']:
                    # 中文特征：数字、标点符号、其他
                    if token.isdigit():
                        tfeat = 'isdigit'
                    elif not token.isalnum():
                        tfeat = 'ispunct'
                    else:
                        tfeat = 'other'
                else:
                    # 英文特征：大小写
                    if token.isupper():
                        tfeat = 'isupper'
                    elif token.islower():
                        tfeat = 'islower'
                    elif token.istitle():
                        tfeat = 'istitle'
                    elif token.isdigit():
                        tfeat = 'isdigit'
                    else:
                        tfeat = 'other'
                caseidx1[j] = morph2idx[tfeat]
            caseidxs.append(caseidx1)

        return caseidxs

    def case_feature_spanLevel(self, spancase2idx_dic, span_idxs, words):
        '''
		this function use to characterize the capitalization feature.
		:return:
		'''
        if self.dataname in ['msra', 'weibo', 'ontonotes4', 'cluener', 'cmeee', 'navigation']:
            # 中文数据集的特征映射
            case2idx = {'isdigit': 0, 'ispunct': 1, 'other': 2}
        else:
            # 英文数据集的特征映射
            case2idx = {'isupper': 0, 'islower': 1, 'istitle': 2, 'isdigit': 3, 'other': 4}

        caseidx = []
        for idxs in span_idxs:
            sid, eid = idxs
            span_word = words[sid:eid + 1]
            caseidx1 = []
            for token in span_word:
                tfeat = ''
                if self.dataname in ['msra', 'weibo', 'ontonotes4', 'cluener', 'cmeee', 'navigation']:
                    # 中文特征：数字、标点符号、其他
                    if token.isdigit():
                        tfeat = 'isdigit'
                    elif not token.isalnum():
                        tfeat = 'ispunct'
                    else:
                        tfeat = 'other'
                else:
                    # 英文特征：大小写
                    if token.isupper():
                        tfeat = 'isupper'
                    elif token.islower():
                        tfeat = 'islower'
                    elif token.istitle():
                        tfeat = 'istitle'
                    elif token.isdigit():
                        tfeat = 'isdigit'
                    else:
                        tfeat = 'other'
                caseidx1.append(tfeat)

            caseidx1_str = ' '.join(caseidx1)
            if caseidx1_str not in spancase2idx_dic:
                spancase2idx_dic[caseidx1_str] = len(spancase2idx_dic) + 1
            caseidx.append(spancase2idx_dic[caseidx1_str])

        return caseidx, spancase2idx_dic

    def pad(self, lst, value=None, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst

    def convert2tokenIdx(self, words, tokens, type_ids, offsets, span_idxs, span_idxLab):
        # convert the all the span_idxs from word-level to token-level

        # ---- 当 words 为字符列表（中文/导航），span_idxs 已是字符级索引：直接用 ----
        char_level = all(isinstance(w, str) and len(w) == 1 for w in words)

        if char_level:
            sidxs = [x1 for (x1, x2) in span_idxs]
            eidxs = [x2 for (x1, x2) in span_idxs]
        else:
            max_length = self.max_length
            sidxs = [x1 + sum([len(w) for w in words[:x1]]) for (x1, x2) in span_idxs]
            eidxs = [x2 + sum([len(w) for w in words[:x2 + 1]]) for (x1, x2) in span_idxs]

        span_idxs_new_label = {}
        for ns, ne, ose in zip(sidxs, eidxs, span_idxs):
            os, oe = ose
            oes_str = "{};{}".format(os, oe)
            nes_str = "{};{}".format(ns, ne)
            if oes_str in span_idxLab:
                label = span_idxLab[oes_str]
                span_idxs_new_label[nes_str] = label
            else:
                span_idxs_new_label[nes_str] = 'O'

        origin_offset2token_sidx = {}
        origin_offset2token_eidx = {}
        for token_idx in range(len(tokens)):
            token_start, token_end = offsets[token_idx]
            if token_start == token_end == 0:
                continue
            origin_offset2token_sidx[token_start] = token_idx
            origin_offset2token_eidx[token_end] = token_idx  # 注意：token_end 为右开

        # convert the position from character-level to token-level.
        span_new_sidxs = []
        span_new_eidxs = []
        n_span_keep = 0

        max_length = self.max_length
        for start, end in zip(sidxs, eidxs):
            # 关键修复：优先把 end 当“右开”的 end+1 去找；找不到再用 end（右闭）
            ts = origin_offset2token_sidx.get(start, None)
            te = origin_offset2token_eidx.get(end + 1, None)
            if te is None:
                te = origin_offset2token_eidx.get(end, None)

            if ts is not None and te is not None:
                if te > max_length - 1 or ts > max_length - 1:
                    continue
                span_new_sidxs.append(ts)
                span_new_eidxs.append(te)
                n_span_keep += 1
            else:
                continue

        all_span_word = []
        for (sidx, eidx) in span_idxs:
            all_span_word.append(words[sidx:eidx + 1])
        all_span_word = all_span_word[:n_span_keep + 1]

        span_idxs_ltoken = []
        for sidx, eidx in zip(span_new_sidxs, span_new_eidxs):
            span_idxs_ltoken.append((sidx, eidx))

        return span_idxs_ltoken, all_span_word, span_idxs_new_label


def worker_init(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(args, data_dir, prefix, shuffle, limit: int = None):
    # 根据数据集名称确定文件名格式
    if args.dataname == 'msra':
        if prefix == 'train':
            filename = 'msra_train_span.json'
        elif prefix == 'dev':
            filename = 'msra_dev_span.json'
        elif prefix == 'test':
            filename = 'msra_test_span.json'
        else:
            filename = f"spanner.{prefix}"
    elif args.dataname == 'navigation':
        if prefix == 'train':
            filename = 'navigation_train_span.json'
        elif prefix == 'dev':
            filename = 'navigation_dev_span.json'
        elif prefix == 'test':
            filename = 'navigation_test_span.json'
        else:
            filename = f"spanner.{prefix}"
    else:
        filename = f"spanner.{prefix}"

    json_path = os.path.join(data_dir, filename)
    print("json_path: ", json_path)

    vocab_path = os.path.join(args.bert_config_dir, "vocab.txt")
    print("use BertWordPieceTokenizer as the tokenizer ")

    # 检查是否为中文数据集
    if args.dataname in ['msra', 'weibo', 'ontonotes4', 'cluener', 'cmeee', 'navigation']:
        print("Using Chinese BERT tokenizer for Chinese dataset")
        # **关键最小改动**：使用 Fast 分词器以获得可靠的 offset_mapping
        from transformers import BertTokenizerFast as BertTokenizer
        chinese_bert_path = "/root/autodl-tmp/HybridNER/models/bert-base-chinese"
        if os.path.exists(chinese_bert_path):
            tokenizer = BertTokenizer.from_pretrained(chinese_bert_path)
            print(f"✓ 使用中文BERT分词器: {chinese_bert_path}")
        else:
            print(f"⚠ 中文BERT不存在，使用英文BERT: {args.bert_config_dir}")
            tokenizer = BertTokenizer.from_pretrained(args.bert_config_dir)
    else:
        tokenizer = BertWordPieceTokenizer(vocab_path)

    dataset = BERTNERDataset(args, json_path=json_path,
                             tokenizer=tokenizer,
                             max_length=args.bert_max_length,
                             pad_to_maxlen=False
                             )

    if limit is not None:
        dataset = TruncateDataset(dataset, limit)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_to_max_length,
        worker_init_fn=worker_init,
    )

    return dataloader
