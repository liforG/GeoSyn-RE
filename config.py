# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Site    :
# @File    : config.py
# @Software: PyCharm

import os


class Config:
    def __init__(self):
        self.data_dir = 'data'
        self.labeled_dir = os.path.join(self.data_dir, 'labeled_data')
        # None means <data_dir>/train.json. Callers may pin an explicit file.
        self.radical_vocab_file = None
        self.vocab_file = os.path.join(self.data_dir, 'vocab.txt')
        self.type_file = os.path.join(self.data_dir, 'type.json')
        self.model_dir = 'figure-6-5'

        self.extra_words = ['[PAD]', '[UNK]', '[CLS]']
        self.max_sent_len = 128
        self.padding = 0  #
        self.train_prefix = 'train'
        self.valid_prefix = 'valid'
        self.test_prefix = 'test'
        self.generated_prefix = 'g'
        self.generate_type = 2

        # radical
        self.radical_feature_dim=100
        self.pooler_dim=768
        self.filter_nums=[50,30,20]
        self.kernel_sizes=[5,3,1]

        self.pool_method="max"

        # model param
        self.random_seed = 66
        self.deterministic = True
        self.batch_size = 4
        self.predict_batch = 1
        self.lr = [1e-4, 1e-5, 5e-6][1]
        self.p_step = 100
        self.train_epoch = 100
        self.device = ['cuda:0', 'cpu'][0]
        self.early_stop = 20
        self.save_top_k = 1
        self.predict_threshold = 0.5
        # Preserve the historical six-layer default. Dataset-specific runners
        # set this to the public maximum layer of the selected dataset.
        self.max_relation_layer = 6

        # lstm param
        self.vocab_size = 6000
        self.embedding_size = 256
        self.hidden_size = 256

        # bert params
        self.bert_dir = 'bert_geo'
        self.bert_hidden = 768
        self.bert_head = 4
        self.bert_lr = [1e-5, 5e-6][0]

        # trans param
        self.transformer_layer = 3
        self.trans_type = 2

        self.cont_weight = 0

        # entity and relation
        self.entity_type = ['GS', 'RK', 'SM', 'TM']
        self.relation_type = ['EQ', 'EC', 'DC', 'IN', 'PO', 'NE', 'SE', 'NW', 'SW', 'E', 'W','S','N','C','A','BW']
        # self.entity_type = ['Main', 'Main-q', 'Labor', 'Labor-q', 'Service', 'Place',
        #                     'Rate', 'RateV', 'Fund', 'FundV', 'Time', 'TimeV', 'Base', 'BaseV']
        # self.relation_type = ['>', '>=', '<', '<=', '=', '+', '-', '/', 'has', 'and', '@']
        self.guide = True
        # Guided evaluation is an oracle/teacher-forcing analysis only. GNRED
        # contains relation layers L1-L3, so do not extend guided predictions
        # into non-existent higher gold layers.
        self.guided_max_layer = 3
        self.guided_compute_auc = False
        self.use_schema = True

        # few shot learning
        self.few_ratio = 0.1
        self.few_scale = 100
        self.few_labeled_prefix = 'few-labeled'
        self.few_unlabeled_prefix = 'few-unlabeled'
        self.few_tea_prefix = 'few-tea'
        self.do_weak_label = True

        self.few_n = 5
        self.few_k = 1  #

        self.few_rel_idx = 3

        # prompt
        self.max_pro_len = 512
        self.rel_start = 100
        self.rel_elem_num = 70
        self.rel_mode = 2
        self.rel_place = 0
        self.template = 2

        self.use_fgm = False

        # ===========================================flat fenxi==========================================================
        self.bert_en = 'PLM/bert-base-uncased'
        self.flat_dir = 'data/flat-relation'
        self.flat_id = 1
        self.flat_name = ['kbp37', 'semeval2010'][self.flat_id]
        self.max_flat_len = 512

        # overlap fenxi
        self.overlap_dir = 'data/overlap-relation'
        self.overlap_id = 0
        self.overlap_name = ['NYT', 'WebNLG'][self.overlap_id]
        self.max_overlap_len = 256

        self.math_dir = 'math'
        self.max_math_pl = 512
        self.math_ent_num = 20
        self.math_rel_start = 10 + self.math_ent_num
        self.math_rel_num = 60
        self.prompt_mode = 'auto'



