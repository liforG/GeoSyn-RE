# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Site    :
# @File    : framework.py
# @Software: PyCharm


import os
from models import ITransformer
from data_utils import get_data_loader1, get_data_loader2, load_vocab, get_type_info
from data_utils import save_predict, analysis_relation_distribution, generate_ne_entity_replace
from utils import set_random, compute_metric, calc_running_avg_loss, compute_metric_by_dist, print_dist
from utils import save_config, compute_metric_by_dist_layers, compute_similarity_examples
from utils import relation_filtration_study, relation_filtration_study_guide
import datetime
import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, RobertaTokenizer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import argparse
import copy
import json
import time
from alphabet import Alphabet
import gc
import copy
from data_utils import load_pllm_scores_jsonl





class FrameWorkTrans(object):
    def __init__(self, config):
        self.config = config
        print("模型目录（传入的）:", config.model_dir)
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.model = ITransformer(config.bert_dir, config.max_sent_len, config.bert_hidden,  # 初始化模型
                                  config.bert_head, config.transformer_layer,
                                  self.all_type, config.trans_type,
                                  runtime_config=config).to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model_dir = config.model_dir
        self.top_checkpoints = []
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "radical_vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.model.get_radical_vocab_metadata(), f, ensure_ascii=False, indent=2)
        # self.config.model_dir = self.model_dir
        # self.model_dir = config.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

    def _build_model_state(self, epoch, f1):
        return {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1,
            'radical_vocab': self.model.get_radical_vocab_metadata(),
        }

    def _save_top_k_model(self, epoch, f1, state):
        top_k = int(getattr(self.config, "save_top_k", 1))
        if top_k <= 1:
            return

        file_name = "model_epoch_{:03d}_f1_{:.5f}.pt".format(epoch, f1)
        file_path = os.path.join(self.model_dir, file_name)
        t.save(state, file_path)
        self.top_checkpoints.append({
            "epoch": epoch,
            "f1": float(f1),
            "file": file_name,
        })
        self.top_checkpoints = sorted(
            self.top_checkpoints,
            key=lambda item: (item["f1"], item["epoch"]),
            reverse=True
        )

        for item in self.top_checkpoints[top_k:]:
            old_path = os.path.join(self.model_dir, item["file"])
            if os.path.exists(old_path):
                os.remove(old_path)
        self.top_checkpoints = self.top_checkpoints[:top_k]

        with open(os.path.join(self.model_dir, "top_checkpoints.json"), "w", encoding="utf-8") as f:
            json.dump(self.top_checkpoints, f, ensure_ascii=False, indent=2)

    def save_model(self, epoch, f1):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        state = self._build_model_state(epoch, f1)
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.validate_radical_vocab_metadata(state.get('radical_vocab'))
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def load_model(self, model_file):
        self._load_model(model_file)

    def _setup_train(self):

        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.bert_lr}
        ])
        self.scheduler = t.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def train(self, train_file=None):
        train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=train_file)  # 将文本数据转换为模型可以处理的格式

        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(),
                                                        batch_len))
        print('*' * 20)

        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                time.sleep(0.003)
                self.model.train()
                self.optimizer.zero_grad()

                sentence = t.tensor(batch['ids'], dtype=t.long,
                                    device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model(x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                                  elements=batch['element'], relations=self.relations,
                                  sim_label=compute_similarity_examples(batch), c_w=self.config.cont_weight)

                if loss:
                    loss.backward()
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1, batch_len,
                        round(running_avg_loss, 5)
                    ))

            p, r, f1 = self.eval(op=self.config.valid_prefix)
            self._save_top_k_model(e, f1, self._build_model_state(e, f1))
            # self.scheduler.step()
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = get_data_loader2(op, self.data_dir, batch_size=max(1, self.config.batch_size // 2),
                                      tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len)

        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_candidate, guide_c_prob = [], []
        guide_predict, guide_elem = [], []
        threshold = getattr(self.config, "predict_threshold", 0.5)
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                # pllm_batch = [self.pllm_scores.get(str(sid), {}) for sid in batch.get('sid', [])]
                predict = self.model.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,entity_type=entity_type,
                    relations=self.relations, threshold=threshold,  # 这里继续用你原来的 threshold，不要动态阈值
                    max_layer=getattr(self.config, "max_relation_layer", 6),
                    # pllm_scores=pllm_batch,alpha=getattr(self.config, "fusion_alpha", 0.7),
                )
                # predict = self.model.predict_scratch(
                #     x=sentence, x_mask=mask, entity_index=entity_index,
                #     entity_type=entity_type, relations=self.relations)
                if self.config.guide:
                    g_p, g_e, g_c, g_prob = self.model.predict_guided(
                        x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations, threshold=threshold,
                        max_layer=getattr(self.config, "guided_max_layer", 3)
                    )
                    guide_predict.extend(g_p)
                    guide_elem.extend(g_e)
                    if getattr(self.config, "guided_compute_auc", False):
                        guide_candidate.extend(g_c)
                        guide_c_prob.extend(g_prob)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])
        if op == self.config.test_prefix:
            all_examples = []
            for batch in valid_iter:
                for sid, entity, element in zip(batch['sid'], batch['entity'], batch['element']):
                    all_examples.append({'sid': sid, 'entity': entity, 'element': element})  # 存储提取的示例数据 也就是数据集中的
            print(
                f"------------------------------predict on test dataset and save to {self.model_dir}/predict.json----------------------------------------")
            print("保存文件的完整路径：", os.path.join(self.model_dir, 'predict.json'))
            save_predict(all_predict, all_examples, os.path.join(self.model_dir, 'predict.json'))
            analysis_relation_distribution(all_elem, all_entity, [r[0] for r in self.relations],  # 关系分布的表格  并不是预测是表格
                                           out_file=os.path.join(self.model_dir, 'labeled_analysis.csv'))
            analysis_relation_distribution([e + p for e, p in zip(all_entity, all_predict)], all_entity,
                                           [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'predict_analysis.csv'))
            relation_filtration_study(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])


        p, r, f1 = compute_metric(all_elem, all_predict,
                                  all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        # [r_[0] for r_ in self.relations]
        # mp, mr, mf1, ten_gold_dist = compute_metric_by_dist(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity,
                                             [r_[0] for r_ in self.relations])

        print_dist(mp, mr, mf1, [r_[0] for r_ in self.relations],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        if self.config.guide:
            if op == self.config.test_prefix:
                relation_filtration_study_guide(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
                # relation_filtration_study_guide(all_elem, guide_predict, [r_[0] for r_ in self.relations])
            # gp, gr, gf1 = compute_metric_by_dist_layers(guide_elem, guide_predict, [r_[0] for r_ in self.relations], ten_gold_dist)
            gp, gr, gf1 = compute_metric_by_dist_layers(guide_elem, guide_predict, [r_[0] for r_ in self.relations])

            # gp, gr, gf1 = compute_metric_by_dist_layers(all_elem, guide_predict, [r_[0] for r_ in self.relations])
            print('#' * 5, 'guide result', '#' * 5)
            print_dist(gp, gr, gf1, [r_[0] for r_ in self.relations],
                       out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'guide.xlsx'),
                       layer_start=1)
            for layer_idx in range(len(gp) - 1):
                print('[INFO] {} | guided L{} result: p = {}, r = {}, f1 = {}'.format(
                    datetime.datetime.now(), layer_idx + 1,
                    round(float(gp[layer_idx][-1]), 5), round(float(gr[layer_idx][-1]), 5),
                    round(float(gf1[layer_idx][-1]), 5)))
            print('[INFO] {} | guided overall result: p = {}, r = {}, f1 = {}'.format(
                datetime.datetime.now(), round(float(gp[-1][-1]), 5), round(float(gr[-1][-1]), 5),
                round(float(gf1[-1][-1]), 5)))
            if getattr(self.config, "guided_compute_auc", False):
                print('[INFO] {} | guided cumulative auc result: {}'.format(
                    datetime.datetime.now(), compute_auc(guide_candidate, guide_c_prob, guide_elem)))
        return p, r, f1

    def eval_pipeline(self, data_file):
        valid_iter = get_pipeline_loader2(self.config.batch_size, self.tokenizer,
                                          max_seq_len=self.config.max_sent_len, data_file=data_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | pipeline batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_raw_entity, all_new_entity = [], [], [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                new_entity_type = [[e[0] for e in se] for se in batch['new_entity']]
                new_entity_index = [[e[1] for e in se] for se in batch['new_entity']]
                predict = self.model.predict_scratch(x=sentence, x_mask=mask, entity_index=new_entity_index,
                                                     entity_type=new_entity_type, relations=self.relations)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_raw_entity.extend(batch['raw_entity'])
                all_new_entity.extend(batch['new_entity'])
        p, r, f1 = compute_pipeline_metric(all_elem, all_predict, all_raw_entity, all_new_entity)
        print('[INFO] {} | pipeline test result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), p, r, f1))

    def predict(self):
        pass


class FrameWorkTranss(object):
    def __init__(self, config):
        print('model name is BERT + Transs')
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.model = ITranssformer(config.bert_dir, config.max_sent_len, config.bert_hidden,
                                   config.bert_head, config.transformer_layer,
                                   self.all_type, config.trans_type).to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model_dir = config.model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        # self.config.model_dir = self.model_dir
        # self.model_dir = config.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def load_model(self, model_file):
        self._load_model(model_file)

    def _setup_train(self):

        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': self.config.bert_lr}
        ])
        self.scheduler = t.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def train(self, train_file=None):
        train_iter = get_data_loader2(
            self.config.train_prefix, self.data_dir, batch_size=self.config.batch_size,
            tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len, data_file=train_file)
        batch_len = len(train_iter)
        self._setup_train()
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                time.sleep(0.003)
                self.model.train()
                self.optimizer.zero_grad()

                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]

                loss = self.model.forward_cont(x=sentence, x_mask=mask, entity_index=entity_index,
                                               entity_type=entity_type,
                                               elements=batch['element'], relations=self.relations,
                                               sim_label=compute_similarity_examples(batch),
                                               c_w=self.config.cont_weight)
                if loss:
                    loss.backward(retain_graph=False)
                else:
                    print(batch['sid'])
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1, batch_len,
                        round(running_avg_loss, 5)
                    ))

            p, r, f1 = self.eval(op=self.config.valid_prefix)
            # self.scheduler.step()
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)

    def eval(self, op='valid'):
        valid_iter = get_data_loader2(op, self.data_dir, batch_size=max(1, self.config.batch_size // 2),
                                      tokenizer=self.tokenizer, max_seq_len=self.config.max_sent_len)
        batch_len = len(valid_iter)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_entity = [], [], []
        guide_candidate, guide_c_prob = [], []
        guide_predict, guide_elem = [], []
        threshold = getattr(self.config, "predict_threshold", 0.5)
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                entity_type = [[e[0] for e in se] for se in batch['entity']]
                entity_index = [[e[1] for e in se] for se in batch['entity']]
                predict = self.model.predict_scratch(
                    x=sentence, x_mask=mask, entity_index=entity_index,
                    entity_type=entity_type, relations=self.relations)
                if self.config.guide:
                    g_p, g_e, g_c, g_prob = self.model.predict_guided(
                        x=sentence, x_mask=mask, entity_index=entity_index, entity_type=entity_type,
                        elements=batch['element'], relations=self.relations, threshold=threshold,
                        max_layer=getattr(self.config, "guided_max_layer", 3)
                    )
                    guide_predict.extend(g_p)
                    guide_elem.extend(g_e)
                    if getattr(self.config, "guided_compute_auc", False):
                        guide_candidate.extend(g_c)
                        guide_c_prob.extend(g_prob)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_entity.extend(batch['entity'])
        if op == self.config.test_prefix:
            all_examples = []
            for batch in valid_iter:
                for sid, entity, element in zip(batch['sid'], batch['entity'], batch['element']):
                    all_examples.append({'sid': sid, 'entity': entity, 'element': element})
            print(
                f"------------------------------predict on test dataset and save to {self.model_dir}/predict.json----------------------------------------")
            save_predict(all_predict, all_examples, os.path.join(self.model_dir, 'predict.json'))
            analysis_relation_distribution(all_elem, all_entity, [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'labeled_analysis.csv'))
            analysis_relation_distribution([e + p for e, p in zip(all_entity, all_predict)], all_entity,
                                           [r[0] for r in self.relations],
                                           out_file=os.path.join(self.model_dir, 'predict_analysis.csv'))

            relation_filtration_study(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, [r_[0] for r_ in self.relations])
        print_dist(mp, mr, mf1, [r_[0] for r_ in self.relations],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        if self.config.guide:
            if op == self.config.test_prefix:
                relation_filtration_study_guide(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
            gp, gr, gf1 = compute_metric_by_dist_layers(guide_elem, guide_predict, [r_[0] for r_ in self.relations])
            # gp, gr, gf1 = compute_metric_by_dist_layers(guide_elem, guide_predict, [r_[0] for r_ in self.relations], op,
            #                                             test_prefix=self.config.test_prefix)
            print('#' * 5, 'guide result', '#' * 5)
            print_dist(gp, gr, gf1, [r_[0] for r_ in self.relations],
                       out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'guide.xlsx'),
                       layer_start=1)
            for layer_idx in range(len(gp) - 1):
                print('[INFO] {} | guided L{} result: p = {}, r = {}, f1 = {}'.format(
                    datetime.datetime.now(), layer_idx + 1,
                    round(float(gp[layer_idx][-1]), 5), round(float(gr[layer_idx][-1]), 5),
                    round(float(gf1[layer_idx][-1]), 5)))
            print('[INFO] {} | guided overall result: p = {}, r = {}, f1 = {}'.format(
                datetime.datetime.now(), round(float(gp[-1][-1]), 5), round(float(gr[-1][-1]), 5),
                round(float(gf1[-1][-1]), 5)))
            if getattr(self.config, "guided_compute_auc", False):
                print('[INFO] {} | guided cumulative auc result: {}'.format(
                    datetime.datetime.now(), compute_auc(guide_candidate, guide_c_prob, guide_elem)))
        return p, r, f1

    def eval_pipeline(self, data_file):
        valid_iter = get_pipeline_loader2(self.config.batch_size, self.tokenizer,
                                          max_seq_len=self.config.max_sent_len, data_file=data_file)
        batch_len = len(valid_iter)
        print('[INFO] {} | pipeline batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        self.model.eval()
        all_predict, all_elem, all_raw_entity, all_new_entity = [], [], [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                time.sleep(0.003)
                sentence = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                mask = sentence.ne(self.tokenizer.pad_token_id)
                new_entity_type = [[e[0] for e in se] for se in batch['new_entity']]
                new_entity_index = [[e[1] for e in se] for se in batch['new_entity']]
                predict = self.model.predict_scratch(x=sentence, x_mask=mask, entity_index=new_entity_index,
                                                     entity_type=new_entity_type, relations=self.relations)
                all_predict.extend(predict)
                all_elem.extend(batch['element'])
                all_raw_entity.extend(batch['raw_entity'])
                all_new_entity.extend(batch['new_entity'])
        p, r, f1 = compute_pipeline_metric(all_elem, all_predict, all_raw_entity, all_new_entity)
        print('[INFO] {} | pipeline test result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), p, r, f1))

    def predict(self):
        pass



def my_all_job():
    from config import Config
    nest_model = [('BERT+Transformer+CNN', FrameWorkTrans)]
    do_w_policy = True
    bs = [4]
    bert_lr = [1e-5]
    lr = [2e-5]
    random_seeds = [111, 66, 1234, 999, 6789]
    cont_weight = [0]
    for model_idx in range(len(nest_model)):
        for bs_idx in range(len(bs)):
            for lr_idx in range(len(lr)):
                for k in range(0, 1):
                    for random_seed in random_seeds[1:2]:
                        for cw in cont_weight:
                            if not do_w_policy:
                                continue
                            if model_idx not in {0}:
                                continue
                            config = Config()
                            config.train_epoch = 100
                            config.random_seed = random_seed
                            config.device = "cuda:0"

                            config.cont_weight, config.lr, config.bert_lr, config.batch_size = \
                                0, lr[lr_idx], bert_lr[lr_idx], bs[bs_idx]

                            config.cont_weight = cw

                            config.model_dir = 'figure-6-5'
                            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            config.data_dir = os.path.join(config.data_dir, '5-%d' % 1)
                            config.model_dir = os.path.join('figure-6-5',
                                                            f'bs_{bs[bs_idx]}_lr_{lr[lr_idx]}_seed_{random_seed}_cw_{cw}_{current_time}')

                            print("当前模型存储地址：", config.model_dir)
                            set_random(config.random_seed)
                            print(
                                'Model Introduction: %s, bs = %d, bert_lr = %.6f, lr = %.6f, fold = %d, use_scheme = %d, cont_weight = %.2f, random = %d' % (
                                    nest_model[model_idx][0], bs[bs_idx], bert_lr[lr_idx], lr[lr_idx], k + 1,
                                    config.use_schema, config.cont_weight, random_seed))
                            fm = nest_model[model_idx][1](config)
                            fm.train()


def do_position_job():
    from config import Config
    bs_list = [4, 8][1:]
    lr_list = [1e-5, 5e-6]
    cw_list = [0, 0.1, 0.2, 0.3]
    nest_model = [('BT', FrameWorkTrans)]
    for m_ in range(len(nest_model)):
        for lr_ in range(len(lr_list)):
            for bs_ in range(len(bs_list)):
                for cw_ in range(len(cw_list)):
                    for k in range(1, 6):
                        print('train model = %s | fold = %d | lr = %.6f | cont-weight = %.2f | batch-size = %d' % (
                            nest_model[m_][0], k, lr_list[lr_], cw_list[cw_], bs_list[bs_]))
                        config = Config()
                        config.train_epoch = 30
                        config.cont_weight = cw_list[cw_]
                        config.lr = lr_list[lr_]
                        config.bert_lr = lr_list[lr_]
                        config.batch_size = bs_list[bs_]
                        config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
                        config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
                        set_random(config.random_seed)
                        fm = nest_model[m_][1](config)
                        fm.train()


def get_pipeline_loader1(batch_size, word2id=None, max_seq_len=64, data_file=None):
    from torch.utils.data import DataLoader
    from data_utils import NDataSet

    def collate_fn(batch):
        sentence, sen_len, raw_entity, new_entity, element, sid = [], [], [], [], [], []
        b_dict = {}
        for idx, line in enumerate(batch):
            item = json.loads(line)
            t_id = item['sid']
            t_sent = [word2id[w] if w in word2id.keys() else word2id['[UNK]'] for w in item['sentence']]
            t_len = len(item['sentence'])
            t_raw_entity = [(e[0], tuple(e[1])) for e in item['raw_entity']]
            t_new_entity = [(e[0], tuple(e[1])) for e in item['new_entity']]
            t_element = [(e[0], tuple(e[1])) for e in item['element'][len(t_raw_entity):]]
            b_dict[idx] = (t_sent, t_len, t_raw_entity, t_new_entity, t_raw_entity + t_element, t_id)
        items = sorted(b_dict.items(), key=lambda x: x[1][1], reverse=True)
        for k, v in items:
            sentence.append(v[0])
            sen_len.append(v[1])
            raw_entity.append(v[2])
            new_entity.append(v[3])
            element.append(v[4])
            sid.append(v[5])
        batch = {'ids': [t.tensor(s) for s in sentence], 'sent_len': sen_len,
                 'raw_entity': raw_entity, 'new_entity': new_entity, 'element': element, 'sid': sid}
        return batch

    _collate_fn = collate_fn
    data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=batch_size,
                           shuffle=False, collate_fn=_collate_fn, num_workers=0)
    return data_iter


def get_pipeline_loader2(batch_size=4, tokenizer=None, max_seq_len=64, data_file=None):
    from torch.utils.data import DataLoader
    from data_utils import NDataSet

    def convert_entity(words, entities):
        """"""
        converted_entities = []
        c_idx2w_idx = []  # character idx -> word idx
        for i in range(len(words)):
            c_idx2w_idx.extend([i] * len(words[i]))
        for e in entities:
            converted_entities.append((e[0], (c_idx2w_idx[e[1][0] + 5], c_idx2w_idx[e[1][1] + 5])))
        return converted_entities

    def collate_fn(batch):
        sentence, raw_entity, new_entity, element, sid = [], [], [], [], []
        for idx, line in enumerate(batch):
            item = json.loads(line)
            t_id = item['sid']
            t_words = ['[CLS]'] + tokenizer.tokenize(item['sentence'])
            t_word_ids = tokenizer.convert_tokens_to_ids(t_words)[:max_seq_len]
            t_word_ids = t_word_ids + [tokenizer.pad_token_id] * (max_seq_len - len(t_words))
            try:
                t_raw_entity = convert_entity(t_words, item['raw_entity'])
                t_new_entity = convert_entity(t_words, item['new_entity'])
            except:
                print(line)
            t_element = t_raw_entity + [(e[0], tuple(e[1])) for e in item['element'][len(t_raw_entity):]]
            sentence.append(t_word_ids)
            raw_entity.append(t_raw_entity)
            new_entity.append(t_new_entity)
            element.append(t_element)
            sid.append(t_id)
        return {'ids': sentence, 'raw_entity': raw_entity, 'new_entity': new_entity, 'element': element, 'sid': sid}

    _collate_fn = collate_fn
    data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=batch_size,
                           shuffle=False, collate_fn=_collate_fn, num_workers=0)
    return data_iter


def compute_pipeline_metric(all_elements, all_predict, all_raw_entity, all_new_entity):
    def get_entity_rep(predict, entity):
        p2er = [e for e in entity]
        for elem in predict:
            if elem in entity:
                continue
            p2er.append((elem[0], (p2er[elem[1][0]][1], p2er[elem[1][1]][1])))
        return p2er

    def get_correct_num(elements, raw_entity, predict, new_entity):
        gold_rep = get_entity_rep(elements, raw_entity)[len(raw_entity):]
        pred_rep = get_entity_rep(predict, new_entity)[len(new_entity):]
        correct = list(set(gold_rep) & set(pred_rep))
        return len(correct)

    print('compute pipeline metric')
    assert len(all_elements) == len(all_predict) == len(all_raw_entity) == len(all_new_entity)
    correct, predict_num, gold_num = 0, 0, 0
    for i in range(len(all_elements)):
        correct += get_correct_num(all_elements[i], all_raw_entity[i], all_predict[i], all_new_entity[i])
        predict_num += len(all_predict[i])
        gold_num += (len(all_elements[i]) - len(all_raw_entity[i]))
    if not predict_num:
        predict_num = 1
    if not gold_num:
        gold_num = 1
    p = round(correct / predict_num, 5)
    r = round(correct / gold_num, 5)
    f1 = round(2 * p * r / (p + r), 5) if p and r else 0
    return p, r, f1


def compute_auc(all_candidates, all_prob, all_elements):
    from sklearn import metrics
    """"""
    label, prob = [], []
    assert len(all_candidates) == len(all_prob) == len(all_elements)
    #
    for l in range(6):
        for i_ in range(len(all_candidates)):
            assert len(all_candidates[i_]) == len(all_elements[i_])
            for l_ in range(len(all_candidates[i_][l:l + 1])):
                for j_ in range(len(all_candidates[i_][l + l_])):
                    prob.append(all_prob[i_][l + l_][j_])
                    if all_candidates[i_][l + l_][j_] in all_elements[i_][l + l_]:
                        label.append(1)
                    else:
                        label.append(0)
        print('gold count {} | {} '.format(sum(label), len(label)))
        print('[INFO] {} | {} layers\' auc result: {}'.format(
            datetime.datetime.now(), l + 1, metrics.roc_auc_score(label, prob)))

    return metrics.roc_auc_score(label, prob)


if __name__ == '__main__':
    my_all_job()

