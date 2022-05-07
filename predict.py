#  -*-coding:utf8 -*-
import os
import pickle

from model import BERT
from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, label_dict_path, weights_path
from dataloader import NamedEntityRecognizer
from utils.tokenizers import Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.backend import K

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def predict(txt, weights_path, label_dict_path):
    with open(label_dict_path, 'rb') as f:  # 打开文件
        categories = pickle.load(f)
    
    # 建立分词器
    
    tokenizer = Tokenizer(dict_path, do_lower_case = True)
    
    bert = BERT(config_path,
                checkpoint_path,
                categories,
                summary = False)
    model = bert.get_model()
    model.load_weights(weights_path)
    # CRF = bert.get_CRF()
    NER = NamedEntityRecognizer(tokenizer, model, categories)
    entities = []
    for start, end, tag in set(NER.recognize(txt)):
        entities.append((txt[start:end + 1], tag, start, end))
    return sorted(entities, key = lambda d: d[2])


if __name__ == '__main__':
    # segment_ids后长于512的部分将被截断，无法预测
    txt = '治疗以手术为主，术前要做腹、盆腔CT，观察腹膜后淋巴结情况'
    for i in predict(txt = txt,
                     weights_path = weights_path + '/chip_roformer_v2_AdaFactorEMA_FINAL.h5',
                     label_dict_path = label_dict_path):
        print(i)
