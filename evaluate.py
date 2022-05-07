#! -*- coding: utf-8 -*-
import os

# bert tiny
import pickle

import pandas as pd

from model import BERT
from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, weights_path, label_dict_path, categories_f1_path
from dataloader import load_data, NamedEntityRecognizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.tokenizers import Tokenizer
from tqdm import tqdm

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def get_score(data, NER, tqdm_verbose = False):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    if tqdm_verbose:
        loop = tqdm(data, ncols = 100)
        for d in loop:
            loop.set_description("Evaluating General F1")
            R = set(NER.recognize(d[0]))
            T = set([tuple(i) for i in d[1:]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    
    else:
        for d in data:
            R = set(NER.recognize(d[0]))
            T = set([tuple(i) for i in d[1:]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def get_catetories_score(data, NER, categories, tqdm_verbose = False):
    """评测函数
    """
    labeded_set = {}
    for i in categories:
        labeded_set[i] = {'TP': 1e-10, 'TP+FP': 1e-10, 'TP+FN': 1e-10}
    if tqdm_verbose:
        loop = tqdm(data, ncols = 100)
        for d in loop:
            loop.set_description("Evaluating F1 of each Categories")
            for i in categories:
                R = set(NER.recognize(d[0]))
                R_labeled = set()
                for s, r, label in R:
                    if label == i:
                        R_labeled.add((s, r, label))
                T = set([tuple(i) for i in d[1:]])
                T_labeled = set()
                for s, r, label in T:
                    if label == i:
                        T_labeled.add((s, r, label))
                
                labeded_set[i]["TP"] += len(R_labeled & T_labeled)
                labeded_set[i]["TP+FP"] += len(R_labeled)
                labeded_set[i]["TP+FN"] += len(T_labeled)
    # print(labeded_set)
    for i in labeded_set:
        labeded_set[i]["precision"] = round(labeded_set[i]["TP"] / labeded_set[i]["TP+FP"], 4)
        labeded_set[i]["recall"] = round(labeded_set[i]["TP"] / labeded_set[i]["TP+FN"], 4)
        labeded_set[i]["f1"] = round(2 * labeded_set[i]["TP"] / (labeded_set[i]["TP+FP"] + labeded_set[i]["TP+FN"]), 4)
        labeded_set[i]["TP"] = int(labeded_set[i]["TP"])
        labeded_set[i]["TP+FP"] = int(labeded_set[i]["TP+FP"])
        labeded_set[i]["TP+FN"] = int(labeded_set[i]["TP+FN"])
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return labeded_set


def evaluate(title, data, NER):
    f1, precision, recall = get_score(data, NER, tqdm_verbose = True)
    print(title + ':  f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))
    return f1, precision, recall


def evaluate_categories(title, data, categories, NER):
    result = get_catetories_score(data, NER, categories, tqdm_verbose = True)
    # for i in result:
    #     print(i, result[i])
    df = pd.DataFrame(result)
    df = df.T
    df[["TP", "TP+FP", "TP+FN"]] = df[["TP", "TP+FP", "TP+FN"]].astype(int)
    # 设置value的显示长度为200，默认为50
    pd.set_option('max_colwidth', 200)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    # 显示所有行，把列显示设置成最大
    pd.set_option('display.max_rows', None)
    print(df)
    return df


def evaluate_one(save_file_path, dataset_path, csv_path = categories_f1_path, evaluate_categories_f1 = False):
    with open(label_dict_path, 'rb') as f:  # 打开文件
        categories = set(pickle.load(f))
    
    bert = BERT(config_path,
                checkpoint_path,
                categories,
                summary = False)
    model = bert.get_model()
    
    # 标注数据
    test_data = load_data(dataset_path, categories)
    categories = list(sorted(categories))
    
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case = True)
    
    model.load_weights(save_file_path)
    NER = NamedEntityRecognizer(tokenizer, model, categories)
    
    print("\nweight path:" + save_file_path)
    print("evaluate dataset path:" + dataset_path)
    f1, precision, recall = evaluate("General", test_data, NER)
    if evaluate_categories_f1:
        df = evaluate_categories("Each Categories:", test_data, categories, NER)
        df.to_csv(csv_path, encoding = 'utf-8-sig')
    return f1, precision, recall


if __name__ == '__main__':
    evaluate_one(save_file_path = weights_path + '/chip_roformer_v2_AdaFactorEMA_FINAL.h5',
                 dataset_path = "./data/chip.validate",
                 csv_path = './report/chip_roformer_v2_AdaFactorEMA_FINAL.csv',
                 evaluate_categories_f1 = True)
