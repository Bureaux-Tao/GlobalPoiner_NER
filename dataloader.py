from random import shuffle

import numpy as np

from config import maxlen
from utils.snippets import sequence_padding, DataGenerator
from utils.snippets import open, to_array


def load_data(filename, categories, type = "chip"):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        f = f.read()
        if type == "bio":
            for l in f.split('\n\n'):
                # print(l)
                if not l:
                    continue
                d = ['']
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split('\t')
                    d[0] += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                        categories.add(flag[2:])
                    elif flag[0] == 'I':
                        # print(d) # 在此报错是因为BIO的缘故！
                        d[-1][1] = i
                D.append(d)
        elif type == "chip":
            for l in f.split("\n"):
                if not l:
                    continue
                d = ['']
                for i, item in enumerate(l.strip("|||").split("|||")):
                    if i == 0:
                        d[0] += item
                    else:
                        d.append([int(item.split("    ")[0]), int(item.split("    ")[1]), item.split("    ")[2]])
                        categories.add(item.split("    ")[2])
                D.append(d)
    shuffle(D)
    return D


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __init__(self, data, batch_size, tokenizer, categories, maxlen = maxlen):
        super().__init__(data = data, batch_size = batch_size)
        self.tokenizer = tokenizer
        self.categories = categories
        self.maxlen = maxlen
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = self.tokenizer.tokenize(d[0], maxlen = maxlen)
            mapping = self.tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = self.tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(self.categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = self.categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims = 3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    
    def __init__(self, tokenizer, model, categories):
        self.tokenizer = tokenizer
        self.model = model
        self.categories = categories
    
    def recognize(self, text, threshold = 0):
        tokens = self.tokenizer.tokenize(text, maxlen = 512)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = self.model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], self.categories[l])
            )
        return entities


if __name__ == '__main__':
    categories = set()
    for i in load_data("./data/chip.validate", categories, type = "chip"):
        print(i)
    print(categories)
    
