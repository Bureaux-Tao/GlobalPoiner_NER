import pickle

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, label_dict_path
from utils.backend import keras, K, multilabel_categorical_crossentropy, sparse_multilabel_categorical_crossentropy
from utils.models import build_transformer_model
from utils.optimizers import Adam, extend_with_piecewise_linear_lr, extend_with_exponential_moving_average, \
    extend_with_gradient_accumulation, AdaFactor
from utils.layers import GlobalPointer as GlobalPointer
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout, concatenate
from keras.models import Model
from config import *


class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """
    
    def __init__(self, layer, lamb, is_ada = False):
        self.layer = layer
        self.lamb = lamb  # 学习率比例
        self.is_ada = is_ada  # 是否自适应学习率优化器
    
    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma',
                    'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb  # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb ** 0.5  # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb)  # 更改初始化
                setattr(self.layer, key, weight * lamb)  # 按比例替换
        return self.layer(inputs)


class BERT:
    def __init__(self, config_path,
                 checkpoint_path,
                 categories,
                 summary = True):
        model = build_transformer_model(
            config_path,
            checkpoint_path,
            model = model_type,
            load_pretrained_model = True
        )
        lstm = SetLearningRate(
            Bidirectional(LSTM(lstm_hidden_units,
                               return_sequences = True,
                               recurrent_dropout = dropout_rate,
                               kernel_initializer = 'he_normal')),
            20, True
        )(model.output)
        x = concatenate(
            [lstm, model.output],
            axis = -1
        )  # [batch_size, seq_length, lstm_units * 2 + 768]
        x = SetLearningRate(
            TimeDistributed(Dense(512, activation = 'relu',
                                  kernel_initializer = 'he_normal')),
            20, True
        )(x)
        x = TimeDistributed(
            Dropout(dropout_rate))(x)
        final_output = SetLearningRate(
            GlobalPointer(len(categories), 64, kernel_initializer = "he_normal"),
            10, True
        )(x)
        
        # lstm = Bidirectional(LSTM(lstm_hidden_units,
        #                           return_sequences = True,
        #                           recurrent_dropout = dropout_rate,
        #                           kernel_initializer = 'he_normal'))(model.output)
        # x = concatenate(
        #     [lstm, model.output],
        #     axis = -1)
        # x = TimeDistributed(Dense(512, activation = 'relu',
        #                           kernel_initializer = 'he_normal'))(x)
        # final_output = GlobalPointer(len(categories), 64, kernel_initializer = "he_normal")(x)
        
        # final_output = SetLearningRate(
        #     GlobalPointer(len(categories), 64, use_bias = False, kernel_initializer = "he_normal"),
        #     20, True
        # )(model.output)
        
        # final_output = GlobalPointer(
        #     heads = len(categories),
        #     head_size = 64,
        #     use_bias = False,
        #     kernel_initializer = "he_normal"
        # )(model.output)
        
        model = Model(model.input, final_output)
        for layer in model.layers:
            layer.trainable = True
        
        if summary:
            model.summary()
        
        # optimizer_name = "Adam"
        # self.optimizer = Adam(lr = max_lr)
        
        # optimizer_name = "AdamEMA"
        # AdamEMA = extend_with_exponential_moving_average(Adam, name = optimizer_name)
        # self.optimizer = AdamEMA(lr = max_lr)
        
        # optimizer_name = "AdamW"
        # AdamW = extend_with_weight_decay(Adam, optimizer_name)
        # optimizer = AdamW(lr = learning_rate, weight_decay_rate = 0.001)
        
        # optimizer_name = "AdamLR"
        # AdamLR = extend_with_piecewise_linear_lr(Adam, name = optimizer_name)
        # self.optimizer = AdamLR(lr = max_lr * 10, lr_schedule = {
        #     1000: 1,
        #     2000: 0.1
        # })
        
        # optimizer_name = "AdaFactor"
        # self.optimizer = AdaFactor(
        #     learning_rate = max_lr, beta1 = 0.9, min_dim_size_to_factor = 10 ** 6
        # )
        
        optimizer_name = "AdaFactorEMA"
        AdaFactorEMA = extend_with_exponential_moving_average(AdaFactor, name = optimizer_name)
        self.optimizer = AdaFactorEMA(learning_rate = max_lr, beta1 = 0.9, min_dim_size_to_factor = 10 ** 6)
        
        model.compile(
            loss = self.global_pointer_crossentropy,
            optimizer = self.optimizer,
            metrics = [self.global_pointer_f1_score]
        )
        
        self.bert_model = model
        self.optimizer_name = optimizer_name
    
    def global_pointer_crossentropy(self, y_true, y_pred):
        """给GlobalPointer设计的交叉熵
        """
        bh = K.prod(K.shape(y_pred)[:2])
        y_true = K.reshape(y_true, (bh, -1))
        y_pred = K.reshape(y_pred, (bh, -1))
        return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))
    
    def global_pointer_f1_score(self, y_true, y_pred):
        """给GlobalPointer设计的F1
        """
        y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
        return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)
    
    def get_model(self):
        return self.bert_model
    
    def get_optimizer(self):
        return self.optimizer, self.optimizer_name


if __name__ == '__main__':
    config_path = BASE_CONFIG_NAME
    checkpoint_path = BASE_CKPT_NAME
    dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)
    with open(label_dict_path, 'rb') as f:  # 打开文件
        categories = pickle.load(f)
    bert = BERT(config_path,
                checkpoint_path,
                categories)
    
    model = bert.get_model()
    plot_model(model, to_file = './images/model.jpg', show_shapes = True)
