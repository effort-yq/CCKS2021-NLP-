# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2021/6/16 20:15
# software: PyCharm

"""
文件说明：
    
"""
import json
import os
os.environ['TF_KERAS'] = '1'     # 不加这个会报错
from bert4keras.backend import keras, K, set_gelu, search_layer, recompute_grad
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.layers import *
from bert4keras.optimizers import *
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Input, Dense
from keras.models import Model
from tqdm import tqdm
import jieba
from keras.losses import kullback_leibler_divergence as kld
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
pretrain_epochs = 10
fine_tune_epochs = 5
pretrain_lr = 5e-5
fine_tune_lr = 2e-5
batch_size = 16
max_len = 64
jieba.initialize()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
label_map = {
    '不匹配': 0,
    '部分匹配': 1,
    '完全匹配': 2
}
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)
set_gelu('tanh')  # 切换gelu版本

id2label = {}
save_model_dir = './output'
predict_dir = f'{save_model_dir}/predict_result'
for k, v in label_map.items():
    id2label[v] = k
def label2id(x):
    if x == '不匹配':
        return 0
    elif x == '部分匹配':
        return 1
    else:
        return 2
train_df = pd.read_csv('train.csv')[['id', 'query', 'text', 'label']]
train_df['label'] = train_df['label'].apply(label2id)
train_df = train_df[['id', 'query', 'text', 'label']].values
test_df = pd.read_csv('test.csv')[['id', 'query', 'text', 'label']].values
test = pd.read_csv('test.csv')[['id', 'query', 'text', 'label']]
all_data = train_df
train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=21, shuffle=True)
user_dict_path = './new_dict.txt'
new_words = []
with open(user_dict_path, encoding='utf-8') as f:
    for l in f:
        w = l.strip()
        new_words.append(w)
        jieba.add_word(w)

# 句子组成对
data = []
for i in all_data:
    query = i[1]
    text = i[2]
    data.append([str(query), str(text)])


# whole word mask
pretrain_data = [[jieba.lcut(line) for line in d] for d in data]

config_path = './pre_model/nezha/bert_config.json'
checkpoint_path = './pre_model/nezha/bert_model.ckpt'
dict_path = './pre_model/nezha/vocab.txt'

jieba.load_userdict(user_dict_path)
def load_user_dict(filename):
    """加载用户词典
    """
    user_dict = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            w = l.split()[0]
            user_dict.append(w)
    return user_dict

# if os.path.exists('./compound_word.josn'):
    #token_dict, keep_tokens, compound_tokens = json.load(
        # open('./compound_word.josn')
    # )
# else:
    # token_dict, keep_tokens = load_vocab(
        # dict_path=dict_path,
        # simplified=True,
        # startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    # )
    # pure_tokenizer = Tokenizer(token_dict.copy(), do_lower_case=True)
    # user_dict = []
    # for w in load_user_dict(user_dict_path):
        # if w not in token_dict:
            # token_dict[w] = len(token_dict)
            # user_dict.append(w)
    # compound_tokens = [pure_tokenizer.encode(w)[0][1:-1] for w in user_dict]
    # json.dump([token_dict, keep_tokens, compound_tokens], open('./compound_word.josn', 'w', encoding='utf-8'))

tokenizer = Tokenizer(dict_path, do_lower_case=True, pre_tokenize=lambda s: jieba.lcut(s, HMM=False))

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数

def random_masking(lines):
    """对输入进行随机mask, 支持多行
    """
    if type(lines[0]) != list:
        lines = [lines]

    sources, targets = [tokenizer._token_start_id], [0]
    segments = [0]

    for i, sent in enumerate(lines):
        source, target = [], []
        segment = []
        rands = np.random.random(len(sent))
        for r, word in zip(rands, sent):
            word_token = tokenizer.encode(word)[0][1:-1]

            if r < 0.15 * 0.8:
                source.extend(len(word_token) * [tokenizer._token_mask_id])     # 随机mask
                target.extend(word_token)
            elif r < 0.15 * 0.9:
                source.extend(word_token)   # 保持不变
                target.extend(word_token)
            elif r < 0.15:
                source.extend([np.random.choice(tokenizer._vocab_size - 5) + 5 for _ in range(len(word_token))])   # 随机替换
                target.extend(word_token)
            else:
                source.extend(word_token)    # 保持不变
                target.extend([0] * len(word_token))

        # add end token
        source.append(tokenizer._token_end_id)
        target.append(0)

        if i == 0:
            segment = [0] * len(source)
        else:
            segment = [1] * len(source)

        sources.extend(source)
        targets.extend(target)
        segments.extend(segment)

    return sources, targets, segments

def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * epsilon * gp

class pretrain_data_generator(DataGenerator):

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked, = [], [], [], []
        batch_nsp = []
        for is_end, item in self.sample(random):
            # 50% shuffle order
            label = 1
            p = np.random.random()
            if p < 0.5:
                label = 0
                item = item[::-1]

            source_tokens, target_tokens, segment_ids = random_masking(item)
            is_masked = [0 if i == 0 else 1 for i in target_tokens]
            batch_token_ids.append(source_tokens)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target_tokens)
            batch_is_masked.append(is_masked)
            batch_nsp.append([label])

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_target_ids = sequence_padding(batch_target_ids)
                batch_is_masked = sequence_padding(batch_is_masked)
                batch_nsp = sequence_padding(batch_nsp)
                yield [batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked, batch_nsp], None

                batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked = [], [], [], []
                batch_nsp = []

# 补齐最后一个batch
more_ids = batch_size - (len(pretrain_data) % batch_size)
pretrain_data = pretrain_data + pretrain_data[: more_ids]
pretrain_generator = pretrain_data_generator(data=pretrain_data, batch_size=batch_size)

def build_transformer_model_with_mlm():
    """带mlm的bert模型
    """

    bert = build_transformer_model(
        config_path,
        checkpoint_path=None,
        with_mlm='linear',
        with_nsp=True,
        dropout_rate=0.3,
        # keep_tokens=keep_tokens,
        # compound_tokens=compound_tokens,
        model='nezha',
        return_keras_model=False,
    )
    proba = bert.model.output
  
    # 辅助输入
    token_ids = Input(shape=(None,), dtype='int64', name='token_ids')  # 目标id
    is_masked = Input(shape=(None,), dtype=K.floatx(), name='is_masked')  # mask标记
    nsp_label = Input(shape=(None,), dtype='int64', name='nsp')  # nsp
    def mlm_loss(inputs):
        """
        计算loss的函数，封装为一个层
        """
        y_true, y_pred, mask = inputs
        _, y_pred = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        return loss

    def mlm_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        _, y_pred = y_pred
        y_true = K.cast(y_true, K.floatx())
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc

    def nsp_loss(inputs):
        """计算nsp loss的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_pred, _ = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred
        )
        loss = K.mean(loss)
        return loss

    def nsp_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_pred, _ = y_pred
        y_true = K.cast(y_true, K.floatx())
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.mean(acc)
        return acc

    mlm_loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, proba, is_masked])
    mlm_acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, proba, is_masked])
    nsp_loss = Lambda(nsp_loss, name='nsp_loss')([nsp_label, proba])
    nsp_acc = Lambda(nsp_acc, name='nsp_acc')([nsp_label, proba])
    train_model = Model(
        bert.model.inputs + [token_ids, is_masked, nsp_label], [mlm_loss, mlm_acc, nsp_loss, nsp_acc]
    )

    loss = {
        'mlm_loss': lambda y_true, y_pred: y_pred,
        'mlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),   # 停止更新梯度
        'nsp_loss': lambda y_true, y_pred: y_pred,
        'nsp_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),   # 停止更新梯度
    }


    return bert, train_model, loss

bert, train_model, loss = build_transformer_model_with_mlm()

Opt = extend_with_weight_decay(Adam)
Opt = extend_with_gradient_accumulation(Opt)
Opt = extend_with_piecewise_linear_lr(Opt)

opt = Opt(learning_rate=pretrain_lr,
          exclude_from_weight_decay=['Norm', 'bias'],
          grad_accum_steps=2,
          lr_schedule={int(len(pretrain_generator) * pretrain_epochs * 0.1): 1.0,
                       len(pretrain_generator) * pretrain_epochs: 0},
          weight_decay_rate=0.01,
          )

train_model.compile(loss=loss, optimizer=opt)
# 如果传入权重，则加载。注：须在此处加载，才保证不报错。
if checkpoint_path is not None:
    bert.load_weights_from_checkpoint(checkpoint_path)

train_model.summary()

model_saved_path = './pre_train/bert-wwm-model.ckpt'

class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self):
        self.loss = 1e6

    """自动保存最新模型
    """

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.loss:
            self.loss = logs['loss']
            bert.save_weights_as_checkpoint(model_saved_path)

        a = np.random.randint(0, len(all_data))
        test_data = all_data[a][1]
        token_ids, segment_ids = tokenizer.encode(str(test_data))
        token_ids[2] = token_ids[3] = tokenizer._token_mask_id

        probs = bert.model.predict([np.array([token_ids]), np.array([segment_ids])])
        print(tokenizer.decode(probs[1][0, 2:4].argmax(axis=1)), str(test_data))

    def on_batch_end(self, batch, logs=None):
        if batch % 1000 == 0 and batch != 0:
            if logs['loss'] < self.loss:
                self.loss = logs['loss']
                bert.save_weights_as_checkpoint(model_saved_path)

        a = np.random.randint(0, len(all_data))
        test_data = all_data[a][1]
        token_ids, segment_ids = tokenizer.encode(str(test_data))
        token_ids[2] = token_ids[3] = tokenizer._token_mask_id

        probs = bert.model.predict([np.array([token_ids]), np.array([segment_ids])])
        print(tokenizer.decode(probs[1][0, 2:4].argmax(axis=1)), str(test_data))

# fine-tune data generator
class data_generator_rdrop(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_labels = []
        for is_end, (id, query, text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                str(query), str(text), maxlen=max_len
            )
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [
                          batch_token_ids, batch_segment_ids
                      ], batch_labels
                batch_token_ids, batch_segment_ids = [], []
                batch_labels = []



class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_labels = []
        for is_end, (id, query, text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                str(query), str(text), maxlen=max_len
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [
                          batch_token_ids, batch_segment_ids
                      ], batch_labels
                batch_token_ids, batch_segment_ids = [], []
                batch_labels = []

train_generator = data_generator_rdrop(data=train_data, batch_size=batch_size)
valid_generator = data_generator_rdrop(val_data, batch_size)
test_generator = data_generator(test_df, batch_size)

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, model):
        self.best_val_f1 = -1e9
        self.model = model

    def on_train_begin(self, logs=None):
        if not ('f1_score' in self.params['metrics']):
            self.params['metrics'].append('f1_score')

    def on_epoch_end(self, epoch, logs=None):
        logs['f1_score'] = float('-inf')
        metrics = self.evaluate()
        logs['f1_score'] = metrics['macro_f1']
        if metrics['macro_f1'] > self.best_val_f1:
            self.best_val_f1 = metrics['macro_f1']
            self.model.save_weights('best_baseline.weights')
        print(u'val_f1: %.5f, f1_a: %.5f, f1_b: %.5f, f1_c: %.5f, best_val_f1: %.5f\n' % (
        metrics['macro_f1'], metrics['f1_a'], metrics['f1_b'], metrics['f1_c'], self.best_val_f1))


    def on_batch_end(self, batch, logs=None):
        logs['f1_score'] = float('-inf')
        if batch % 1000 == 0 and batch != 0:
            metrics = self.evaluate()
            logs['f1_score'] = metrics['macro_f1']
            if metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = metrics['macro_f1']
                self.model.save_weights('best_baseline.weights')
            print(u'val_f1: %.5f, f1_a: %.5f, f1_b: %.5f, f1_c: %.5f, best_val_f1: %.5f\n' % (
            metrics['macro_f1'], metrics['f1_a'], metrics['f1_b'], metrics['f1_c'], self.best_val_f1))


    def evaluate(self):
        """评测函数（A、B两类分别算F1然后求平均）
        """
        pred = []
        true = []

        total_a, right_a = 0., 0.  # 不匹配
        total_b, right_b = 0., 0.  # 部分匹配
        total_c, right_c = 0., 0.  # 完全匹配
        for x_true, y_true in tqdm(valid_generator):
            y_pred = self.model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]

            pred.append(y_pred)
            true.append(y_true)

            for i in range(len(y_pred)):
                if y_pred[i] == 0 and y_true[i] == 0:
                    right_a += 1
                elif y_pred[i] == 1 and y_true[i] == 1:
                    right_b += 1
                elif y_pred[i] == 2 and y_true[i] == 2:
                    right_c += 1

            total_a += list(y_true).count(0)
            total_b += list(y_true).count(1)
            total_c += list(y_true).count(2)

        f1_a = right_a / total_a
        f1_b = right_b / total_b
        f1_c = right_c / total_c

        macro_f1 = (f1_a + f1_b + f1_c) / 3

        return {'macro_f1': macro_f1, 'f1_a': f1_a, 'f1_b': f1_b, 'f1_c': f1_c}


def crossentropy_with_rdrop(y_true, y_pred, alpha=4):
    """配合R-Drop的交叉熵损失
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    loss1 = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return loss1 + K.mean(loss2) / 4 * alpha


def predict_test(model):
    """测试集预测到文件
    """

    # pred = model.predict_generator(test_generator.forfit(), steps=len(test_generator), verbose=1)  # 所有数据，[72329,3]
    # s_1 = pred[:, 1]
    # s_2 = pred[:, 2]
    # idx1 = s_1.shape[0]
    # for i in range(idx1):
    #     if s_2[i] >= 0.42:
    #          test_df[i, 3] = 2
    #     elif s_1[i] >= 0.47:
    #         test_df[i, 3] = 1
    #     else:
    #         test_df[i, 3] = 0
    #
    # tmp_df = pd.DataFrame(test_df)
    # test['label'] = tmp_df[3]
    # print(test)
    pred_label = []
    for x_true, y_true in tqdm(test_generator):
        id = x_true[0]   #  代表token_ids
        id = id.shape[0]   # 代表batch的大小
        y_pred = model.predict(x_true)

        for i in range(id):
            pred_label.append(id2label[y_pred[i].argmax()])

    test['label'] = pred_label

    test[['id', 'query', 'text', 'label']].to_json(f'tmp.json', orient='records', force_ascii=False)

if __name__ == '__main__':
    # pretrain bert use task data
    # 保存模型
    checkpoint = ModelCheckpoint()
    # # 记录日志
    csv_logger = keras.callbacks.CSVLogger('training.log')
    train_model.fit(
        pretrain_generator.forfit(),
        steps_per_epoch=len(pretrain_generator),
        epochs=pretrain_epochs,
        callbacks=[checkpoint, csv_logger],
    )
    early_stopping = EarlyStopping(monitor='f1_score', verbose=1, patience=2, mode='max')
    plateau = ReduceLROnPlateau(monitor="f1_score", verbose=1, mode='max', factor=0.1, patience=2)
    idx = 11
    feed_forward_name = 'Transformer-%d-FeedForward' % idx
    bert_without_mlm = bert.layers[feed_forward_name]
    output = Lambda(lambda x: x[:, 0])(bert_without_mlm.output)
    output = Dense(3, activation='softmax')(output)
    # #
    model = Model(bert.inputs, output)
    model.summary()
    # #
    model.compile(
                  loss=crossentropy_with_rdrop,
                  optimizer=Adam(fine_tune_lr),
                  metrics=['accuracy'])
    adversarial_training(model, 'Embedding-Token', 0.5)
    evaluator = Evaluator(model)
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=fine_tune_epochs,
                        callbacks=[evaluator, early_stopping, plateau])

    model.load_weights('best_baseline.weights')
    predict_test(model)
    print('finish')
