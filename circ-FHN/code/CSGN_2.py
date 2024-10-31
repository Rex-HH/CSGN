# -*- coding: utf-8 -*-
#from tensorflow.python.keras.utils.generic_utils import default
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Attention, GlobalAveragePooling1D, MultiHeadAttention, Add

from four2 import *
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GRU, Bidirectional, Conv1D, AveragePooling1D, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
import numpy as np
import argparse
from sklearn.metrics import precision_recall_curve, average_precision_score
from MultiScale import MultiScaleLayer
from fastkan.tffastkan import FastKANLayer
# 配置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义函数run_CRIP(parser)，该函数用于运行CRIP模型的训练和评估过程。
def run_CRIP(parser):
    protein = parser.protein
    batch_size = parser.batch_size
    hiddensize = parser.hiddensize
    n_epochs = parser.n_epochs
    nbfilter = parser.nbfilter
    trainXeval, test_X, trainYeval, test_y = dealwithdata(protein)
    test_y = test_y[:, 1]
    kf = KFold(n_splits=5)

    fpr_list, tpr_list, aucs, Acc = [], [], [], []
    precision1, recall1, fscore1 = [], [], []
    i = 0
    for train_index, eval_index in kf.split(trainYeval):
        train_X = trainXeval[train_index]
        train_y = trainYeval[train_index]
        eval_X = trainXeval[eval_index]
        eval_y = trainYeval[eval_index]

        print('configure cnn network')
        # model = Sequential()
        # model.add(Conv1D(filters=nbfilter, kernel_size=7, activation='relu', input_shape=(101, 56)))
        # model.add(AveragePooling1D(pool_size=5))
        # model.add(Dropout(0.5))
        # # 使用自定义的 MultiScaleLayer
        # model.add(MultiScaleLayer())
        # model.add(Bidirectional(GRU(hiddensize, return_sequences=True)))
        # # 在 LSTM/GRU 层之后添加 Attention 层
        # model.add(Attention())
        # model.add(Flatten())
        #
        #
        # # model.add(Dense(nbfilter, activation='relu'))
        # # model.add(Dropout(0.25))
        # # model.add(Dense(2))
        #
        # # 替换原来的全连接层
        # model.add(FastKANLayer(input_dim=4560, output_dim=nbfilter))  # hiddensize * 2 是 Bidirectional 的输出维度
        # model.add(Dropout(0.3))
        # model.add(FastKANLayer(input_dim=nbfilter, output_dim=2))  # 最终的输出维度为2
        #
        # model.add(Activation('softmax'))
        inputs = Input(shape=(101, 56))

       # 添加卷积和池化层
        x = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(inputs)
        x = AveragePooling1D(pool_size=5)(x)

        # mu = MultiHeadAttention(num_heads=4, key_dim=56)(inputs, inputs)  # 使用输入作为 query 和 value
        # mu = Dropout(0.5)(mu)
        # mu = Add()([inputs, mu])
        # mu = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(mu)
        # mu = AveragePooling1D(pool_size=5)(mu)
        # x = Add()([mu, x])
        # x = Dense(64, activation='relu')(x)

        # 使用自定义的 MultiScaleLayer
        # x = MultiScaleLayer()(x)

        # 添加 Bidirectional GRU 层
        x = Bidirectional(GRU(hiddensize, return_sequences=True))(x)
        # gru = Bidirectional(GRU(hiddensize, return_sequences=True))(x)
        # gru = Bidirectional(GRU(hiddensize, return_sequences=True))(gru)
        # x = Add()([gru, x])

        # 使用多头注意力机制，output_dim 取决于上层输出的大小
        # attention_output = MultiHeadAttention(num_heads=8, key_dim=128)(x, x)

        # 添加全局平均池化或其他处理方式
        #x = GlobalAveragePooling1D()(attention_output)

        x = Flatten(name='HH')(x)

        # 全连接层
        x = Dense(64, activation='relu' )(x)
        x = Dropout(0.3)(x)
        outputs = Dense(2, activation='softmax')(x)

        # # 添加 Attention 层
        # # 注意，Attention 层的输入应该是列表
        # attention_output = Attention()([x, x])  # 将 x 作为 query 和 value
        #
        # # 将注意力的输出展平
        # x = Flatten()(attention_output)
        #
        # # 替换原来的全连接层
        # x = FastKANLayer(input_dim=4560, output_dim=nbfilter)(x)  # hiddensize * 2 是 Bidirectional 的输出维度
        # x = Dropout(0.3)(x)
        # x = FastKANLayer(input_dim=nbfilter, output_dim=2)(x)  # 最终的输出维度为2
        #
        # # 输出层
        # outputs = Activation('softmax')(x)

        # 构建模型
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
        # 打印模型概述
        model.summary()

        print('model training')
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        model.fit(train_X, train_y, batch_size=batch_size, epochs=n_epochs, verbose=1,
                  validation_data=(eval_X, eval_y),
                  callbacks=[earlystopper])

        # 假设模型名为 'model'
        # 提取 Flatten() 和 第一层 Dense 之前的输出
        # layer_name = 'dense_1'  # 第一层全连接层的名称，也可以通过 model.layers[index] 获取
        intermediate_model = Model(inputs=model.input, outputs=model.get_layer('HH').output)

        # 使用你的数据（如验证集/测试集）获取该层之前的输出
        intermediate_output = intermediate_model.predict(test_X)

        # 进行 PCA 分析，降维到 2 个主成分
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(intermediate_output)

        # 绘制 PCA 图
        # 创建散点图，红色代表负样本，蓝色代表正样本，样本点大小为 10
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=test_y, cmap='bwr', s=10)  # 'bwr' 是红-蓝颜色映射
        plt.title('circ-FHN MOV10')
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        # plt.colorbar()
        # 保存图像

        i += 1
        plt.savefig(f'../plot2/fhn_PUM2{i}.png', dpi=300)
        plt.show()

        predictions = model.predict(test_X)[:, 1]
        pre = np.argmax(model.predict(test_X), axis=-1)
        fpr, tpr, _ = roc_curve(test_y, predictions)
        precision, recall, thresholds = precision_recall_curve(test_y, predictions)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        AP = average_precision_score(test_y, predictions)
        auc = roc_auc_score(test_y, predictions)
        precision = precision_score(test_y, pre)
        recall = recall_score(test_y, pre)
        fscore = f1_score(test_y, pre)
        acc = accuracy_score(test_y, pre)

        aucs.append(auc)
        Acc.append(acc)
        precision1.append(precision)
        recall1.append(recall)
        fscore1.append(fscore)

    if len(fpr_list) == 5:
        roc_data = np.column_stack((fpr_list[-1], tpr_list[-1]))
        np.savetxt('../ROC curve and AUC value/{}(AUC={:.4f}).txt'.format(protein, np.mean(aucs)), roc_data, delimiter='\t')

    print("acid AUC: %.4f " % np.mean(aucs), protein)
    print("acid ACC: %.4f " % np.mean(Acc), protein)
    print("acid precision1: %.4f " % np.mean(precision1), protein)
    print("acid recall1: %.4f " % np.mean(recall1), protein)
    print("acid fscore1: %.4f " % np.mean(fscore1), protein)

# 定义函数parse_arguments(parser)，用于解析命令行参数。
def parse_arguments(parser):
    parser.add_argument('--protein', type=str, default= 'PUM2',metavar='<data_file>',
                        help='the protein for training model')
    parser.add_argument('--nbfilter', type=int, default=102, help='use this option for CNN convolution')
    parser.add_argument('--hiddensize', type=int, default=120, help='use this option for LSTM')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The size of a single mini-batch (default value: 50)')
    parser.add_argument('--n_epochs', type=int, default=40, help='The number of training epochs (default value: 30)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    run_CRIP(args)
