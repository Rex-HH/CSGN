# -*- coding: utf-8 -*-
#from tensorflow.python.keras.utils.generic_utils import default
import time

from sklearn.decomposition import PCA
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Attention, GlobalAveragePooling1D, MultiHeadAttention, Add, Embedding, \
    BatchNormalization, Concatenate, Conv2D, MaxPooling2D, Reshape

from four import *
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
from MultiScale2 import MultiScaleLayer
from  GraphConvolution import GraphConvolution
#from spektral.layers import GraphConv
from fastkan.tffastkan import FastKANLayer,FastKAN
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
    trainXeval, test_X, trainSeqeval, test_seq, trainAdjeval, test_adj, trainYeval, test_y, embedding_matrix = dealwithdata(protein)
    # traineval_probMatrs,test_probMatrs, traineval_NDs,test_NDs, traineval_DPCPs,test_DPCPs, traineval_Kmers,test_Kmers, trainSeqeval, test_seq, trainAdjeval, test_adj, trainYeval, test_y, embedding_matrix = dealwithdata(protein)
    test_X = [test_X, test_seq,test_adj]
    # test_X = [test_probMatrs,test_NDs,test_DPCPs,test_Kmers, test_seq,test_adj]
    test_y = test_y[:, 1]
    kf = KFold(n_splits=5)

    fpr_list, tpr_list, aucs, Acc = [], [], [], []
    precision1, recall1, fscore1 = [], [], []
    i = 0
    for train_index, eval_index in kf.split(trainYeval):
        # train_probMatrs = traineval_probMatrs[train_index]
        # eval_probMatrs = traineval_probMatrs[eval_index]
        # train_NDs = traineval_NDs[train_index]
        # eval_NDs = traineval_NDs[eval_index]
        # train_DPCPs = traineval_DPCPs[train_index]
        # eval_DPCPs = traineval_DPCPs[eval_index]
        # train_Kmers = traineval_Kmers[train_index]
        # eval_Kmers = traineval_Kmers[eval_index]

        train_x = trainXeval[train_index]
        eval_x = trainXeval[eval_index]
        train_seq = trainSeqeval[train_index]
        train_adj = trainAdjeval[train_index]
        # train_X = [train_probMatrs,train_NDs,train_DPCPs, train_Kmers, train_seq, train_adj]
        train_X = [train_x, train_seq, train_adj]
        train_y = trainYeval[train_index]

        eval_seq = trainSeqeval[eval_index]
        eval_adj = trainAdjeval[eval_index]
        # eval_X = [eval_probMatrs, eval_NDs, eval_DPCPs, eval_Kmers, eval_seq, eval_adj]
        eval_X = [eval_x, eval_seq, eval_adj]
        eval_y = trainYeval[eval_index]

        print('configure cnn network')
        # model = Sequential()
        # model.add(Conv1D(filters=nbfilter, kernel_size=7, activation='relu', input_shape=(101, 56)))
        # model.add(AveragePooling1D(pool_size=5))
        # model.add(Dropout(0.5))
        # # 使用自定义的 MultiScaleLayer
        # model.add(MultiScaleLayer())
        # model.add(Bidirectional(GRU(hiddensize, return_sequences=True)))
        # # # 在 LSTM/GRU 层之后添加 Attention 层
        # # model.add(Attention())
        # model.add(Flatten())
        #
        #
        # model.add(Dense(nbfilter, activation='relu'))
        # model.add(Dropout(0.25))
        # model.add(Dense(2))

        # # 替换原来的全连接层
        # model.add(FastKANLayer(input_dim=4560, output_dim=nbfilter))  # hiddensize * 2 是 Bidirectional 的输出维度
        # model.add(Dropout(0.3))
        # model.add(FastKANLayer(input_dim=nbfilter, output_dim=2))  # 最终的输出维度为2

        # model.add(Activation('softmax'))


        inputs1 = Input(shape=(101, 56))
        # inputs1_1 = Input(shape=(101, 3))
        # inputs1_2 = Input(shape=(101, 1))
        # inputs1_3 = Input(shape=(101, 11))
        # inputs1_4 = Input(shape=(101, 41))
        inputs2 = Input(shape=(101,))
        inputs3 = Input(shape=(101,101))
        # 添加卷积和池化层
        dro1 = 0.25
        x = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(inputs1)
        x = AveragePooling1D(pool_size=5)(x)
        x = Dropout(dro1)(x)

        # x1 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(inputs1_1)
        # x1 = AveragePooling1D(pool_size=5)(x1)
        # x1 = Dropout(dro1)(x1)
        #
        # x2 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(inputs1_2)
        # x2 = AveragePooling1D(pool_size=5)(x2)
        # x2 = Dropout(dro1)(x2)
        #
        # x3 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(inputs1_3)
        # x3 = AveragePooling1D(pool_size=5)(x3)
        # x3 = Dropout(dro1)(x3)
        #
        # x4 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(inputs1_4)
        # x4 = AveragePooling1D(pool_size=5)(x4)
        # x4 = Dropout(dro1)(x4)

        dro2 = 0.4
        embedding1 = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                              weights=[embedding_matrix], trainable=False)(inputs2)
       # embedding = tf.reduce_mean(embedding, axis=2)  # 结果形状为 (None, 101, 30)

        embedding = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(embedding1)
        embedding = AveragePooling1D(pool_size=5)(embedding)
        embedding = Dropout(dro2)(embedding)
        # print(x.shape)
        # print(embedding.shape)

        # 创建图神经网络输入
        node_features = tf.eye(101)  # 使用单位矩阵作为初始节点特征（可根据实际情况调整）
        node_features = tf.constant(node_features, dtype=tf.float32)  # 创建图节点特征输入
        dr = 0
        # # 使用自定义 GCN 层对邻接矩阵进行特征提取
        # # 添加多层 GCN
        # gcn1 = GraphConvolution(units=nbfilter, activation='relu')([embedding1, inputs3])
        # gcn1 = Dropout(dr)(gcn1)
        #
        # gcn2 = GraphConvolution(units=nbfilter, activation='relu')([gcn1, inputs3])  # 第二层 GCN
        # gcn2 = Dropout(dr)(gcn2)
        #
        # gcn3 = GraphConvolution(units=nbfilter, activation='relu')([gcn2, inputs3])  # 第三层 GCN
        # # gcn3_residual = gcn3 + gcn1
        # gcn3 = Dropout(dr)(gcn3)
        #
        # gcn4 = GraphConvolution(units=nbfilter, activation='relu')([gcn3, inputs3])  # 第四层 GCN
        # #gcn4_residual = gcn4 + gcn1  # 跳跃连接
        # gcn4 = Dropout(dr)(gcn4)
        # #
        # # gcn5 = GraphConvolution(units=nbfilter, activation='relu')([gcn4, inputs3])  # 第五层 GCN
        # # gcn5 = Dropout(0.5)(gcn5)
        #
        # graph_features = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(gcn4)
        # graph_features = AveragePooling1D(pool_size=5)(graph_features)
        # graph_features = Dropout(0.25)(graph_features)


        # 一层GCN
        '''
        gcn1_2_1 = GraphConvolution(units=nbfilter, activation='relu')([node_features, inputs3])
        graph_features_2_1 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(gcn1_2_1)

        graph_features_2_1 = AveragePooling1D(pool_size=5)(graph_features_2_1)
        graph_features_2_1 = Dropout(0.25)(graph_features_2_1)

        # 两层GCN
        gcn1_2_2 = GraphConvolution(units=nbfilter, activation='relu')([node_features, inputs3])
        gcn2_2_2 = GraphConvolution(units=nbfilter, activation='relu')([gcn1_2_2, inputs3])  # 第二层 GCN

        graph_features_2_2 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(gcn2_2_2)
        graph_features_2_2 = AveragePooling1D(pool_size=5)(graph_features_2_2)
        graph_features_2_2 = Dropout(0.25)(graph_features_2_2)

        # 三层GCN
        gcn1_2_3 = GraphConvolution(units=nbfilter, activation='relu')([node_features, inputs3])
        gcn2_2_3 = GraphConvolution(units=nbfilter, activation='relu')([gcn1_2_3, inputs3])  # 第二层 GCN
        gcn3_2_3 = GraphConvolution(units=nbfilter, activation='relu')([gcn2_2_3, inputs3])  # 第三层 GCN

        graph_features_2_3 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(gcn3_2_3)
        graph_features_2_3 = AveragePooling1D(pool_size=5)(graph_features_2_3)
        graph_features_2_3 = Dropout(0.25)(graph_features_2_3)

'''
        #四层GCN
        gcn1_2 = GraphConvolution(units=nbfilter, activation='relu')([node_features, inputs3])
        gcn1_2 = Dropout(dr)(gcn1_2)

        gcn2_2 = GraphConvolution(units=nbfilter, activation='relu')([gcn1_2, inputs3])  # 第二层 GCN
        gcn2_2 = Dropout(dr)(gcn2_2)

        gcn3_2 = GraphConvolution(units=nbfilter, activation='relu')([gcn2_2, inputs3])  # 第三层 GCN
        # gcn3_residual_2 = gcn3_2 + gcn1_2
        gcn3_2 = Dropout(dr)(gcn3_2)

        gcn4_2 = GraphConvolution(units=nbfilter, activation='relu')([gcn3_2, inputs3])  # 第四层 GCN
        # gcn4_residual_2 = gcn4_2 + gcn1_2  # 跳跃连接
        gcn4_2 = Dropout(dr)(gcn4_2)
        #
        # gcn5 = GraphConvolution(units=nbfilter, activation='relu')([gcn4, inputs3])  # 第五层 GCN
        # gcn5 = Dropout(0.5)(gcn5)

        graph_features_2 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(gcn4_2)
        graph_features_2 = AveragePooling1D(pool_size=5)(graph_features_2)
        graph_features_2 = Dropout(0.25)(graph_features_2)

        # 使用自定义 GCN 层对邻接矩阵进行特征提取
        # 添加多层 GCN
        '''
        gcn1_2_5 = GraphConvolution(units=nbfilter, activation='relu')([node_features, inputs3])
        gcn2_2_5 = GraphConvolution(units=nbfilter, activation='relu')([gcn1_2_5, inputs3])  # 第二层 GCN
        gcn3_2_5 = GraphConvolution(units=nbfilter, activation='relu')([gcn2_2_5, inputs3])  # 第三层 GCN
        gcn4_2_5 = GraphConvolution(units=nbfilter, activation='relu')([gcn3_2_5, inputs3])  # 第四层 GCN
        gcn5_2_5 = GraphConvolution(units=nbfilter, activation='relu')([gcn4_2_5, inputs3])  # 第五层 GCN

        graph_features_2_5 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(gcn5_2_5)
        graph_features_2_5 = AveragePooling1D(pool_size=5)(graph_features_2_5)
        graph_features_2_5 = Dropout(0.25)(graph_features_2_5)
'''
        # # 假设 inputs3 是邻接矩阵，形状为 (batch_size, num_nodes, num_nodes)
        # # 1. 将邻接矩阵扩展为 4D 张量 (batch_size, num_nodes, num_nodes, 1)
        # inputs3_4d = Reshape((inputs3.shape[1], inputs3.shape[2], 1))(inputs3)
        #
        # # 2. 使用 Conv2D + 池化层替换原始的 GCN 层和 Dropout 层
        # # 第一层 Conv2D + MaxPooling2D
        # conv1_2 = Conv2D(filters=nbfilter, kernel_size=(3, 3), activation='relu', padding='same')(inputs3_4d)
        # conv1_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)  # 使用 2x2 池化
        #
        # # 第二层 Conv2D + MaxPooling2D
        # conv2_2 = Conv2D(filters=nbfilter, kernel_size=(3, 3), activation='relu', padding='same')(conv1_2)
        # conv2_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)
        #
        # # 第三层 Conv2D + MaxPooling2D
        # conv3_2 = Conv2D(filters=nbfilter, kernel_size=(3, 3), activation='relu', padding='same')(conv2_2)
        # # 可以在此添加残差连接，例如 conv3_residual_2 = conv3_2 + conv1_2
        # conv3_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3_2)
        #
        # # 第四层 Conv2D + MaxPooling2D
        # conv4_2 = Conv2D(filters=nbfilter, kernel_size=(3, 3), activation='relu', padding='same')(conv3_2)
        # # 可以在此添加跳跃连接，例如 conv4_residual_2 = conv4_2 + conv1_2
        # conv4_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4_2)
        #
        # # 3. 将最后一层卷积的 4D 张量转换为 3D 张量，以便继续使用 Conv1D 处理
        # # 假设 conv4_2 的形状为 (batch_size, reduced_nodes, reduced_nodes, nbfilter)
        # conv4_2_reshaped = Reshape((7 * 7, 102))(conv4_2)  # 修改为 (batch_size, 49, 102)
        # # 1. 第一步一维卷积，保持输出的特征数量为 102
        # Conv_features = Conv1D(filters=102, kernel_size=5, activation='relu', padding='same')(
        #     conv4_2_reshaped)  # 仍为 (batch_size, 49, 102)
        #
        # # 2. 第一步池化，输出形状应为 (batch_size, 24, 102)
        # Conv_features = AveragePooling1D(pool_size=3, strides=2)(Conv_features)  # 变为 (batch_size, 24, 102)
        #
        # # 3. 第二步卷积，保持输出的特征数量为 102
        # Conv_features = Conv1D(filters=102, kernel_size=3, activation='relu', padding='same')(
        #     Conv_features)  # 仍为 (batch_size, 24, 102)
        #
        # # 4. 第二步池化，最终将输出调整为 (batch_size, 19, 102)
        # Conv_features = AveragePooling1D(pool_size=3, strides=1)(Conv_features)  # 变为 (batch_size, 22, 102)
        #
        # # 如果仍需调整到 19，可以添加一个卷积，或通过一个额外的池化层进行微调
        # Conv_features = Conv1D(filters=102, kernel_size=3, activation='relu', padding='same')(
        #     Conv_features)  # 输出仍为 (batch_size, 22, 102)
        #
        # # 最后一个池化层，从 22 调整到 19
        # Conv_features = AveragePooling1D(pool_size=2, strides=1)(Conv_features)  # 变为 (batch_size, 21, 102)
        #
        # # 如果需要最终调整到 19，可以考虑将 pool_size 改为 2 或 3
        # Conv_features = AveragePooling1D(pool_size=3, strides=1)(Conv_features)  # 输出 (batch_size, 19, 102)
        # Conv_features = Dropout(0.25)(Conv_features)
        #
        # # 打印输出形状以检查
        # print("Conv_features shape:", Conv_features.shape)  # 检查最终形状，期待 (batch_size, 19, 102)
        #
        # # 使用自定义 GCN 层对邻接矩阵进行特征提取
        # # 添加多层 GCN
        # gcn1_3 = GraphConvolution(units=nbfilter, activation='relu')([inputs1, inputs3])
        # gcn1_3 = Dropout(dr)(gcn1_3)
        #
        # gcn2_3 = GraphConvolution(units=nbfilter, activation='relu')([gcn1_3, inputs3])  # 第二层 GCN
        # gcn2_3 = Dropout(dr)(gcn2_3)
        #
        # gcn3_3 = GraphConvolution(units=nbfilter, activation='relu')([gcn2_3, inputs3])  # 第三层 GCN
        # # gcn3_residual_2 = gcn3_2 + gcn1_2
        # gcn3_3 = Dropout(dr)(gcn3_3)
        #
        # gcn4_3 = GraphConvolution(units=nbfilter, activation='relu')([gcn3_3, inputs3])  # 第四层 GCN
        # gcn4_residual_3 = gcn4_3 + gcn1_3  # 跳跃连接
        # gcn4_3 = Dropout(dr)(gcn4_residual_3)
        # #
        # # gcn5 = GraphConvolution(units=nbfilter, activation='relu')([gcn4, inputs3])  # 第五层 GCN
        # # gcn5 = Dropout(0.5)(gcn5)
        #
        # graph_features_3 = Conv1D(filters=nbfilter, kernel_size=7, activation='relu')(gcn4_3)
        # graph_features_3 = AveragePooling1D(pool_size=5)(graph_features_3)
        # graph_features_3 = Dropout(0.25)(graph_features_3)

        # x = Concatenate(axis=-1)([x, embedding])
        # x_gcn = Concatenate(axis=-1)([graph_features, graph_features_2, graph_features_3])
        x = Concatenate(axis=-1)([x, embedding, graph_features_2])
        # x = Concatenate(axis=-1)([embedding, graph_features_2])
        # x = Concatenate(axis=-1)([x, embedding, graph_features])
        # x = Concatenate(axis=-1)([x, embedding, graph_features_2])
        # x1 = Concatenate(axis=-1)([x, graph_features_2])
        # # x2 = Concatenate(axis=-1)([embedding, graph_features_2])
        # x3 = Concatenate(axis=-1)([x, embedding, graph_features])
        # x4 = Concatenate(axis=-1)([x, embedding, graph_features_3])
        # o0 = Concatenate(axis=-1)([x2, x3, x4, embedding, graph_features_2])
        # o1 = Concatenate(axis=-1)([x1, x3, x4, embedding, graph_features_2])
        # o2 = Concatenate(axis=-1)([x1, x2, x4, embedding, graph_features_2])
        # o3 = Concatenate(axis=-1)([x1, x2, x3, embedding, graph_features_2])
        # o4 = Concatenate(axis=-1)([x1, x2, x3, x4, embedding, graph_features_2_1])
        # o5 = Concatenate(axis=-1)([x1, x2, x3, x4, embedding, graph_features_2_2])
        # o6 = Concatenate(axis=-1)([x1, x2, x3, x4, embedding, graph_features_2_3])
        # o7 = Concatenate(axis=-1)([x1, x2, x3, x4, embedding, graph_features_2_5])

        # x_list = [o0, o1, o2, o3, o4, o5, o6, o7]
        # x = x_list[i]

        # Conv_features_aligned = Reshape((19, 102))(Conv_features)
        # x = Concatenate(axis=-1)([x, embedding, Conv_features])




        # print(embedding.shape)
        # profile = Conv1D(filters=nbfilter, kernel_size=3, padding='same')(embedding)
        # profile = BatchNormalization(axis=-1)(profile)
        # profile = Activation('relu')(profile)
        # # print(x.shape)
        # # print(profile.shape)
        # x = Concatenate(axis=-1)([x, profile])
        # 使用自定义的 MultiScaleLayer
        # x = MultiScaleLayer()(x)
        # x = Concatenate(axis=-1)([x, x_gcn])
        # 添加 Bidirectional GRU 层
        x = Bidirectional(GRU(hiddensize, return_sequences=True))(x)


        x = Flatten(name='HH')(x)
        # 全连接层
        x = Dense(nbfilter, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(2, activation='softmax')(x)

        # # 添加 Attention 层
        # # 注意，Attention 层的输入应该是列表
        # attention_output = Attention()([x, x])  # 将 x 作为 query 和 value
        #
        # 将注意力的输出展平
        # x = Flatten()(x)

        # # 替换原来的全连接层
        # x = FastKANLayer(input_dim=4560, output_dim=nbfilter)(x)  # hiddensize * 2 是 Bidirectional 的输出维度
        # x = Dropout(0.3)(x)
        # x = FastKANLayer(input_dim=nbfilter, output_dim=int(nbfilter/2))(x)  # 最终的输出维度为2
        # x = Dropout(0.3)(x)
        # x = FastKANLayer(input_dim=int(nbfilter/2), output_dim=2)(x)  # 最终的输出维度为2
        # 替换后的代码
        # 构建 FastKAN 模型
        # layers_hidden = [4560, nbfilter, 2]
        # fastkan = FastKAN(layers_hidden)
        # #
        # # # 使用 FastKAN 替换原来的层
        # x = fastkan(x)
        #x = Dropout(0.3)(x)
        #
        # # 输出层
        # outputs = Activation('softmax')(x)

        #构建模型
        # model = Model(inputs=[inputs1_1,inputs1_2,inputs1_3,inputs1_4, inputs2, inputs3], outputs=outputs)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
        # 打印模型概述
        model.summary()

        print('model training')
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

        # 记录训练的起始时间
        train_start_time = time.time()

        # 开始训练
        history = model.fit(train_X, train_y, batch_size=batch_size, epochs=n_epochs, verbose=1,
                            validation_data=(eval_X, eval_y), callbacks=[earlystopper])

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
        plt.title('CSGN PUM2')
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        # plt.colorbar()
        # 保存图像

        i += 1
        plt.savefig(f'../plot/pca_csgn_pum2{i}.png', dpi=300)
        plt.show()


        # 记录训练结束时间
        train_end_time = time.time()

        # 计算总的训练时间
        total_train_time = train_end_time - train_start_time

        # 获取实际运行的 epoch 数（如果早停，可能小于 n_epochs）
        actual_epochs = len(history.epoch)

        # 计算每个 epoch 的平均时间
        average_epoch_time = total_train_time / actual_epochs

        print(f'Total training time: {total_train_time:.2f} seconds')
        print(f'Average time per epoch: {average_epoch_time:.2f} seconds')

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


    # if len(fpr_list) == 5:
        # roc_data = np.column_stack((fpr_list[-1], tpr_list[-1]))
        # np.savetxt('../ROC curve and AUC value/ninputs1/{}(AUC={:.4f}).txt'.format(protein, np.mean(aucs)), roc_data, delimiter='\t')
        # 创建包含五种指标平均值的数组
        # metrics_data = np.array([np.mean(aucs), np.mean(Acc), np.mean(precision1), np.mean(recall1), np.mean(fscore1)])
        #
        # # 保存到文件
        # np.savetxt('../ROC curve and AUC value/noinputs1/{}(AUC={:.4f}).txt'.format(protein, np.mean(aucs)),
        #            metrics_data.reshape(1, -1),
        #            fmt='%.4f', delimiter='\t', header='AUC\tACC\tPrecision\tRecall\tF1-Score')

    print("acid AUC: %.4f " % np.mean(aucs), protein)
    print("acid ACC: %.4f " % np.mean(Acc), protein)
    print("acid precision1: %.4f " % np.mean(precision1), protein)
    print("acid recall1: %.4f " % np.mean(recall1), protein)
    print("acid fscore1: %.4f " % np.mean(fscore1), protein)

# 定义函数parse_arguments(parser)，用于解析命令行参数。
def parse_arguments(parser, protein):
    parser.add_argument('--protein', type=str, default= protein,metavar='<data_file>',
                        help='the protein for training model')
    parser.add_argument('--nbfilter', type=int, default=102, help='use this option for CNN convolution')
    parser.add_argument('--hiddensize', type=int, default=120, help='use this option for LSTM')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--n_epochs', type=int, default=30, help='The number of training epochs (default value: 30)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # proteins = ['AUF1', 'DGCR8', 'EWSR1', 'FMRP', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP3', 'LIN28A', 'MOV10', 'PTB', 'PUM2', 'TAF15', 'TDP43', 'U2AF65']
    # proteins = ['EWSR1', 'FMRP', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP3', 'LIN28A', 'MOV10', 'PTB', 'PUM2', 'TAF15', 'TDP43', 'U2AF65']
    # proteins = ['HUR', 'IGF2BP1', 'IGF2BP3', 'LIN28A', 'MOV10', 'PTB', 'PUM2', 'TAF15', 'TDP43', 'U2AF65']
    # proteins = ['PUM2']
    proteins = ['PUM2']
    # # for i in range(2, 8):
    # i = 1
    # print(i, "th model will be train ...")
    for protein in proteins:
        parser = argparse.ArgumentParser()
        args = parse_arguments(parser, protein)
        print(protein,"will be train ...")
        run_CRIP(args)
