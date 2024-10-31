from sklearn.decomposition import PCA
from four import *
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Bidirectional
from keras.layers.convolutional import Convolution1D, AveragePooling1D
import argparse
from sklearn.manifold import TSNE
from keras.layers import Lambda
from sklearn.model_selection import KFold
from keras.callbacks import  EarlyStopping

# 配置GPU和TensorFlow会话：
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# 定义函数run_CRIP(parser)，该函数用于运行CRIP模型的训练和评估过程。它接受一个parser参数，用于解析命令行参数。
def run_CRIP(parser):
    # 首先从命令行参数中获取要处理的蛋白质名称、批处理大小、隐藏层大小、训练轮数等信息。
    protein = parser.protein
    batch_size = parser.batch_size
    hiddensize = parser.hiddensize
    n_epochs = parser.n_epochs
    nbfilter = parser.nbfilter
    trainXeval, test_X, trainYeval, test_y = dealwithdata(protein)  # 使用dealwithdata(protein)函数处理数据，得到训练集和测试集的特征和标签。
    kf = KFold(n_splits=5)

    for train_index, eval_index in kf.split(trainYeval):
        train_X = trainXeval[train_index]
        train_y = trainYeval[train_index]
        eval_X = trainXeval[eval_index]
        eval_y = trainYeval[eval_index]
        print('configure cnn network')
        model = Sequential()  # 创建一个序贯模型model = Sequential()，用于构建卷积神经网络模型。
    # 添加卷积层Convolution1D到模型中，指定输入维度为56、输入长度为101，卷积核数量为nbfilter，卷积核长度为7，边界模式为"valid"，激活函数为ReLU，子采样长度为1。
        model.add(
        Convolution1D(input_dim=56, input_length=101, nb_filter=nbfilter, filter_length=7, border_mode="valid",
                      ####
                      activation="relu", subsample_length=1))  # 56*101
    # 添加平均池化层AveragePooling1D到模型中，指定池化窗口大小为5。
        model.add(AveragePooling1D(pool_size=5))
    # 添加Dropout层Dropout到模型中，设置丢弃率为0.5，用于防止过拟合。
        model.add(Dropout(0.5))
    # model.add(LSTM(128, input_dim=102, input_length=31, return_sequences=True))
    # 添加双向GRU层Bidirectional(GRU)到模型中，隐藏层大小为hiddensize，设置返回完整序列。
        model.add(Bidirectional(GRU(hiddensize, return_sequences=True)))
    # model.add(Bidirectional(LSTM(hiddensize, return_sequences=True)))
    # 添加展平层Flatten到模型中，用于将输入展平为一维向量。
        model.add(Flatten())
    # 添加全连接层Dense到模型中，神经元数量为nbfilter，激活函数为ReLU。
        #model.add(Dense(nbfilter, activation='relu'))
    # -------------------------------------------------------------------------- #
    # 添加Lambda层，用于提取全连接层的输出作为t-SNE的输入
        model.add(Lambda(lambda x: x))
    # 编译模型
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4))
    # 在训练完成后，获取Lambda层的输出作为特征向量
        print('model training')
        features = model.predict(test_X)
        # 使用PCA进行降维
        pca = PCA(n_components=2)
        new_data = pca.fit_transform(features)

        # 绘制降维后的数据
    plt.scatter(new_data[:, 0], new_data[:, 1],c=test_y[:, 1],cmap='coolwarm' ,s=10.0,marker='^')
    #plt.xlabel('Principal Component 1')
    #plt.ylabel('Principal Component 2')
    plt.title('circ-FHN PUM2')
    plt.show()
def parse_arguments(parser):
    # 创建一个命令行参数解析器，并添加protein、nbfilter、hiddensize、batch_size、n_epochs等参数。
    parser.add_argument('--protein', type=str, metavar='<data_file>', required=True,
                        help='the protein for training model')
    parser.add_argument('--nbfilter', type=int, default=102, help='use this option for CNN convolution')
    parser.add_argument('--hiddensize', type=int, default=120, help='use this option for LSTM')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='The size of a single mini-batch (default value: 50)')
    parser.add_argument('--n_epochs', type=int, default=30, help='The number of training epochs (default value: 30)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    run_CRIP(args)










