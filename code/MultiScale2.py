from tensorflow import add
from tensorflow.python.keras.layers import Dropout, Convolution1D, BatchNormalization, Concatenate

from tensorflow_addons.utils.types import Activation
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Concatenate, Add, Layer
from tensorflow.keras.models import Sequential

def ConvolutionBlock(input, f, k):
    input_conv = Convolution1D(filters=f, kernel_size=k, padding='same')(input)
    input_bn = BatchNormalization(axis=-1)(input_conv)
    input_at = Activation('relu')(input_bn)
    input_dp = Dropout(0.4)(input_at)
    return input_dp


class MultiScaleLayer(Layer):
    def __init__(self, **kwargs):
        super(MultiScaleLayer, self).__init__(**kwargs)

        # 在 __init__ 中创建所有层，这样每次调用时就不会重复创建
        self.conv_a = Conv1D(64, 1, padding='same', activation='relu')

        self.conv_b1 = Conv1D(64, 1, padding='same', activation='relu')
        self.conv_b2 = Conv1D(64, 3, padding='same', activation='relu')

        self.conv_c1 = Conv1D(64, 1, padding='same', activation='relu')
        self.conv_c2 = Conv1D(64, 5, padding='same', activation='relu')
        self.conv_c3 = Conv1D(64, 5, padding='same', activation='relu')

        self.conv_d1 = Conv1D(64, 1, padding='same', activation='relu')
        self.conv_d2 = Conv1D(64, 7, padding='same', activation='relu')
        self.conv_d3 = Conv1D(64, 7, padding='same', activation='relu')
        self.conv_d4 = Conv1D(64, 7, padding='same', activation='relu')

        self.conv_e1 = Conv1D(64, 1, padding='same', activation='relu')
        self.conv_e2 = Conv1D(64, 3, padding='same', activation='relu')
        self.conv_e3 = Conv1D(64, 5, padding='same', activation='relu')
        self.conv_e4 = Conv1D(64, 7, padding='same', activation='relu')
        self.conv_e5 = Conv1D(64, 9, padding='same', activation='relu')
        #
        # self.conv_f1 = Conv1D(64, 1, padding='same', activation='relu')
        # self.conv_f2 = Conv1D(64, 3, padding='same', activation='relu')
        # self.conv_f3 = Conv1D(64, 5, padding='same', activation='relu')
        # self.conv_f4 = Conv1D(64, 7, padding='same', activation='relu')
        # self.conv_f5 = Conv1D(64, 9, padding='same', activation='relu')
        # self.conv_f6 = Conv1D(64, 11, padding='same', activation='relu')
        # #
        # self.conv_g1 = Conv1D(64, 1, padding='same', activation='relu')
        # self.conv_g2 = Conv1D(64, 3, padding='same', activation='relu')
        # self.conv_g3 = Conv1D(64, 5, padding='same', activation='relu')
        # self.conv_g4 = Conv1D(64, 7, padding='same', activation='relu')
        # self.conv_g5 = Conv1D(64, 9, padding='same', activation='relu')
        # self.conv_g6 = Conv1D(64, 15, padding='same', activation='relu')
        # self.conv_g7 = Conv1D(64, 17, padding='same', activation='relu')

        self.shortcut_conv = Conv1D(320, 1, padding='same')
        self.bn = BatchNormalization()
        self.concat = Concatenate(axis=-1)
        self.add = Add()
        self.act = Activation('relu')

    def call(self, inputs):
        # 只使用已经定义的层，而不是在 call() 中创建新层
        A = self.conv_a(inputs)

        B = self.conv_b1(inputs)
        B = self.conv_b2(B)

        C = self.conv_c1(inputs)
        C = self.conv_c2(C)
        C = self.conv_c3(C)

        D = self.conv_d1(inputs)
        D = self.conv_d2(D)
        D = self.conv_d3(D)
        D = self.conv_d4(D)

        E = self.conv_e1(inputs)
        E = self.conv_e2(E)
        E = self.conv_e3(E)
        E = self.conv_e4(E)
        E = self.conv_e5(E)
        #
        # F = self.conv_f1(inputs)
        # F = self.conv_f2(F)
        # F = self.conv_f3(F)
        # F = self.conv_f4(F)
        # F = self.conv_f5(F)
        # F = self.conv_f6(F)
        #
        # G = self.conv_g1(inputs)
        # G = self.conv_g2(G)
        # G = self.conv_g3(G)
        # G = self.conv_g4(G)
        # G = self.conv_g5(G)
        # G = self.conv_g6(G)
        # G = self.conv_g7(G)

        # 合并所有分支
        merged = self.concat([A, B, C, D, E])
        # merged = self.concat([A, B, C, D, E, F, G])

        # 添加残差连接
        shortcut_y = self.shortcut_conv(inputs)
        shortcut_y = self.bn(shortcut_y)

        # 相加并激活
        result = self.add([shortcut_y, merged])
        result = self.act(result)

        return result