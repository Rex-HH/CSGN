import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class GraphConvolution(Layer):
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        # 创建 GCN 层中的权重
        node_features_shape = input_shape[0][-1]  # 获取输入的节点特征的形状
        self.kernel = self.add_weight(shape=(node_features_shape, self.units),
                                      initializer='glorot_uniform',
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer='zeros',
                                        name='bias')
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs, training=False):
        node_features, adjacency_matrix = inputs
        # 进行矩阵运算：X' = A * X * W
        h = tf.matmul(node_features, self.kernel)  # X * W
        output = tf.matmul(adjacency_matrix, h)    # A * (X * W)

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.units

    def get_config(self):
        config = super(GraphConvolution, self).get_config()
        config.update({"units": self.units,
                       "activation": tf.keras.activations.serialize(self.activation),
                       "use_bias": self.use_bias})
        return config
