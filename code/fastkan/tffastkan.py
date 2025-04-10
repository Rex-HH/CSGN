import tensorflow as tf
import numpy as np

# 引入 FastKANLayer 类
class SplineLinear(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, init_scale=0.1, **kwargs):
        super(SplineLinear, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = init_scale

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.in_features, self.out_features),
            initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=self.init_scale),
            trainable=True,
            name="weight"
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.weight)

class RadialBasisFunction(tf.keras.layers.Layer):
    def __init__(self, grid_min=-2., grid_max=2., num_grids=8, denominator=None, **kwargs):
        super(RadialBasisFunction, self).__init__(**kwargs)
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def build(self, input_shape):
        grid = np.linspace(self.grid_min, self.grid_max, self.num_grids).astype(np.float32)
        self.grid = tf.convert_to_tensor(grid, dtype=tf.float32)

    def call(self, x):
        x = tf.expand_dims(x, -1)
        return tf.exp(-((x - self.grid) / self.denominator) ** 2)

class FastKANLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        grid_min=-2.,
        grid_max=2.,
        num_grids=8,
        use_base_update=True,
        use_layernorm=True,
        base_activation=tf.nn.silu,
        spline_weight_init_scale=0.1,
        **kwargs
    ):
        super(FastKANLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_base_update = use_base_update
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)

        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = tf.keras.layers.Dense(output_dim)

    def call(self, x, use_layernorm=True):
        # 确保输入 x 的形状符合预期
        assert len(x.shape) == 2 and x.shape[-1] == self.input_dim, \
            f"输入形状不正确: x.shape={x.shape}，期望形状为 (batch_size, {self.input_dim})"

        # 确保经过 spline_basis 计算后的形状一致
        spline_basis = self.rbf(x)
        spline_basis = tf.reshape(spline_basis, [-1, self.input_dim * self.rbf.num_grids])
        #print(f"Shape of spline_basis: {spline_basis.shape}")

        ret = self.spline_linear(spline_basis)
        # print(f"Shape of ret: {ret.shape}")

        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            # print(f"Shape of base: {base.shape}")

            # 如果形状不一致，则调整形状
            if ret.shape != base.shape:
                # print(f"Shape mismatch detected. Retrying with reshaped base...")
                base = tf.reshape(base, tf.shape(ret))
                # print(f"Reshaped base to match ret: {base.shape}")

            ret = ret + base  # 这里是出现错误的地方
        return ret


class FastKAN(tf.keras.Model):
    def __init__(
        self,
        layers_hidden,
        grid_min=-2.,
        grid_max=2.,
        num_grids=8,
        use_base_update=True,
        base_activation=tf.nn.silu,
        spline_weight_init_scale=0.1,
        **kwargs
    ):
        super(FastKAN, self).__init__(**kwargs)
        self.mylayers = []
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.mylayers.append(FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale
            ))

    def call(self, x):
        for layer in self.mylayers:
            x = layer(x)
        return x


class AttentionWithFastKANTransform(tf.keras.Model):
    def __init__(
        self,
        q_dim,
        k_dim,
        v_dim,
        head_dim,
        num_heads,
        gating=True,
        **kwargs
    ):
        super(AttentionWithFastKANTransform, self).__init__(**kwargs)
        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        else:
            self.linear_g = None
        # precompute the 1/sqrt(head_dim)
        self.norm = 1. / np.sqrt(head_dim)

    def call(self, q, k, v, bias=None):
        wq = tf.reshape(self.linear_q(q) * self.norm, [-1, 1, self.num_heads, -1])
        wk = tf.reshape(self.linear_k(k), [-1, k.shape[-2], self.num_heads, -1])
        att = tf.nn.softmax(tf.reduce_sum(wq * wk, axis=-1), axis=-2)
        if bias is not None:
            att += bias[..., None]

        wv = tf.reshape(self.linear_v(v), [-1, v.shape[-2], self.num_heads, -1])
        o = tf.reduce_sum(att[..., None] * wv, axis=-3)
        o = tf.reshape(o, [-1, self.num_heads * o.shape[-1]])

        if self.linear_g is not None:
            g = tf.sigmoid(self.linear_g(q))
            o = g * o

        o = self.linear_o(o)
        return o
