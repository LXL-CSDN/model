# model.py
# 基于现有结构，加入：logits 输出（默认）、L2 正则、SpatialDropout1D、Head Dropout
# 与训练脚本配套：在 origin_train.py 中使用 logits 版损失 + AdamW

from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation,
    Add, Flatten, Dense, SpatialDropout1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np


# ===== 全局可调的正则 & Dropout 超参 =====
L2_STRENGTH = 1e-4          # 卷积/全连接的 L2 正则
STEM_SPATIAL_DROPOUT = 0.15 # Stem 后的 SpatialDropout1D
HEAD_DROPOUT = 0.30         # 分类头前的 Dropout
DEFAULT_KEEP_PROB = 0.8     # ResidualUnit 默认 keep_prob（与原始代码一致，等价 Dropout=0.2）


class ResidualUnit(object):
    """Residual unit block (unidimensional).
    与原实现一致，额外：为所有 Conv1D 加上 L2 正则。
    参数说明见原注释。
    """

    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
                 dropout_keep_prob=DEFAULT_KEEP_PROB, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # 1x1 conv to match channels
            y = Conv1D(
                self.n_filters_out, 1, padding='same',
                use_bias=False, kernel_initializer=self.kernel_initializer,
                kernel_regularizer=l2(L2_STRENGTH)
            )(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)

        # 1st layer
        x = Conv1D(
            self.n_filters_out, self.kernel_size, padding='same',
            use_bias=False, kernel_initializer=self.kernel_initializer,
            kernel_regularizer=l2(L2_STRENGTH)
        )(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(
            self.n_filters_out, self.kernel_size, strides=downsample,
            padding='same', use_bias=False, kernel_initializer=self.kernel_initializer,
            kernel_regularizer=l2(L2_STRENGTH)
        )(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


def get_model(n_classes, last_layer=None,
              kernel_size=16, kernel_initializer='he_normal',
              stem_spatial_dropout=STEM_SPATIAL_DROPOUT,
              head_dropout=HEAD_DROPOUT,
              residual_keep_prob=DEFAULT_KEEP_PROB):
    """
    参数
    ----
    n_classes : int
        输出类别数。
    last_layer : None | 'sigmoid'
        - None（默认）：输出 logits（无激活），适用于 logits 版损失（推荐）
        - 'sigmoid'：输出概率（向后兼容）
    其余超参与原模型保持一致。
    """
    signal = Input(shape=(4096, 12), dtype=np.float32, name='signal')

    # Stem
    x = Conv1D(
        64, kernel_size, padding='same', use_bias=False,
        kernel_initializer=kernel_initializer, kernel_regularizer=l2(L2_STRENGTH)
    )(signal)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if stem_spatial_dropout and stem_spatial_dropout > 0:
        x = SpatialDropout1D(stem_spatial_dropout)(x)

    # Residual stacks（保持你的原始通道/长度配置）
    x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer,
                        dropout_keep_prob=residual_keep_prob)([x, x])
    x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer,
                        dropout_keep_prob=residual_keep_prob)([x, y])
    x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer,
                        dropout_keep_prob=residual_keep_prob)([x, y])
    x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer,
                        dropout_keep_prob=residual_keep_prob)([x, y])

    # Head
    x = Flatten()(x)
    if head_dropout and head_dropout > 0:
        x = Dropout(head_dropout)(x)

    # 输出层：默认 logits（无激活）。如需概率输出，传 last_layer='sigmoid'
    activation = None if (last_layer is None) else last_layer
    name = 'logits' if activation is None else 'prob'
    diagn = Dense(
        n_classes,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(L2_STRENGTH),
        name=name
    )(x)

    model = Model(signal, diagn)
    return model


if __name__ == "__main__":
    # 默认 logits 输出
    m = get_model(5)
    m.summary()
