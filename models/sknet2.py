# 2025/9/17
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

# baseline
def ts_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = SKConv(M=3, f=64)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.4)(x)
    x = SKConv(M=3, f=64)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation='swish', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.4)(x)
    x = transformer_token_selection(x, num_heads=8, attn_drop=0.4,  expansion_ratio=3)
    x = layers.LayerNormalization(epsilon=1e-6, name='transformer_LayerNormalization')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(2, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="base_model" )
    return model



def ts_model_ours(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = SKConv(M=3, f=64)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.4)(x)
    x = SKConv(M=3, f=64)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation='swish', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.4)(x)
    x = transformer_token_selection_ours(x, num_heads=8, attn_drop=0.4,  expansion_ratio=3)
    x = layers.LayerNormalization(epsilon=1e-6, name='transformer_LayerNormalization')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(2, activation="softmax", name="classifier", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="base_model_ours" )
    return model

def hybrid_ts_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = SKConv(M=3, f=64)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.4)(x)
    x = SKConv(M=3, f=64)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation='swish', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.4)(x)
    x = transformer_token_selection(x, num_heads=8, attn_drop=0.4,  expansion_ratio=3)
    x = layers.LayerNormalization(epsilon=1e-6, name='transformer_LayerNormalization')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
    x = layers.Dropout(0.4)(x)

    features = layers.Dense(64, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
    outputs_A = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="projection")(features)
    outputs_B = layers.Dense(2, activation="softmax", name="classifier", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = tf.keras.Model(
        inputs=inputs, outputs=[outputs_A, outputs_B], name="hybrid_model" )
    return model


def hybrid_ts_model_ours(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = SKConv(M=3, f=64)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.4)(x)
    x = SKConv(M=3, f=64)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation='swish', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.4)(x)
    x = transformer_token_selection_ours(x, num_heads=8, attn_drop=0.4,  expansion_ratio=3)
    x = layers.LayerNormalization(epsilon=1e-6, name='transformer_LayerNormalization')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
    x = layers.Dropout(0.4)(x)

    features = layers.Dense(64, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
    outputs_A = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="projection")(features)
    outputs_B = layers.Dense(2, activation="softmax", name="classifier", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = tf.keras.Model(
        inputs=inputs, outputs=[outputs_A, outputs_B], name="hybrid_model" )
    return model


def SKConv(M=2,  f=32):

  def wrapper(inputs):   # b, h, w

    d = max(f//4, 32)
    x = inputs
    filters=f
    xs = []
    for m in range(1, M+1):

      _x = layers.Conv1D(filters, 3, dilation_rate=m, padding='same', kernel_initializer='he_normal', use_bias=False,)(x)
      _x = layers.BatchNormalization()(_x)
      _x = layers.Activation('swish')(_x)
      xs.append(_x)

    U = layers.Add()(xs)
    s = keras.backend.mean(U, axis=1, keepdims=True)

    z = layers.Conv1D(d, 1,)(s)
    z = layers.BatchNormalization()(z)  #
    z = layers.Activation('swish')(z)

    x = layers.Conv1D(filters*M, 1, )(z)
    x = layers.Reshape([ 1, filters, M] )(x)
    scale = layers.Softmax()(x)
    x = keras.backend.stack(xs, axis=-1)
    x = Axpby()([scale, x])

    return x
  return wrapper

def SKConv2d(M=2,  filters=24):

  def wrapper(inputs):   # b, h, w

    d = max(filters//3, 8)
    x = inputs
    xs = []
    for m in range(1, M+1):

      _x = layers.Conv2D(filters, m*2-1 , padding='same', kernel_initializer='he_normal', use_bias=False,)(x)
      _x = layers.BatchNormalization()(_x)
      _x = layers.Activation('swish')(_x)
      xs.append(_x)

    U = layers.Add()(xs)
    s = keras.backend.mean(U, axis=(1,2), keepdims=True)

    z = layers.Conv2D(d, 1,)(s)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('swish')(z)

    x = layers.Conv2D(filters*M, 1, )(z)
    x = layers.Reshape([ 1, 1, filters, M] )(x)
    scale = layers.Softmax()(x)
    x = keras.backend.stack(xs, axis=-1)
    x = Axpby()([scale, x])

    return x
  return wrapper


class Axpby(layers.Layer):
  """
  Do this:
    F = a * X + b * Y + ...
    Shape info:
      a:  B x 1 x 1 x C
      X:  B x H x W x C
      b:  B x 1 x 1 x C
      Y:  B x H x W x C
      ...
      F:  B x H x W x C
  """
  def __init__(self, **kwargs):
        super(Axpby, self).__init__(**kwargs)

  def build(self, input_shape):
        super(Axpby, self).build(input_shape)  # Be sure to call this at the end

  def call(self, inputs):
    """ scale: [B, 1, 1, C, M]
        x: [B, H, W, C, M]
    """
    scale, x = inputs
    f = tf.multiply(scale, x, name='product')
    f =  tf.reduce_sum(f, axis=-1, name='sum')
    return f

  def compute_output_shape(self, input_shape):
    return input_shape[0:4]


def token_selection(inputs, num_heads=8, attn_drop=0.4,  expansion_ratio=3):
    # inputs [None, 27, 128]
    B, N, C = inputs.shape
    head_dim = C // num_heads
    scale = head_dim ** -0.5
    qkv_reshape = layers.Reshape((N, 3, num_heads, C // num_heads))  # 3 is (q,k,v)    C//num_heads (head qkv )
    out_reshape = layers.Reshape((N, C))

    # transpose -> [3, batch_size, num_heads, N, C/num_heads ]
    qkv = layers.Dense(input_dim=C, units=C * 3 )(inputs)
    qkv = tf.transpose(qkv_reshape(qkv), perm=[2, 0, 3, 1, 4])
    q, k, v = qkv[0], qkv[1], qkv[2]

    # q     [ batch_size, num_heads, N, C/num_heads ]   @  [batch_size, num_heads, C / num_heads, N]
    # attn  [ batch_size, num_heads, N, N ]
    attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * scale   #
    attn = layers.Softmax(axis=-1)(attn)
    attn = layers.Dropout(attn_drop)(attn)

    # Global Token Selection
    #  [batch_size, N, N, num_heads] -> [batch_size, N, N, num_heads * ratio]
    attn_expansion = layers.Dense(input_dim=num_heads, units=num_heads * expansion_ratio)(
        tf.transpose(attn, perm=[0, 2, 3, 1]))
    # attn_expansion = tf.transpose(attn_expansion, perm=[0, 3, 1, 2])

    # Local Token Selection
    # [batch_size,  N, N, num_heads * ratio]
    temp = layers.Conv2D(num_heads * expansion_ratio, kernel_size=1)(attn_expansion)
    temp = layers.Activation('relu')(temp)
    attn_head_wise_matrix = layers.BatchNormalization(axis=3)(temp)

    # Reduction
    # [batch_size, N, N, num_heads * ratio]  -> [batch_size, N, N, num_heads]
    # transpose: -> [batch_size, num_heads, N, N]
    attn = layers.Dense(input_dim=num_heads * expansion_ratio, units=num_heads, )(attn_head_wise_matrix)  # 2025/1/18修改
    attn = tf.transpose(attn, perm=[0, 3, 1, 2])

    # attn @ v ->  [ batch_size, num_heads, N, C/num_heads ]
    # transpose -> [ batch_size, N , num_heads, C/num_heads ]
    # reshape -> [ batch_size, N , C ]
    x = out_reshape(tf.transpose(attn @ v, perm=[0, 2, 1, 3]))

    return x


def token_selection_ours(inputs, num_heads=8, attn_drop=0.4,  expansion_ratio=3):
    # inputs [None, 27, 128]
    B, N, C = inputs.shape
    head_dim = C // num_heads
    scale = head_dim ** -0.5
    qkv_reshape = layers.Reshape((N, 3, num_heads, C // num_heads))
    out_reshape = layers.Reshape((N, C))

    # transpose -> [3, batch_size, num_heads, N, C/num_heads ]
    qkv = layers.Dense(input_dim=C, units=C * 3 )(inputs)
    qkv = tf.transpose(qkv_reshape(qkv), perm=[2, 0, 3, 1, 4])
    q, k, v = qkv[0], qkv[1], qkv[2]

    # q     [ batch_size, num_heads, N, C/num_heads ]   @  [batch_size, num_heads, C / num_heads, N]
    # attn  [ batch_size, num_heads, N, N ]
    attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2])) * scale
    attn = layers.Softmax(axis=-1)(attn)
    attn = layers.Dropout(attn_drop)(attn)

    # Global Token Selection
    #  [batch_size, N, N, num_heads] -> [batch_size, N, N, num_heads * ratio]
    attn_expansion = layers.Dense(input_dim=num_heads, units=num_heads * expansion_ratio)(
        tf.transpose(attn, perm=[0, 2, 3, 1]))
    # attn_expansion = tf.transpose(attn_expansion, perm=[0, 3, 1, 2])

    # Local Token Selection
    # [batch_size,  N, N, num_heads * ratio]
    temp = SKConv2d(M=3, filters=num_heads * expansion_ratio)(attn_expansion)
    temp = layers.Activation('relu')(temp)
    attn_head_wise_matrix = layers.BatchNormalization(axis=3)(temp)

    # Reduction
    # [batch_size, N, N, num_heads * ratio]  -> [batch_size, N, N, num_heads]
    # transpose: -> [batch_size, num_heads, N, N]
    attn = layers.Dense(input_dim=num_heads * expansion_ratio, units=num_heads)(attn_head_wise_matrix)
    attn = tf.transpose(attn, perm=[0, 3, 1, 2])

    # attn @ v ->  [ batch_size, num_heads, N, C/num_heads ]
    # transpose -> [ batch_size, N , num_heads, C/num_heads ]
    # reshape -> [ batch_size, N , C ]
    x = out_reshape(tf.transpose(attn @ v, perm=[0, 2, 1, 3]))
    x = layers.Dense(input_dim=C, units=C)(x)

    return x



def transformer_encoder(inputs, head_size=64, num_heads=4, dropout=0.4):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout     # key_dim    key_dim * num_heads <= token_feature_dim
    )(x, x)
    x = layers.Dropout(0.5)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(input_dim=inputs.shape[-1], units=inputs.shape[-1] * 2 )(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(input_dim=inputs.shape[-1] * 2, units=inputs.shape[-1] )(x)
    return x + res


def transformer_token_selection(inputs, num_heads=8, attn_drop=0.4,  expansion_ratio=3):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = token_selection(x, num_heads=num_heads, attn_drop=attn_drop,  expansion_ratio=expansion_ratio)
    x = layers.Dropout(0.5)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(input_dim=inputs.shape[-1], units=inputs.shape[-1] * 2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(input_dim=inputs.shape[-1] * 2, units=inputs.shape[-1])(x)
    return x + res

def transformer_token_selection_ours(inputs, num_heads=8, attn_drop=0.4,  expansion_ratio=3):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = token_selection_ours(x, num_heads=num_heads, attn_drop=attn_drop,  expansion_ratio=expansion_ratio)
    x = layers.Dropout(0.5)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(input_dim=inputs.shape[-1], units=inputs.shape[-1] * 2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(input_dim=inputs.shape[-1] * 2, units=inputs.shape[-1])(x)
    return x + res