from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    MultiHeadAttention, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Activation, Masking, Input, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

@dataclass
class TransformerConfig:
    max_length: int  # 1つのデータの長さ（シーケンス長）
    d_model: int  # 入力の特徴量の次元数
    key_dim: int  # 各アテンションヘッドの次元数
    num_heads: int  # アテンションヘッドの数
    ff_dim: int  # FeedForwardネットワークの中間次元数
    num_transformer_blocks: int  # Encoderブロックの数
    dropout: float  # ドロップアウト率
    l2_lambda: float  # L2正則化の係数
    pad: float # パディングの値
    pooling: str  # プーリングの方法（'average' または 'max'）
    task: str  # 実行するタスクの種類（'binary'、'multiclass'、'regression'）
    num_classes: int = None  # 多クラス分類の場合のクラス数（タスクが 'multiclass' の場合に必須）
    regression_units: int = None  # 回帰予測の場合の出力ユニット数（タスクが 'regression' の場合に使用）
    layer_norm_epsilon: float = 1e-6  # レイヤー正規化のイプシロン値

def get_positional_encoding(max_len, d_model):
    # 0からmax_len-1までの各位置番号を縦ベクトルとして作成する
    positions = np.arange(max_len).reshape(-1, 1)
    # 0からd_model-1までの各次元番号を横ベクトルとして作成する
    dims = np.arange(d_model).reshape(1, -1)
    # 各次元で使う角度の割合を計算する
    angle_rates = 1 / (10000 ** ((2 * (dims // 2)) / float(d_model)))
    # 各位置番号と角度割合の掛け算により、位置エンコーディングの元になる値を求める
    pos_matrix = positions * angle_rates
    # 偶数の次元にはsin関数を適用する
    pos_matrix[:, 0::2] = np.sin(pos_matrix[:, 0::2])
    # 奇数の次元にはcos関数を適用する
    pos_matrix[:, 1::2] = np.cos(pos_matrix[:, 1::2])
    # バッチ次元を追加し、TensorFlowのfloat32型に変換して返す
    return tf.cast(pos_matrix[np.newaxis, ...], dtype=tf.float32)

def create_attention_mask(inputs, pad_value):
    # 各要素がpad_valueと等しくないかどうかを判定する（True: 等しくない、False: 等しい）
    not_pad = tf.math.not_equal(inputs, pad_value)
    # 最後の次元ごとに全ての値がpad_valueでないかをチェックする
    # すなわち、あるタイムステップの全特徴量がpad_valueでなければTrueとなる
    valid = tf.reduce_all(not_pad, axis=-1, keepdims=True)
    # 真偽値を浮動小数点数に変換する（True→1.0、False→0.0）
    mask = tf.cast(valid, tf.float32)
    return mask

def transformer_encoder(x, config: TransformerConfig):
    # 入力xに対して、パディング部分を無視するためのアテンションマスクを作成する
    mask_layer = Lambda(lambda x: create_attention_mask(x, config.pad))(x)

    # マルチヘッドアテンション層を適用する
    # Query、Key、Valueすべてに同じ入力xを用い、先に作成したmask_layerでパディング部分を無視する
    attn_out = MultiHeadAttention(
        key_dim=config.key_dim,
        num_heads=config.num_heads,
        dropout=config.dropout
    )(x, x, x, attention_mask=mask_layer)

    # MultiHeadAttentionの出力にドロップアウトを適用して、過学習を防ぐ
    attn_out = Dropout(config.dropout)(attn_out)

    # 残差接続を行い、元の入力xとMultiHeadAttentionの出力を足し合わせた後、層正規化を実施する
    out1 = LayerNormalization(epsilon=config.layer_norm_epsilon)(x + attn_out)

    # Feed Forward Networkの1段目：中間次元に拡大し、活性化関数reluを適用する
    ff_out = Dense(config.ff_dim, activation='relu', kernel_regularizer=l2(config.l2_lambda))(out1)

    # Feed Forward Networkの2段目：元の次元数に戻す
    ff_out = Dense(out1.shape[-1], kernel_regularizer=l2(config.l2_lambda))(ff_out)

    # Feed Forward Networkの出力に対してもドロップアウトを適用する
    ff_out = Dropout(config.dropout)(ff_out)

    # 再び残差接続を行い、先の出力out1とFeed Forward Networkの出力を足してから層正規化を施す
    out2 = LayerNormalization(epsilon=config.layer_norm_epsilon)(out1 + ff_out)

    # 最終的な出力を返す
    return out2

def build_transformer_model(config: TransformerConfig):
    # 入力層を定義する。シーケンス長と各タイムステップの特徴量次元はconfigから設定される
    inputs = Input(shape=(config.max_length, config.d_model), name="input_layer")

    # Masking層を適用する。これは、パディングされた値（config.pad）を無視するための処理である
    x = Masking(mask_value=config.pad, name="masking_layer")(inputs)

    # 位置エンコーディングを取得する。入力の各位置に位置情報を加えるために用いる
    pos_encoding = get_positional_encoding(config.max_length, config.d_model)
    # 位置エンコーディングを入力xに足し合わせる。これで各タイムステップが自身の位置情報を持つ
    x = x + pos_encoding[:, :config.max_length, :]

    # configで指定された回数だけTransformerのEncoderブロックを積み重ねる
    for _ in range(config.num_transformer_blocks):
        x = transformer_encoder(x, config)

    # プーリング処理を行う。各タイムステップの情報を1つのベクトルに集約する
    if config.pooling == 'average':
        x = GlobalAveragePooling1D(name="global_average_pooling")(x)
    elif config.pooling == 'max':
        x = GlobalMaxPooling1D(name="global_max_pooling")(x)

    # 集約後にReLU活性化を適用し、非線形性を加える
    x = Activation('relu')(x)
    # ドロップアウトを適用して、過学習を防止する
    x = Dropout(config.dropout, name="dropout_after_pooling")(x)

    # タスクの種類に応じて出力層のユニット数と活性化関数を設定するためのマッピング
    task_mapping = {
        "binary": (1, "sigmoid"),
        "multiclass": (config.num_classes, "softmax"),
        "regression": (config.regression_units, "linear"),
    }

    # 指定されたタスクがサポートされていない場合はエラーを出す
    if config.task not in task_mapping:
        raise ValueError(f"Unsupported task type '{config.task}'. Choose from {list(task_mapping.keys())}.")

    # タスクに合わせた出力層を定義する
    units, activation = task_mapping[config.task]
    outputs = Dense(units, activation=activation, kernel_regularizer=l2(config.l2_lambda),
                    name="output_layer")(x)

    # 入力と出力を指定してモデルを作成する
    model = Model(inputs=inputs, outputs=outputs, name="transformer_model")
    return model
