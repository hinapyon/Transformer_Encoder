from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LayerNormalization, Dropout, Input, Concatenate, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Transformer_Encoder.py から必要な関数やクラスをインポート
from Transformer_Encoder import transformer_encoder, TransformerConfig, get_positional_encoding

def extract_patches(images, patch_size):
    # バッチサイズを取得
    batch_size = tf.shape(images)[0]

    # tf.image.extract_patches関数を使用して画像からパッチを抽出
    # sizes: 抽出するパッチのサイズ
    # strides: パッチ間の間隔（連続しないようにpatch_sizeと同じ値を設定）
    # rates: 抽出時のサンプリングレート（1ならすべてのピクセルを使用）
    # padding: パディング方法（'VALID'は端のピクセルを無視する）
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    # 抽出されたパッチの次元数を取得
    # 形状は (batch_size, num_patches_y, num_patches_x, patch_size*patch_size*channels)
    patch_dims = patches.shape[-1]

    # パッチをリシェイプして (batch_size, num_patches, patch_dim) の形状にする
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])

    return patches

def build_vit_model(
    image_height,     # 入力画像の高さ
    image_width,      # 入力画像の幅
    patch_size,        # パッチのサイズ（正方形のパッチ）
    num_classes,     # 分類クラス数
    d_model,          # 埋め込み次元（ViT-Base では768など）
    num_heads,         # アテンションヘッドの数
    ff_dim,          # Feed Forward Network の中間次元（通常 d_model の4倍）
    num_layers,        # Encoderブロックの数
    dropout_rate,     # ドロップアウト率
    l2_lambda=1e-4,       # L2正則化の係数
    layer_norm_epsilon=1e-6  # レイヤー正規化のイプシロン値
):
    # 1. 画像入力層の定義
    inputs = Input(shape=(image_height, image_width, 3), name="input_image")

    # 2. 画像をパッチに分割する
    patches = Lambda(
        lambda img: extract_patches(img, patch_size),
        name="extract_patches"
    )(inputs)

    # 3. 静的なパラメータの計算
    # 入力画像のチャネル数（通常は3: RGB）
    channels = int(inputs.shape[-1])
    # 縦横のパッチ数を計算（画像サイズがpatch_sizeで割り切れることが前提）
    num_patches_y = image_height // patch_size
    num_patches_x = image_width // patch_size
    # 全パッチ数
    num_patches = num_patches_y * num_patches_x

    # 4. パッチ埋め込み：各パッチをd_model次元のベクトルに線形変換
    patch_embeddings = Dense(
        d_model,
        kernel_regularizer=l2(l2_lambda),
        name="patch_embedding"
    )(patches)

    # 5. 分類トークン[CLS]の追加
    # 学習可能なパラメータとして初期化
    cls_token = tf.Variable(
        initial_value=tf.zeros((1, 1, d_model)),
        trainable=True,
        name="cls_token"
    )

    # バッチサイズに合わせてブロードキャスト
    cls_tokens = Lambda(
        lambda x: tf.broadcast_to(cls_token, [tf.shape(x)[0], 1, d_model]),
        name="broadcast_cls_token"
    )(patch_embeddings)

    # CLSトークンとパッチ埋め込みを結合
    embeddings = Concatenate(axis=1, name="concat_embeddings")([cls_tokens, patch_embeddings])

    # 6. 位置エンコーディングの付与
    # 位置エンコーディングを生成（CLSトークンを含めた全シーケンス長に対応）
    pos_encoding = get_positional_encoding(num_patches + 1, d_model)
    # 埋め込みに位置エンコーディングを加算
    x = embeddings + pos_encoding[:, :num_patches+1, :]

    # 7. Transformer設定の構成
    config = TransformerConfig(
        max_length=num_patches + 1,  # CLSトークンを含めたシーケンス長
        d_model=d_model,             # モデルの次元数
        key_dim=d_model // num_heads, # 各ヘッドの次元数
        num_heads=num_heads,         # アテンションヘッドの数
        ff_dim=ff_dim,               # フィードフォワードネットワークの中間次元数
        num_transformer_blocks=1,    # 内部で使用するブロック数（ここでは1とし、外部ループで制御）
        dropout=dropout_rate,        # ドロップアウト率
        l2_lambda=l2_lambda,         # L2正則化係数
        pad=0.0,                     # パディング値（ViTでは通常使わない）
        pooling='average',           # プーリング方法（ViTでは使わないが設定は必要）
        task='multiclass',           # タスク（分類）
        num_classes=num_classes,     # クラス数
        layer_norm_epsilon=layer_norm_epsilon  # レイヤー正規化のイプシロン
    )

    # 8. Transformerエンコーダブロックを指定回数積み重ねる
    for _ in range(num_layers):
        x = transformer_encoder(x, config)

    # 9. 分類ヘッドの構築
    # CLSトークンの出力を取得（シーケンスの最初の位置）
    cls_output = x[:, 0, :]
    # CLSトークンの出力に対してレイヤー正規化を適用
    cls_output = LayerNormalization(
        epsilon=layer_norm_epsilon,
        name="cls_norm"
    )(cls_output)
    # ドロップアウトを適用して過学習を防止
    cls_output = Dropout(dropout_rate, name="cls_dropout")(cls_output)
    # 最終的な分類層
    outputs = Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=l2(l2_lambda),
        name="classifier"
    )(cls_output)

    # 10. モデルの構築
    model = Model(inputs=inputs, outputs=outputs, name="VisionTransformer")

    return model
