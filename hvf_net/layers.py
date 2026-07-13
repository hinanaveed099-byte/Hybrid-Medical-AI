"""Custom layers for HVF-Net."""

from tensorflow.keras import layers


class CrossAttentionLayer(layers.Layer):
    """Lightweight consistency layer between classifier and specialist streams."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.query_dense = layers.Dense(units, activation="relu")
        self.key_dense = layers.Dense(units, activation="relu")
        self.value_dense = layers.Dense(units, activation="relu")
        self.out_dense = layers.Dense(units, activation="relu")

    def call(self, inputs):
        cls_feat, spec_feat, img_feat = inputs
        query = self.query_dense(cls_feat)
        key = self.key_dense(spec_feat)
        value = self.value_dense(img_feat)
        score = layers.Dot(axes=-1)([query, key])
        score = layers.Activation("sigmoid")(score)
        attended = layers.Multiply()([value, score])
        merged = layers.Concatenate()([cls_feat, spec_feat, attended])
        return self.out_dense(merged)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
