"""HVF-Net model architecture."""

from tensorflow.keras import Model, layers, metrics
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

from hvf_net.layers import CrossAttentionLayer


def build_hvf_net(
    image_shape=(224, 224, 3),
    num_classes: int = 6,
    route_vocab_size: int = 4,
    learning_rate: float = 1e-3,
):
    image_input = layers.Input(shape=image_shape, name="image")
    classifier_input = layers.Input(shape=(num_classes,), name="classifier_probs")
    specialist_input = layers.Input(shape=(3,), name="specialist_features")
    route_input = layers.Input(shape=(1,), dtype="int32", name="route_id")

    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
    )
    backbone.trainable = False

    img_feat = layers.GlobalAveragePooling2D(name="image_pool")(backbone.output)
    img_feat = layers.Dense(128, activation="relu", name="image_dense")(img_feat)

    cls_feat = layers.Dense(64, activation="relu")(classifier_input)
    cls_feat = layers.Dropout(0.3)(cls_feat)
    cls_feat = layers.Dense(32, activation="relu", name="classifier_branch")(cls_feat)

    spec_feat = layers.Dense(32, activation="relu")(specialist_input)
    spec_feat = layers.Dropout(0.2)(spec_feat)
    spec_feat = layers.Dense(16, activation="relu", name="specialist_branch")(spec_feat)

    route_feat = layers.Embedding(route_vocab_size, 8, name="route_embedding")(route_input[:, 0])
    route_feat = layers.Flatten()(route_feat)

    attn_feat = CrossAttentionLayer(name="cross_attention")([cls_feat, spec_feat, img_feat])

    fused = layers.Concatenate(name="fusion_concat")(
        [img_feat, cls_feat, spec_feat, route_feat, attn_feat]
    )
    fused = layers.Dense(64, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    fused = layers.Dense(32, activation="relu", name="fusion_dense")(fused)

    trust_output = layers.Dense(1, activation="sigmoid", name="trust")(fused)
    action_output = layers.Dense(3, activation="softmax", name="action")(fused)

    model = Model(
        inputs=[image_input, classifier_input, specialist_input, route_input],
        outputs=[trust_output, action_output],
        name="HVF_Net",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            "trust": "binary_crossentropy",
            "action": "sparse_categorical_crossentropy",
        },
        loss_weights={"trust": 1.0, "action": 0.5},
        metrics={
            "trust": ["accuracy", metrics.AUC(name="auc")],
            "action": ["accuracy"],
        },
    )
    return model
