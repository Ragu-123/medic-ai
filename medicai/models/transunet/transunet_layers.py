import keras
from keras import layers, ops

from medicai.layers import TransUNetMLP
from medicai.utils import soft_skeletonize


class MaskedCrossAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim=1024, dropout_rate=0.1, **kwargs):
        """
        A masked cross-attention block for the TransUNet decoder.

        This layer performs masked cross-attention between a set of queries and the
        transformer encoder's output, followed by a multi-layer perceptron (MLP).
        The attention mask is used to selectively attend to relevant parts of the
        encoder output, refining the queries based on spatial context.

        The refinement loop in the decoder uses this layer to iteratively update
        learnable queries, which are then used to predict segmentation masks.

        Args:
            embed_dim (int): The embedding dimension of the input queries and keys/values.
            num_heads (int): The number of attention heads. `embed_dim` must be
                divisible by `num_heads`.
            mlp_dim (int, optional): The hidden dimension of the MLP. Defaults to 1024.
            dropout_rate (float, optional): The dropout rate for attention and MLP.
                Defaults to 0.1.
            **kwargs: Keyword arguments passed to the parent `layers.Layer` class.

        Inputs:
            A list of three tensors:
            - queries (Tensor): The query tensor, typically the learnable queries.
                Shape: `(batch_size, num_queries, embed_dim)`.
            - encoder_output (Tensor): The output from the ViT encoder.
                Shape: `(batch_size, num_tokens, embed_dim)`.
            - mask (Tensor): The attention mask to be applied. A `1` indicates a
                valid token to attend to, while a `0` or negative infinity value
                masks it out. Shape: `(batch_size, num_queries, num_tokens)`.

        Outputs:
            A tensor with the same shape as the `queries` input, representing the
            queries after being refined by the cross-attention and MLP.
            Shape: `(batch_size, num_queries, embed_dim)`.
        """
        super().__init__(**kwargs)

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Input shape should be [query_shape, encoder_output_shape, mask_shape]
        query_shape, encoder_output_shape, _ = input_shape
        self.cross_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            value_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout_rate,
        )
        self.cross_attention.build(query_shape, encoder_output_shape, encoder_output_shape)
        self.mlp_layer = TransUNetMLP(
            self.mlp_dim,
            activation="gelu",
            output_dim=self.embed_dim,
            drop_rate=self.dropout_rate,
            name="mlp_decode",
        )
        self.mlp_layer.build(query_shape)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm1.build((None, None, self.embed_dim))
        self.layernorm2.build((None, None, self.embed_dim))

    def call(self, inputs, training=None):
        # Inputs should be [queries, encoder_output, mask]
        queries, encoder_output, mask = inputs

        # 1. Cross-attention with mask
        queries_norm = self.layernorm1(queries)

        attn_output = self.cross_attention(
            query=queries_norm,
            key=encoder_output,
            value=encoder_output,
            attention_mask=mask,
            training=training,
        )

        # Residual connection
        x = queries + attn_output

        # 2. MLP
        x_norm = self.layernorm2(x)
        mlp_output = self.mlp_layer(x_norm, training=training)
        # Residual connection
        output = x + mlp_output

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class LearnableQueries(layers.Layer):
    """Initializes and provides a set of learnable query tokens.

    These tokens are learnable parameters of the model, initialized with random
    values. During the `call` method, they are tiled to match the batch size
    of the input tensor, providing a unique set of queries for each sample in the batch.

    Args:
        num_queries: The number of learnable queries to create.
        embed_dim: The dimensionality of each query token.

    Inputs:
        Any tensor, used only to infer the batch size for tiling the queries.

    Outputs:
        A tensor of shape `(batch_size, num_queries, embed_dim)` containing
        the tiled learnable queries.
    """

    def __init__(self, num_queries, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.queries = self.add_weight(
            name="learnable_queries",
            shape=[1, self.num_queries, self.embed_dim],
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        output = ops.broadcast_to(self.queries, [batch_size, self.num_queries, self.embed_dim])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_queries, self.embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"num_queries": self.num_queries, "embed_dim": self.embed_dim})
        return config


class TopologyGatedSkip(layers.Layer):
    """Topology-guided gating for skip features using soft skeletonization.

    This layer predicts a gating mask from decoder features and optionally
    sharpens it with differentiable soft skeletonization. The resulting mask
    reweights skip connections to emphasize thin, connected structures.

    Inputs:
        - decoder_features: Tensor with shape (B, *spatial, C).
        - skip_features: Tensor with shape (B, *spatial, C_skip).
    Outputs:
        - gated skip features with the same shape as skip_features.
    """

    def __init__(
        self,
        spatial_dims,
        use_skeleton=True,
        skeleton_iters=10,
        gate_activation="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if spatial_dims not in (2, 3):
            raise ValueError(f"TopologyGatedSkip supports 2D or 3D, got {spatial_dims}D.")
        self.spatial_dims = spatial_dims
        self.use_skeleton = use_skeleton
        self.skeleton_iters = skeleton_iters
        self.gate_activation = gate_activation

    def build(self, input_shape):
        if self.spatial_dims == 2:
            self.gate_conv = layers.Conv2D(1, 1, padding="same", name="topo_gate_conv")
        else:
            self.gate_conv = layers.Conv3D(1, 1, padding="same", name="topo_gate_conv")
        self.gate_activation_layer = layers.Activation(self.gate_activation, name="topo_gate_act")

    def call(self, inputs):
        decoder_features, skip_features = inputs
        gate = self.gate_conv(decoder_features)
        gate = self.gate_activation_layer(gate)
        if self.use_skeleton:
            skeleton = soft_skeletonize(gate, self.skeleton_iters)
            gate = 0.5 * (gate + skeleton)
        gated_skip = skip_features * gate
        return gated_skip

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spatial_dims": self.spatial_dims,
                "use_skeleton": self.use_skeleton,
                "skeleton_iters": self.skeleton_iters,
                "gate_activation": self.gate_activation,
            }
        )
        return config
