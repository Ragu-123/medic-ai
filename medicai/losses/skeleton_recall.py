import keras
from keras import ops

from medicai.utils import DescribeMixin


class SkeletonRecallLoss(keras.losses.Loss, DescribeMixin):
    """Skeleton recall loss for topology-aware segmentation.

    This loss expects `y_true` to provide both the segmentation mask and a
    precomputed tubed skeleton channel, as described in Skeleton Recall Loss.
    """

    def __init__(
        self,
        num_classes,
        foreground_class_id=1,
        mask_channel=0,
        skeleton_channel=1,
        ignore_class_ids=None,
        from_logits=False,
        smooth=1e-6,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name or "skeleton_recall_loss", reduction=reduction, **kwargs)
        self.num_classes = num_classes
        self.foreground_class_id = foreground_class_id
        self.mask_channel = mask_channel
        self.skeleton_channel = skeleton_channel
        self.ignore_class_ids = ignore_class_ids
        self.from_logits = from_logits
        self.smooth = smooth

        if self.num_classes == 1:
            self.foreground_class_id = 0

    def call(self, y_true, y_pred):
        if y_true.shape[-1] <= max(self.mask_channel, self.skeleton_channel):
            raise ValueError(
                "SkeletonRecallLoss expects y_true to contain mask and skeleton channels."
            )

        y_true_mask = y_true[..., self.mask_channel]
        y_true_skeleton = y_true[..., self.skeleton_channel]

        if self.from_logits:
            if self.num_classes == 1:
                y_pred = ops.sigmoid(y_pred)
            else:
                y_pred = ops.softmax(y_pred, axis=-1)

        pred_fg = y_pred[..., self.foreground_class_id]

        if self.ignore_class_ids:
            ignore_classes = ops.convert_to_tensor(self.ignore_class_ids, dtype="int32")
            is_ignored_mask = ops.any(
                ops.equal(y_true_mask[..., None], ignore_classes),
                axis=-1,
            )
            valid_mask = ops.cast(~is_ignored_mask, y_pred.dtype)
        else:
            valid_mask = ops.ones_like(pred_fg, dtype=pred_fg.dtype)

        intersection = ops.sum(
            pred_fg * y_true_skeleton * valid_mask,
            axis=tuple(range(1, len(pred_fg.shape))),
        )
        skeleton_sum = ops.sum(
            y_true_skeleton * valid_mask,
            axis=tuple(range(1, len(pred_fg.shape))),
        )

        has_skeleton = ops.cast(skeleton_sum > 0, pred_fg.dtype)
        recall = (intersection + self.smooth) / (skeleton_sum + self.smooth)
        loss = ops.mean((1.0 - recall) * has_skeleton)
        return loss


SkeletonRecallLoss.__doc__ += """

Args:
    num_classes: Number of prediction classes.
    foreground_class_id: Class index to treat as foreground for the skeleton recall.
    mask_channel: Channel index of the segmentation mask in `y_true`.
    skeleton_channel: Channel index of the precomputed skeleton in `y_true`.
    ignore_class_ids: Optional list of class IDs to ignore when computing recall.
    from_logits: Whether `y_pred` is provided as logits.
    smooth: Smoothing constant to avoid divide-by-zero.
"""
