from keras import ops

from medicai.utils import DescribeMixin, soft_dilate, soft_erode

from .base import BASE_COMMON_ARGS, BaseLoss


def _soft_boundary(mask):
    boundary = soft_dilate(mask) - soft_erode(mask)
    return ops.relu(boundary)


class BaseSurfaceDiceLoss(BaseLoss):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name or "surface_dice_loss",
            **kwargs,
        )

    def compute_loss(self, y_true, y_pred, mask):
        spatial_dims = list(range(1, len(y_pred.shape) - 1))
        y_true_boundary = _soft_boundary(y_true)
        y_pred_boundary = _soft_boundary(y_pred)

        intersection = ops.sum(mask * y_true_boundary * y_pred_boundary, axis=spatial_dims)
        union = ops.sum(mask * y_true_boundary, axis=spatial_dims) + ops.sum(
            mask * y_pred_boundary, axis=spatial_dims
        )
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score


class SparseSurfaceDiceLoss(BaseSurfaceDiceLoss, DescribeMixin):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        return y_pred

    def _process_targets(self, y_true):
        if y_true.shape[-1] == 1:
            y_true = ops.squeeze(y_true, axis=-1)
        y_true = ops.one_hot(y_true, num_classes=self.num_classes)
        return y_true


class CategoricalSurfaceDiceLoss(BaseSurfaceDiceLoss, DescribeMixin):
    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.softmax(y_pred, axis=-1)
        return y_pred


class BinarySurfaceDiceLoss(BaseSurfaceDiceLoss, DescribeMixin):
    def __init__(
        self,
        from_logits,
        num_classes,
        target_class_ids=None,
        ignore_class_ids=None,
        smooth=1e-7,
        reduction="mean",
        name=None,
        **kwargs,
    ):
        if ignore_class_ids is not None and num_classes > 1:
            raise ValueError(
                "`ignore_class_ids` is only supported when `num_classes=1` "
                "(binary or sparse segmentation). One-hot or multi-label cases "
                "with `num_classes > 1` are not supported."
            )
        super().__init__(
            from_logits=from_logits,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class_ids=ignore_class_ids,
            smooth=smooth,
            reduction=reduction,
            name=name or "binary_surface_dice_loss",
            **kwargs,
        )

    def _process_predictions(self, y_pred):
        if self.from_logits:
            return ops.sigmoid(y_pred)
        return y_pred


CATEGORICAL_SURFACE_LOSS_DOCSTRING = """Surface Dice loss for categorical (one-hot encoded) labels.

The loss approximates surface alignment by computing a soft morphological
boundary for each class and applying the Dice loss to those boundaries.

""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="categorical_surface_dice_loss"
)

SPARSE_SURFACE_LOSS_DOCSTRING = """Surface Dice loss for sparse categorical labels.

The loss approximates surface alignment by computing a soft morphological
boundary for each class and applying the Dice loss to those boundaries.

""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="sparse_surface_dice_loss"
)

BINARY_SURFACE_LOSS_DOCSTRING = """Surface Dice loss for binary or multi-label segmentation.

The loss approximates surface alignment by computing a soft morphological
boundary for each class and applying the Dice loss to those boundaries.

""" + BASE_COMMON_ARGS.format(
    specific_args="", default_name="binary_surface_dice_loss"
)

CategoricalSurfaceDiceLoss.__doc__ = CATEGORICAL_SURFACE_LOSS_DOCSTRING
SparseSurfaceDiceLoss.__doc__ = SPARSE_SURFACE_LOSS_DOCSTRING
BinarySurfaceDiceLoss.__doc__ = BINARY_SURFACE_LOSS_DOCSTRING
