import keras
from keras import ops


def supervoxel_connectivity_penalty(
    y_true,
    y_pred,
    critical_channel=2,
    foreground_class_id=1,
    ignore_label=2,
    smooth=1e-6,
):
    """Penalize low confidence on critical supervoxels that preserve topology.

    Args:
        y_true: Tensor with channels [mask, skeleton, critical_supervoxels].
        y_pred: Prediction tensor with class channels.
        critical_channel: Channel index for critical supervoxel mask.
        foreground_class_id: Class id for foreground (ink) probability.
        ignore_label: Integer id to ignore in the mask channel.
        smooth: Numerical stability term.

    Returns:
        Scalar penalty averaged across the batch.
    """
    y_true_mask = y_true[..., 0]
    critical_mask = y_true[..., critical_channel]
    pred_prob = y_pred[..., foreground_class_id]

    valid_mask = ops.cast(y_true_mask != ignore_label, pred_prob.dtype)
    critical_mask = critical_mask * valid_mask

    spatial_axes = tuple(range(1, len(pred_prob.shape)))
    critical_sum = ops.sum(critical_mask, axis=spatial_axes)
    pred_sum = ops.sum(pred_prob * critical_mask, axis=spatial_axes)

    has_critical = ops.cast(critical_sum > 0, pred_prob.dtype)
    mean_pred = pred_sum / (critical_sum + smooth)
    loss = (1.0 - mean_pred) * has_critical

    return ops.mean(loss)


class SupervoxelConnectivityLoss(keras.losses.Loss):
    """Wrapper loss for supervoxel connectivity penalty."""

    def __init__(
        self,
        critical_channel=2,
        foreground_class_id=1,
        ignore_label=2,
        smooth=1e-6,
        name="supervoxel_connectivity_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.critical_channel = critical_channel
        self.foreground_class_id = foreground_class_id
        self.ignore_label = ignore_label
        self.smooth = smooth

    def call(self, y_true, y_pred):
        return supervoxel_connectivity_penalty(
            y_true,
            y_pred,
            critical_channel=self.critical_channel,
            foreground_class_id=self.foreground_class_id,
            ignore_label=self.ignore_label,
            smooth=self.smooth,
        )
