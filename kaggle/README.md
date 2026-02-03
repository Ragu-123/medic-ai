# Kaggle baseline notes

## Supervoxel connectivity penalty

The training pipeline now computes a critical-supervoxel mask on CPU using NumPy/scikit-image
(`tf.numpy_function` + `slic` + 3D skeleton analysis). The loss adds a lightweight supervoxel
connectivity term that discourages low foreground probability inside critical supervoxels
identified near skeleton endpoints and junctions. This acts as a topology-preservation prior
without adding GPU memory pressure on 160³ crops.

**Expected VOI impact:** modest reduction in VOI (fewer split/merge errors) when the base model
misses thin junctions. The effect is typically in the low single-digit percent range, but is
data-dependent. The penalty is weighted at `0.1` to avoid over-regularization.

**Compute overhead:** the CPU preprocessing adds a small per-sample cost (supervoxel SLIC and
skeleton neighbor counting). On typical 160³ patches, this is expected to be seconds per sample
on CPU during data loading, while GPU training memory remains unchanged.
