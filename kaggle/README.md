# Kaggle Baseline Update Notes

This document summarizes the changes applied to the Kaggle baseline and the
TransUNet stack, along with the expected impact on the evaluation metrics.

## What changed

1. **Surface-aware skip gating (TransUNet)**  
   Skip connections can now apply a boundary-aware gate that emphasizes soft
   morphological edges. This is meant to directly target **SurfaceDice@τ** by
   amplifying boundary-relevant features during decoding.

2. **Surface Dice loss term**  
   A differentiable, soft-boundary Dice loss is added to the training objective.
   The loss is computed on morphological boundaries of predictions and labels
   to encourage better surface alignment, which maps to the leaderboard’s
   SurfaceDice@τ component.

3. **Skeleton Recall loss integration**  
   The Skeleton Recall Loss (using precomputed tubed skeletons) is retained and
   wired as a reusable loss module. This targets the **TopoScore** component
   without heavy GPU overhead.

4. **Composite loss (baseline)**  
   The baseline now blends:
   - Sparse Dice + Cross-Entropy
   - Skeleton Recall
   - Surface Dice (boundary)
   - A modest false-positive volume penalty

## Why these changes map to the score components

| Leaderboard component | Primary change targeting it |
| --- | --- |
| **TopoScore** | Skeleton Recall loss + topology-guided skip gating |
| **SurfaceDice@τ** | Surface Dice loss + surface-aware skip gating |
| **VOI (split/merge)** | Skeleton Recall + FP penalty (reduces spurious components) |

## Expected impact (honest range)

These additions are intended to **nudge** the score upward without increasing
memory consumption or changing the 160×160×160 crop size. Based on the behavior
seen so far (TopoScore and VOI improving by ~0.002–0.003 each), a realistic
estimate is **+0.006 to +0.015** total score if the new loss terms converge as
intended. This puts the expected score in the **~0.544–0.553** range.  

**Important:** This still does **not guarantee** crossing 0.60. The current
changes are low-risk, topology- and surface-focused improvements. A >0.60 score
likely requires additional, higher-impact changes (stronger data augmentation,
longer training, or architectural scaling), which are outside this minimal
update.

## Checklist for training

- Keep the input crop at **160×160×160**.
- Ensure the training loader adds the **tubed skeleton channel**.
- Use `use_topology_guidance=True` and `use_surface_gating=True` in TransUNet.
- Train for enough epochs to let the surface loss stabilize (surface loss tends
  to improve later in training).
