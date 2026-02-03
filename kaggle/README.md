# Kaggle Notes

## Loss composition

We add a lightweight clDice (centerline Dice) term to the composite loss to
encourage topology-preserving predictions for thin, tubular structures, as
introduced by Shit et al. (CVPR 2021). In this baseline we use
`SparseCenterlineDiceLoss` with `target_class_ids=1` and `ignore_class_ids=2`,
weighted at **0.1** so it complements (but does not dominate) Dice/CE and the
skeleton recall term.
