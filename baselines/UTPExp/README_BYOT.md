# BYOT Self-Distillation for UTPExp

## Overview
- Implements Bring-Your-Own-Teacher (BYOT) style self-distillation for forecasting in `UTPExp`.
- Two loss sources only:
  - `loss_gt`: hard data loss against ground truth (pinball in quantile mode or Huber with `delta=2`).
  - `loss_soft`: soft distillation loss aligning student heads to the teacher (Quantile‑Huber Distillation).
- No L2 feature hints are used.

## Architecture
- Auxiliary prediction heads are attached to selected transformer layers.
- The final prediction head is the teacher; heads at configured layers are students.
- Configure which layers act as students via `AUX_HEAD_LAYERS` (list of indices in `[0, num_layers-1]`).

## Configuration
Add `BYOT` to `CFG.MODEL.PARAM` in `config/utp_base.py`:

```
BYOT = {
  'ENABLED': True,
  'AUX_HEAD_LAYERS': [3, 7],
  'LOSS_WEIGHTS': {
    'loss_gt': 1.0,
    'loss_soft': 0.2
  }
}
```

- `ENABLED`: turn BYOT training on.
- `AUX_HEAD_LAYERS`: which transformer layers have auxiliary student heads. Deduplicated and validated.
- `LOSS_WEIGHTS`: global weights for combining types. Same‑type losses are averaged equally; do not set per‑head weights.

## Losses
### `loss_gt` (hard loss)
- Quantile mode: masked pinball loss per head, averaged across horizon and quantiles.
- Point mode: masked Huber with `delta=2` per head, averaged across horizon.

### `loss_soft` (soft distillation)
- Quantile‑Huber Distillation (QHD): directly align student and teacher quantile vectors.
- For each time step, compute `mean_i Huber(student_q[i] − teacher_q[i], delta=2)`; apply label mask; average over horizon and batch.
- If quantiles are not configured, fallback to masked Huber between student and teacher point/median predictions.
- Rationale vs KL:
  - KL needs density estimation from quantiles and can suffer from weak gradients when variance is small.
  - QHD operates in quantile space matching the model head, provides smooth, robust gradients, and naturally aligns in magnitude with Huber‑based `loss_gt` (both use `delta=2`).

## Logging
- Training and validation meters record per‑layer losses:
  - `train/loss_gt_head_<idx>`, `train/loss_soft_student_<idx>`
  - `val/loss_gt_head_<idx>`, `val/loss_soft_student_<idx>`
- The runner also keeps `train/val huberloss` on teacher median for reference.

## Usage
1. Set `BYOT.ENABLED=True` and choose `AUX_HEAD_LAYERS`.
2. Start training; the combined loss is:
   `L_total = LOSS_WEIGHTS.loss_gt * sum(loss_gt_head_i) + LOSS_WEIGHTS.loss_soft * sum(loss_soft_student_j)`.
3. Monitor per‑layer meters and adjust the two global weights to balance magnitudes as needed.

## Notes
- Same‑type losses are naturally equally weighted and averaged; no per‑head weights.
- Feature L2 hint losses are intentionally excluded to avoid forcing intermediate tokens to mimic final tokens.
- Inference and evaluation use only the teacher head, preserving baseline behavior.
