# Prepared checkpoint for the change-of-mind lab

`early.pt` is a 2.4 MiB teaching fixture using the project's real
`PolicyValueNetwork` (one residual block, eight channels). It is deliberately
biased toward a pawn capture. It is **not** a checkpoint from human games or
evidence of chess strength. No download or training run is needed to use it.

The reproducible preparation is in `build_checkpoint()` in
`selfplay/mind_change_lab.py`:

- CPU, one PyTorch thread, seed 7.
- Start with the usual randomly initialised network.
- Train for 48 Adam steps at learning rate 0.001 on ONE position:
  `4r1k1/p4ppp/8/8/8/6P1/5P1P/R5K1 w - - 0 1`.
- The move target is `a1a7` (Rxa7). The value target is zero, a preparation
  convention, not a proven evaluation. The loss is hard-target cross-entropy
  plus squared value error. BatchNorm running statistics remain fixed.
- This board has a white pawn on g3, giving the king an escape square after
  Rxa7 Re1+. The lab's original and similar mate-threat boards are absent
  from preparation.

Generated with Python 3.11 and PyTorch 2.6.0 on an ARM CPU. To reproduce
without overwriting the shipped artifact:

```bash
python3 selfplay/mind_change_lab.py --build-checkpoint /tmp/mind-change-early.pt
python3 selfplay/mind_change_lab.py --checkpoint /tmp/mind-change-early.pt
```

The generator records the seed, preparation position, steps and PyTorch
version in the checkpoint. Different library versions or processors can
change rounding and search tie-breaking; the shipped weights are the
reference. Tests check the important behaviour, not exact decimals.

The lab updates an in-memory copy and never saves over this file. Its 60
updates use ONLY the original mate-threat position's search policy. There
is no final game result, so it uses the policy component of the existing
`soft_target_loss()` and does not invent a value label. Shared network
features can still change the value output. This isolates one policy lesson;
it is not a full self-play iteration.

The escape-square board is a control seen during checkpoint preparation,
not an unseen test set. The similar board with Black's king on h8 is unseen
in both preparation and the lab's updates. Neither probe is used for those
updates. Lowering the capture probability on all three boards illustrates
why transfer alone is not proof of useful generalisation.
