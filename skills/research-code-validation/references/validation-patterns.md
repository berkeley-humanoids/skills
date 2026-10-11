# Validation patterns

Use only the sections relevant to the current pipeline. The policies below adapt the source note into operational guidance; the linked documentation supports language and framework behavior, not a universal guarantee of experimental correctness.

## Contents

- [Boundary checks](#boundary-checks)
- [Loss and gradient diagnostics](#loss-and-gradient-diagnostics)
- [Tiny-problem and reproducibility checks](#tiny-problem-and-reproducibility-checks)
- [Evaluation accounting](#evaluation-accounting)
- [Recovery and checkpointing](#recovery-and-checkpointing)
- [Sources](#sources)

## Boundary checks

Check semantic contracts as well as tensor dimensions. A valid shape does not establish that images and actions are time-aligned, joint arrays use the same order, or poses use the same coordinate frame.

| Boundary | Example checks |
| --- | --- |
| Dataset to model | Nonempty splits; no unintended train/eval overlap; matching sample IDs and timestamps; expected channel order, normalization, mask meaning, and label domain. |
| Policy to simulator/controller | Joint order, action meaning and units, reference frame, control rate/decimation, finite state, and correct termination/truncation handling. |
| Loss to optimizer | Scalar reduction and nonempty valid denominator; finite objective; gradients on intended parameters; intended optimizer parameter groups. |
| Checkpoint to runner | Compatible model/config/data identity; complete requested resume state; explicit distinction between resuming and initializing a new run from weights. |
| Runner to metrics | Correct planned/completed/failed counts; aggregation weights; visible worker errors; matching comparison configuration. |

Use an explicit exception for a required runtime contract, for example:

```python
if prediction.shape != target.shape:
    raise ValueError(
        f"prediction shape {prediction.shape} != target shape {target.shape}"
    )
```

Use this only when the objective requires matching shapes; classification logits and class-index labels normally have different shapes. Python's [`assert` statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement) is removed under optimization. Do not depend on it to reject invalid experiments.

## Loss and gradient diagnostics

### Initial loss

Derive the expectation from the implemented objective, target distribution, weighting, masking, reduction, and initialization.

- For ordinary unweighted cross-entropy with normalized targets and exactly uniform predictions over `C` classes, the per-example loss is `log(C)`. Randomly initialized logits are not necessarily uniform. Use this as a controlled baseline or sanity range, not an exact assertion for every random network.
- Account for class weighting, smoothing, reduction, ignored entries, and regularization before comparing losses. With no valid entries, reject the reduction unless the algorithm explicitly defines that case.
- For regression, diffusion, or flow matching, derive a baseline from target scaling and the actual prediction target. Fix noise and timesteps for a diagnostic when their resampling would obscure the result.
- For RL, inspect component rewards, bootstrapping, masks, returns, and a known small transition sequence. A policy-gradient loss need not be positive, monotonic, or converge to zero.

The [`CrossEntropyLoss` contract](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html) defines supported target shapes and reductions. Do not assume its soft-target interface validates every probability constraint for you.

### Ordinary PyTorch update

Adapt this fragment to an existing training step. It illustrates an unscaled, dense-gradient diagnostic; it is not a generic trainer or an AMP/distributed implementation.

```python
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(inputs), targets)
if loss.ndim != 0 or not torch.isfinite(loss).item():
    raise FloatingPointError(f"invalid loss at step {step}: {loss.detach()}")
loss.backward()
for name, parameter in model.named_parameters():
    if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
        raise FloatingPointError(f"nonfinite gradient at step {step}: {name}")
optimizer.step()
```

In the diagnostic case, also verify that the intended parameters receive gradients and actually update. A missing gradient may be correct for a frozen or conditional branch; check the known contract rather than requiring every parameter to update on every batch. Handle sparse gradients with the appropriate value representation. Do not apply dense-gradient code blindly.

### Mixed precision and distributed execution

Follow the installed framework's [AMP procedure](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html): scale the loss, backpropagate, and unscale before gradient inspection or clipping. When accumulating gradients, unscale once after all microbatches for that optimizer update; do not mix scaled and unscaled gradients during accumulation.

`GradScaler.step()` can intentionally skip an update containing nonfinite gradients while the scale adapts. Record overflow/skipped-update counts and actual optimizer-update counts; do not advance update-based schedules as though a skipped update happened. Use an explicit diagnostic policy for persistent overflow or lack of progress. A nonfinite forward loss still requires investigation; do not sanitize it to force a step.

In distributed training, use the runner's error propagation and whole-job termination behavior. Do not assume a rank-local exception ends the other ranks, and do not introduce a new collective into an exception path after peers may have exited. Exercise the failure path in a bounded multi-worker test when changing it.

## Tiny-problem and reproducibility checks

For supervised learning, use a fixed small dataset and controlled stochasticity. Temporarily remove augmentation and regularization only in the isolated diagnostic configuration, then restore the intended experiment settings. Seek the known attainable loss or clear fitting behavior. Investigate failure before scaling; do not declare every failure proof of a bug when capacity, objective, or optimization can explain it.

For diffusion or flow matching, a fixed-noise diagnostic tests a narrower path than training over resampled noise. Keep that limitation visible. For RL, use known reward/termination cases and a short bounded learning task or rollout; do not use a universal loss-to-zero requirement. The [policy-gradient explanation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) distinguishes fixed-batch optimization loss from expected return.

Pin the code, data order, worker seeding, software, device, and relevant deterministic settings for a repeatability check. Seed every RNG actually used, including independent NumPy generators and the simulator. Compare the first divergence in inputs, outputs, loss, or updates before demanding matching long learning curves. PyTorch documents [limits to reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html) across releases, platforms, and CPU/GPU execution. Use numerical tolerances or repeated-seed statistics appropriate to the comparison.

When timing frameworks, keep diagnostic settings explicit. Host reads of accelerator tensors and exhaustive checks can change throughput; use equivalent validation modes and document any limitations in correctness coverage. Do not treat matching seeds across different backends as evidence of identical trajectories.

## Evaluation accounting

Establish the episode set, seeds, metric, and recovery/exclusion rules before evaluating.

Suppose 100 episodes are planned, 80 complete, and 20 crash. If 60 completed episodes succeed, `60 / 80 = 75%` describes only the completed subset. It is not a complete benchmark result.

- If the 20 events are policy failures under the protocol, include them as failures: `60 / 100 = 60%`.
- If they are infrastructure failures, retain their IDs and mark the evaluation incomplete. Apply only the predefined retry policy to the same cases. Do not replace them with fresh seeds or easier episodes.
- If a protocol permits exclusions, report the reasons, excluded counts, and denominator explicitly. A low exclusion rate alone does not establish absence of bias.
- Record attempts separately from final episode outcomes so retries do not double-count success. Keep zero completed episodes undefined or failed, never a fabricated zero or perfect score.

This accounting policy follows the definition of the chosen metric. Use the same policy for every compared method. The official [MLQA evaluator](https://github.com/facebookresearch/MLQA/blob/main/mlqa_evaluation_v1.py) provides one concrete example: missing predictions remain in its denominator and contribute zero. Do not generalize that penalty to arbitrary metrics; a zero loss would reward a failed loss computation.

## Recovery and checkpointing

Keep compute exceptions outside transient-I/O retry scopes. Retry only an operation whose identity and side effects are understood. A timeout after an acknowledged or partially completed write may need an idempotency check before another attempt.

For local checkpoint files, write a temporary file in the destination filesystem, finish and validate the write as appropriate, then publish it through the storage backend's supported atomic operation. Retain the prior valid checkpoint on failure. Python documents [`os.replace`](https://docs.python.org/3/library/os.html#os.replace) as atomic when successful and warns that moves across filesystems can fail; atomic publication is not itself a guarantee of power-loss durability or remote-object-store semantics.

For a full training resume, restore the model, optimizer, scheduler, scaler when used, RNGs, step counters, and relevant sampler/environment state supported by the pipeline. If exact continuation is unsupported, label the limitation; weight initialization is a new run, not a full resume. Run smoke checks in separate state so testing does not advance or overwrite the real checkpoint.

Use the existing scheduler/runner for retries with finite attempt and time budgets. Preserve the failing command, traceback, exit status, and checkpoint lineage. Never report a failed or partially saved run as complete.

## Sources

- Source policy: [Best Practice on Error Handling in Research Code](https://app.notion.com/p/tk233/Best-Practice-on-Error-Handling-in-Research-Code-3c3f0df2b51280a695d1e10da7dce2f5). Access may require the author's Notion workspace; the skill is self-contained.
- Language/framework references are linked beside the relevant behavior. Verify the installed version before using version-sensitive APIs.
