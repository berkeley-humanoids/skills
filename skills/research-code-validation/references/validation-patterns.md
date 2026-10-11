# Validation patterns

Use relevant sections only.

## Contents

- [Boundary checks](#boundary-checks)
- [Loss and gradient diagnostics](#loss-and-gradient-diagnostics)
- [Tiny-problem and reproducibility checks](#tiny-problem-and-reproducibility-checks)
- [Evaluation accounting](#evaluation-accounting)
- [Recovery and checkpointing](#recovery-and-checkpointing)
- [Sources](#sources)

## Boundary checks

Validate shapes and semantics:

| Boundary | Example checks |
| --- | --- |
| Dataset to model | Nonempty splits. No unintended train/eval overlap. Matching sample IDs/timestamps. Time-aligned images/actions. Channel order, normalization, mask meaning, label domain. |
| Policy to simulator/controller | Joint order, action meaning/units, coordinate frame, control rate/decimation, finite state, termination/truncation handling. |
| Loss to optimizer | Scalar, finite objective. Nonempty valid denominator. Intended optimizer parameter groups. |
| Checkpoint to runner | Compatible model/config/data identity. Requested resume state. |
| Runner to metrics | Planned/completed/failed counts. Aggregation weights. Visible worker errors. Matching comparison config. |

Reject runtime contract violations with exceptions:

```python
if prediction.shape != target.shape:
    raise ValueError(
        f"prediction shape {prediction.shape} != target shape {target.shape}"
    )
```

Use this example only for objectives with matching shapes. Classification logits and class-index labels normally have different shapes. Avoid Python's [`assert` statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement) for runtime contracts. Optimization removes it.

## Loss and gradient diagnostics

### Initial loss

Derive baselines from the objective, target distribution, initialization, weighting, smoothing, masking, ignored entries, reduction, and regularization. Reject empty reductions unless the algorithm defines them.

| Objective | Diagnostic |
| --- | --- |
| Cross-entropy | Unweighted loss with normalized targets and uniform predictions over `C` classes: `log(C)` per example. Controlled baseline only. Random logits can differ. |
| Regression, diffusion, flow matching | Target scaling and the prediction target. Fixed noise/timesteps if resampling obscures diagnostics. |
| RL | Component rewards, bootstrapping, masks, returns, and a known short transition sequence. No positive, monotonic, or zero-loss requirement for policy gradients. |

The [`CrossEntropyLoss` contract](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html) defines target shapes and reductions. Validate soft-target probability constraints yourself.

### Ordinary PyTorch update

Adapt this unscaled, dense-gradient diagnostic to the existing training step. It is not a trainer or AMP/distributed implementation.

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

Frozen or conditional branches can lack gradients per batch. For sparse gradients, examine the appropriate value representation.

### Mixed precision and distributed execution

Follow the installed framework's [AMP procedure](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html):

1. Scale the loss.
2. Backpropagate.
3. Unscale before gradient examination or clipping.

For accumulation, unscale once per optimizer update after all microbatches. Do not mix scaled and unscaled gradients.

`GradScaler.step()` can skip updates with nonfinite gradients as the scale adapts. Record overflow, skipped updates, and actual optimizer updates. Do not advance update-based schedules for skipped updates. Define a diagnostic policy for persistent overflow or no progress. Investigate nonfinite forward losses. Do not sanitize losses to force updates.

Rank-local exceptions might leave peers active. Do not add exception-path collectives after peers might exit. After failure-path changes, run a bounded multi-worker test.

## Tiny-problem and reproducibility checks

For supervised diagnostics, control stochasticity on fixed small datasets. Disable augmentation/regularization only within diagnostics. Restore experiment settings afterward. Seek known attainable loss or clear fitting. Before scaling, investigate failures, including capacity, objective, and optimization limits.

Fixed-noise diagnostics cover less than resampled diffusion or flow matching. State this limitation. For RL, use known reward/termination cases and short, bounded learning tasks or rollouts. The [policy-gradient explanation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) separates fixed-batch loss and expected return.

Pin code, data order, worker seeding, software, device, and deterministic settings. Seed every RNG, including independent NumPy generators and simulators. Compare the first divergence in inputs, outputs, loss, or updates before comparing long learning curves. PyTorch's [limits to reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html) cover releases, platforms, and CPU/GPU execution. Matching seeds across backends do not establish identical trajectories.

For framework timing, state diagnostic settings and correctness limits. Accelerator-tensor host reads and exhaustive checks can affect throughput.

## Evaluation accounting

Predefine episodes, seeds, metrics, and recovery/exclusion rules.

Example: 100 planned episodes, 80 complete, 20 crash, 60 succeed. `60 / 80 = 75%` describes only completed episodes.

- For policy failures, count crashes as failures: `60 / 100 = 60%`.
- For infrastructure failures, retain IDs. Mark evaluation incomplete. Retry the same cases under predefined rules. Do not substitute fresh seeds or easier episodes.
- For permitted exclusions, report reasons, counts, and denominators. Low exclusion rates do not establish absence of bias.
- Record attempts separately from final outcomes to avoid duplicate successes. Zero completed episodes means undefined or failed results, never zero or perfect scores.

Follow the metric's definition consistently across methods. The [MLQA evaluator](https://github.com/facebookresearch/MLQA/blob/main/mlqa_evaluation_v1.py) retains missing predictions in its denominator with zero contributions. Do not generalize this penalty: zero loss rewards failed loss computation.

## Recovery and checkpointing

Keep compute exceptions outside transient-I/O retries. Retry only operations with known identity and side effects. After acknowledged or partial writes time out, assess idempotency before retries.

For local checkpoints:

1. Write a temporary file in the destination filesystem.
2. Finish the write.
3. Validate the write as appropriate.
4. Publish through the backend's supported atomic operation.

On failure, retain the previous valid checkpoint. Successful [`os.replace`](https://docs.python.org/3/library/os.html#os.replace) is atomic, but cross-filesystem moves can fail. Atomic publication guarantees neither power-loss durability nor remote-object-store semantics.

For full resumes, restore model, optimizer, scheduler, scaler when used, RNGs, step counters, and supported sampler/environment state. State any limits to exact continuation. Weight initialization starts a new run.

Use the existing scheduler/runner with finite retry attempt/time budgets.

## Sources

- Source policy: [Best Practice on Error Handling in Research Code](https://app.notion.com/p/tk233/Best-Practice-on-Error-Handling-in-Research-Code-3c3f0df2b51280a695d1e10da7dce2f5). Access may depend on the author's Notion workspace. This skill is self-contained.
- References appear beside behavior. Validate installed versions before version-sensitive API use.
