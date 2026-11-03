---
name: research-code-validation
description: Validate ML/RL research code before expensive runs. Use when building, changing, reviewing, or debugging training frameworks, data pipelines, simulation environments, losses, checkpoints, or evaluation. Use before training/evaluation launch, resume, or scale-up, and for NaNs, silent fallbacks, or suspect metrics. Exclude unrelated application builds, formatting-only edits, and general ML explanations.
---

# Research Code Validation

Stop invalid experiments early. Recover only through a defined procedure that preserves the experiment.

## Scope the work

- Read project instructions, the entry point, resolved configuration, and relevant tests. Identify the objective, data, backend, and run budget.
- Limit edits and checks to concrete failure modes. Reuse existing validators and runners. Avoid unrelated refactors, speculative abstractions, and full experiments for code review.
- Reuse passing checks when code, configuration, data, and environment remain unchanged. Repeat checks that a change invalidates.
- Keep smoke-test outputs and state separate. Start the experiment from its intended initialization or checkpoint.

## Preserve failure information

- Validate inputs at actual boundaries. Avoid duplicate checks for guaranteed internal states.
- Let unexpected exceptions retain their traceback and nonzero job status. Never replace failures with defaults, zero losses, skipped samples, or silent backend changes.
- Raise explicit exceptions for required checks. Python can remove `assert` under `-O`.
- Catch specific exceptions only for defined recovery, useful context, or cleanup. Preserve the original cause. Avoid redundant wrappers and logs.
- Record the step, sample/episode ID, rank, expected contract, and observed value. Make fatal worker errors stop the whole job.
- Treat clipping, masking, `nan_to_num`, reset-on-error, and filtering as explicit algorithm or protocol choices.

## Validate before scaling

Run the cheapest relevant checks first:

| Stage | Evidence to obtain |
| --- | --- |
| Entry checks | Validate configuration, data/assets, splits, outputs, checkpoint compatibility, device/backend/precision, and hyperparameters. Reject ignored options. Resolve imports and builds. |
| One batch or rollout | Run actual preprocessing and inference or simulation. Validate shapes, dtypes, devices, finite values, masks, ranges, units, frames, and timing. Visualize ambiguous data. |
| Objective and update | Validate the initial-loss baseline, gradient flow, and optimizer update. For RL, validate rewards, actions, resets, termination, and truncation. |
| Tiny problem | For supervised learning, fit a fixed batch with controlled randomness and regularization. For other objectives, use a controlled task, rollout, or noise sample. Set an objective-specific target. Zero is not universal. |
| Short end-to-end run | Exercise the real runner, logging, evaluation accounting, and checkpoint save/load. Test resume and workers when relevant. |
| Scale-up | Run authorized experiments after applicable checks pass. Complete cheaper checks before expensive closed-loop or hardware evaluation. Smoke tests do not prove convergence or correctness. |

Read [validation patterns](references/validation-patterns.md) for contracts, numerical checks, reproducibility, and recovery details.

## Keep runtime checks meaningful

- Keep required boundary and finite-loss checks active. Validate ordinary gradients before updates. For AMP, inspect unscaled gradients and preserve recorded overflow recovery.
- Validate intended parameter updates in diagnostics. Allow frozen or conditionally unused parameters. Validate checkpoint state before resume.
- Limit exhaustive checks and expensive diagnostics to bounded runs. Measure their cost. Use and record comparable validation modes in benchmarks.
- Repeat a bounded case with controlled RNGs in a pinned setup. Use justified tolerances or repeated-seed statistics. Matching seeds do not guarantee identical curves across systems.

## Recover only under a defined contract

| Event | Required behavior |
| --- | --- |
| Invalid configuration/data, shape error, nonfinite forward loss, or unexpected simulator failure | Stop and expose the cause. Fix it or document a versioned data/protocol change before retrying. |
| Transient I/O failure | Retry only defined exceptions/statuses with timeouts, bounded backoff, and logs. Preserve operation identity. Prevent duplicate effects. Fail after exhaustion. |
| Checkpoint write failure | Retry or save to an explicit, reported fallback destination. Preserve the last valid checkpoint. Never mark an incomplete save as resumable or successful. |
| Evaluation failure | Retain planned IDs and outcomes. Count task failures per protocol. Mark unresolved infrastructure failures incomplete. Never present surviving cases as the full evaluation. |
| OOM, backend fallback, or reduced workload | Record failed and replacement configurations. Treat batch size, precision, backend, environment count, or data changes as explicit experiment changes. |
| Normal episode termination or documented AMP overflow | Follow the algorithm/framework behavior and record counts. |

Keep restart/retry budgets in the runner or scheduler. Do not repeat deterministic failures indefinitely or resume from suspect checkpoints. Preserve `KeyboardInterrupt` and `SystemExit`.

## Finish with evidence

Report commands, configuration, completed checks, results, unresolved failures, and skipped checks with reasons.

Record revision/local changes, resolved configuration, data/split identity, seeds/RNGs, software/hardware/backend, precision, checkpoint lineage, and validation mode in existing run metadata.

If a check fails, fix its scoped cause and repeat it. Continue the authorized run after checks pass. State missing evidence or resources. For active jobs, report running status and log/output location. Do not claim completion or validation without evidence.
