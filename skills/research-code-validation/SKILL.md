---
name: research-code-validation
description: Preserve experimental validity with early checks, visible failures, and staged validation. Use when building, modifying, reviewing, or debugging ML/RL training frameworks, data pipelines, simulation environments, losses, checkpointing, or evaluation code; before launching, resuming, or scaling training and evaluation runs; and when investigating NaNs, silent fallbacks, or misleading metrics. Do not apply to unrelated application builds, formatting-only edits, or general ML explanations.
---

# Research Code Validation

Make invalid experiments fail visibly before they consume substantial compute or produce trusted results. Detect broken assumptions close to their source. Recover only when the operation has a defined recovery contract that preserves the experiment.

## Scope the work

- Read the project's instructions, entry point, resolved configuration, and relevant tests. Identify the actual objective, backend, data, and requested run budget before choosing checks.
- For implementation or review, inspect the changed path and its boundaries. Add the smallest checks and targeted verification needed for concrete failure modes. Do not refactor unrelated code or launch a full experiment just to validate an edit.
- For a launch or resume, reuse relevant passing checks for the same code, configuration, data, and environment. Repeat checks invalidated by a change; do not repeat a whole validation ladder for every command.
- Keep smoke runs isolated from reported experiments: use separate outputs and state, then start the requested run from its intended initialization or checkpoint.
- Follow existing coding conventions. Reuse validators and the project's runner. Do not introduce a generic validation framework or new configuration surface without a current need.

## Preserve failure information

- Let unexpected exceptions propagate with their traceback and a nonzero job status. Do not convert them into default tensors, zero losses, fake successes, skipped batches, or silent CPU/backend fallbacks.
- Validate at boundaries where invalid state can enter: config parsing, data decoding, transformations, model/loss interfaces, simulator transitions, checkpoint restore, and metric aggregation. Do not repeat checks for states already guaranteed internally.
- Use explicit exceptions for required runtime checks. In Python, `assert` can disappear under `-O`; reserve it for tests and optional debugging assertions.
- Catch a specific exception only to perform a defined recovery, add useful context while preserving its cause, or clean up and re-raise. Avoid redundant catch-and-rethrow layers and repeated logging of the same traceback.
- Treat clipping, masking, `nan_to_num`, reset-on-error, and sample filtering as algorithm or protocol choices. Do not add them merely to make a broken run finish.
- Preserve the originating step, sample/episode ID, rank, expected contract, and observed value or summary. Keep cleanup from replacing the original error. Ensure a fatal worker error reaches the launcher instead of leaving peers running or blocked.

## Validate before scaling

Run the cheapest relevant checks first. Use the project's existing commands and suitable equivalents for its framework.

| Stage | Evidence to obtain |
| --- | --- |
| Entry checks | Reject invalid or ignored config options; validate data/assets, splits, checkpoint compatibility, output location, requested device/backend/precision, and meaningful hyperparameter constraints. Resolve imports and any required build before a long run. |
| One batch or rollout | Inspect real inputs and outputs through the actual preprocessing and model/environment path. Check shape, dtype, device, finite values, masks, ranges, units, frames, and temporal alignment where applicable. Visualize a representative sample when tensor checks cannot establish its meaning. |
| Objective and update | Check the initial objective against a justified baseline, finite loss, intended gradient flow, and an actual optimizer update. For RL, check reward components, actions, reset behavior, and termination versus truncation. |
| Tiny problem | For a suitable supervised objective, fit a fixed small batch with controlled randomness and regularization. For RL or stochastic objectives, use a small task, fixed rollout, or fixed-noise diagnostic with a stated expected result. Do not require every objective to reach zero. |
| Short end-to-end run | Exercise the real runner, logging, evaluation accounting, and checkpoint save/load. Exercise resume and multiple workers when those paths are part of the requested run or change. Record what was actually tested. |
| Scale-up | Start the authorized run only after applicable checks pass. Gate expensive closed-loop or hardware evaluation on relevant cheaper checks. A passing smoke test establishes basic validity, not convergence or scientific correctness. |

For objective baselines, PyTorch examples, reproducibility, and recovery details, read [validation patterns](references/validation-patterns.md) as needed.

## Keep runtime checks meaningful

- Keep required boundary and finite-loss checks active in full runs. Verify ordinary gradients before an update; for mixed precision, inspect unscaled gradients and follow the framework's recorded overflow/step-skipping protocol. Do not replace AMP's defined behavior with arbitrary batch skipping.
- Check that intended trainable parameters receive gradients and update in the diagnostic case; account for intentionally frozen or conditionally unused parameters. Validate checkpoint state before trusting a resume.
- Use exhaustive tensor inspection, anomaly detection, deterministic debug settings, and expensive visualizations for bounded diagnosis. Measure their cost before retaining them everywhere; per-tensor host synchronization can distort throughput.
- Record the validation mode used for performance measurements and keep it comparable across systems. Do not silently disable correctness checks to improve a benchmark.
- Reproduce a bounded case in a pinned software/hardware setup with controlled RNGs. Use justified numerical tolerances where exact equality is unsupported. Do not demand identical learning curves across hardware, framework versions, or nondeterministic simulators.

## Recover only under a defined contract

| Event | Required behavior |
| --- | --- |
| Invalid config, corrupt sample, shape error, nonfinite forward loss, or unexpected simulator failure | Stop and expose the cause. Fix the source or establish an explicit, versioned data/protocol change before rerunning. |
| Expected transient I/O failure | Retry only the specified exception/status, with a timeout, bounded backoff, and attempt logs. Preserve request/data identity and ensure retries cannot duplicate side effects. Exhaustion remains a failure. |
| Checkpoint write failure | Retry or use an explicit, reported fallback destination. Preserve the last valid checkpoint; never mark an incomplete write as resumable or silently report a successful save. |
| Evaluation failure | Retain the planned episode/sample and its outcome. Count policy/task failures under the metric's rules. Treat infrastructure failures as incomplete evaluation unless a predefined protocol handles them; never average only surviving episodes and present that as the full score. |
| OOM, backend fallback, or reduced workload | Report the failing configuration. Treat changed batch size, precision, backend, environment count, or data selection as an explicit new configuration; never silently substitute it in a comparison. |
| Normal episode termination or documented AMP overflow | Apply the algorithm/framework's intended behavior and record relevant counts. Distinguish these events from corrupted state or unexpected exceptions. |

Keep restart and retry budgets in the experiment runner or scheduler. Do not retry deterministic failures indefinitely or restart from a suspect checkpoint. Preserve interruption semantics and do not swallow `KeyboardInterrupt` or `SystemExit`.

## Finish with evidence

Report the checks performed, relevant results, unresolved failures, and the exact command/configuration used or proposed. Name skipped checks and their reason. Record the code revision and local changes, resolved config, seed/RNG setup, data and split identity, software/hardware/backend/precision, checkpoint lineage, and validation mode using the existing run metadata mechanism.

For a requested launch, continue with the authorized run when the relevant checks pass. If they fail, fix the scoped cause and rerun the affected check; do not bypass the failure. When evidence or resources are unavailable, state the limitation without claiming the run is validated. Mark a still-running job as running and give its log/output location, not a completed result.
