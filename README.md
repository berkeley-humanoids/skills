# Skills

## Install

1. Add the marketplace.

   ```
   /plugin marketplace add Berkeley-Humanoids/Skills
   ```

2. Install plugins.

   ```
   /plugin install berkeley-humanoids-skills@berkeley-humanoids-skills
   ```

## Upgrade an installed version

The plugin tracks this repository, so the marketplace refresh also moves the installed plugin to the current commit:

1. Update the marketplace.

   ```
   /plugin marketplace update berkeley-humanoids-skills
   ```

2. Restart Claude Code. The new version applies after the restart.

`claude plugin list` prints the installed version of each plugin as a commit hash.

## Research code validation

[research-code-validation](skills/research-code-validation/SKILL.md) applies when building or reviewing ML/RL research pipelines and before launching, resuming, or scaling training and evaluation. It covers boundary checks, staged validation, visible failures, and recovery that preserves experimental validity.

Invoke it explicitly in Claude Code:

```text
/berkeley-humanoids-skills:research-code-validation Check this training pipeline, run the relevant smoke checks, and then launch the requested experiment.
```

The skill description supports automatic selection for relevant tasks. To make the expectation explicit in a consuming project's `CLAUDE.md` or `AGENTS.md`, add:

```text
Use the research-code-validation skill when changing research training/evaluation pipelines and before launching or resuming their experiments. Run the cheapest relevant checks first, reuse still-valid results, and report unresolved failures before scaling up.
```

Skill selection and project instructions guide the agent; they do not intercept every shell command. Put mandatory checks in the project's training entry point or runner so they also apply to manual and scheduled launches. See the official [skill invocation](https://code.claude.com/docs/en/skills#control-who-invokes-a-skill) and [hooks](https://code.claude.com/docs/en/hooks-guide) documentation for agent-side integration.

## License

CC0 1.0 Universal. See [LICENSE](LICENSE).
