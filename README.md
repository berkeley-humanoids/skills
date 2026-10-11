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

[research-code-validation](skills/research-code-validation/SKILL.md) defines checks for ML/RL pipelines and experiment launches, resumes, and larger runs.

Invoke it explicitly in Claude Code:

```text
/berkeley-humanoids-skills:research-code-validation Check this training pipeline, run the relevant smoke checks, and then launch the requested experiment.
```

Agents can select the skill automatically. Add this rule to the project's `CLAUDE.md` or `AGENTS.md`:

```text
Use the research-code-validation skill when changing research training/evaluation pipelines and before launching or resuming their experiments. Run the cheapest relevant checks first, reuse still-valid results, and report unresolved failures before scaling up.
```

Skill selection does not intercept every command. Put mandatory checks in the training entry point or runner, including manual and scheduled launches. See [skill invocation](https://code.claude.com/docs/en/skills#control-who-invokes-a-skill) and [hooks](https://code.claude.com/docs/en/hooks-guide).

## License

CC0 1.0 Universal. See [LICENSE](LICENSE).
