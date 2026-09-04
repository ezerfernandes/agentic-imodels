# Agent Skill Guide

The root `SKILL.md` is designed for coding agents such as Codex, Claude Code, and Copilot-style terminal agents. It tells the agent when to use the package, how to install it, how to pick models, and how to report fitted model text.

## How To Reference The Skill

Point the agent at the repository root:

```markdown
Use the agentic-imodels skill from https://github.com/ezerfernandes/agentic-imodels.
```

In a local checkout, reference the root skill file:

```markdown
Read and follow ./SKILL.md for interpretable tabular regression tasks.
```

The root is intentional: it is both the package install target and the skill entrypoint.

## Install the skill

Install the skill in the same environment as the Python package so an agent can both load the instructions and import `agentic_imodels`.

### Claude Code

For a user-scope install, copy the skill into `~/.claude/skills/agentic-imodels/`:

```bash
mkdir -p ~/.claude/skills/agentic-imodels
curl -fsSL https://raw.githubusercontent.com/ezerfernandes/agentic-imodels/main/SKILL.md \
  -o ~/.claude/skills/agentic-imodels/SKILL.md
```

For a project-scope install, run the equivalent commands from the project root:

```bash
mkdir -p .claude/skills/agentic-imodels
curl -fsSL https://raw.githubusercontent.com/ezerfernandes/agentic-imodels/main/SKILL.md \
  -o .claude/skills/agentic-imodels/SKILL.md
```

The project-scope copy can be checked into the repository when the whole team should use the skill.

### Codex CLI

The current [Codex skill documentation](https://developers.openai.com/codex/skills/) lists `~/.agents/skills` as the user-scope location and `.agents/skills` in a repository as the project-scope location. Create a directory named `agentic-imodels` under the chosen location and copy this file to its `SKILL.md`:

```bash
# User scope
mkdir -p ~/.agents/skills/agentic-imodels
cp SKILL.md ~/.agents/skills/agentic-imodels/SKILL.md

# Project scope (run from the repository root)
mkdir -p .agents/skills/agentic-imodels
cp SKILL.md .agents/skills/agentic-imodels/SKILL.md
```

Codex detects skill changes automatically; restart it if a newly installed skill does not appear. For a local checkout, `scripts/install_skill.sh` performs the Claude Code user-scope copy. The package must also be installed in the environment used by the agent:

```bash
uv add git+https://github.com/ezerfernandes/agentic-imodels
```

## What Agents Should Do

When the user has a tabular regression question and wants interpretability, the agent should:

1. Keep `X` as a DataFrame so printed models use real column names.
2. Install or import `agentic_imodels`.
3. Identify the target column, candidate predictors, and controls.
4. Fit at least one honest model and one high-rank decoupled model when the user wants a robust interpretation.
5. Print the fitted models.
6. Use the printed text to describe feature direction, magnitude, thresholds, and robustness.
7. Explicitly disclose when a model is display-predict decoupled.

## Recommended Agent Defaults

| Situation | Model |
| --- | --- |
| User simply asks for an interpretable regressor | `HingeEBMRegressor` |
| User needs the printed equation to be the model | `HingeGAMRegressor` or `SmartAdditiveRegressor` |
| User wants maximum LLM-readable output | `TeacherStudentRuleSplineRegressor` |
| User wants a very small tree | `TinyDTDepth2Regressor` |

## Reporting Pattern

Agents should return:

- the model class used
- whether it is honest or display-predict decoupled
- the metric used for evaluation, if any
- the fitted model text, unless it is too long for the current context
- a short interpretation grounded in that text

Example:

```text
I fit HingeGAMRegressor, an honest model, so the displayed hinge terms are the terms used for prediction. The strongest positive effect is feature_2 above 1.4; feature_0 is near zero after selection.
```

## What Agents Should Avoid

- Do not call a decoupled model's display the complete model.
- Do not paraphrase the fitted text when the user asked to inspect the model.
- Do not use these regressors for classification without explaining that they are regression models.
- Do not rely on categorical or missing-value handling inside the estimator; preprocess upstream.
