# Agent Skill Guide

The root `SKILL.md` is designed for coding agents such as Codex, Claude Code, and Copilot-style terminal agents. It tells the agent when to use the package, how to install it, how to pick models, and how to report fitted model text.

## How To Reference The Skill

Point the agent at the repository root:

```markdown
Use the agentic-imodels skill from https://github.com/csinva/agentic-imodels.
```

In a local checkout, reference the root skill file:

```markdown
Read and follow ./SKILL.md for interpretable tabular regression tasks.
```

The root is intentional: it is both the package install target and the skill entrypoint.

## What Agents Should Do

When the user has a tabular regression question and wants interpretability, the agent should:

1. Install or import `agentic_imodels`.
2. Identify the target column, candidate predictors, and controls.
3. Fit at least one honest model and one high-rank decoupled model when the user wants a robust interpretation.
4. Print the fitted models.
5. Use the printed text to describe feature direction, magnitude, thresholds, and robustness.
6. Explicitly disclose when a model is display-predict decoupled.

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
