## Summary

<!-- What does this PR change, in 1–3 sentences? -->

## Motivation

<!-- Why is this change needed? Link the related issue or internship task if one exists. -->

Closes #

## Stage

<!-- Which part of the roadmap does this belong to? e.g. Week 1 – Tokenizer -->

## Changes

<!-- Key changes, grouped by area (e.g. tokenizer, CLI, tests, CI, docs). -->

-

## How it was tested

<!-- Commands run and results. Include any new test cases added. -->

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

- [ ] All of the above pass locally
- [ ] New or updated tests cover the change

## Results / outputs

<!-- For tokenizer, evaluation, or training changes: attach token counts, metrics, or
     report excerpts. Delete this section if not applicable. -->

## Checklist

- [ ] Code is formatted with Ruff and passes lint and type checks
- [ ] Tests were added or updated where behavior changed
- [ ] Pre-commit hooks ran (`uv run pre-commit run --all-files`)
- [ ] Docs or `README.md` were updated if behavior or setup changed
- [ ] Reports in `reports/` were updated if results changed
- [ ] No secrets, credentials, or large data files were committed
- [ ] README claims match what is actually implemented

## Notes for reviewers

<!-- Anything you want reviewers to focus on, open questions, or trade-offs. -->