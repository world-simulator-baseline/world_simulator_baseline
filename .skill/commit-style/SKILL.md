---
name: commit-style
description: Apply the user's general code commit message requirements. Use when Codex is asked to commit code, write a commit message, review a proposed commit title, or summarize staged changes into the required single-line format.
---
# Commit Style

Format: `TYPE(SCOPE): SUBJECT`

## Types
- `feat` - new feature, user-facing enhancement
- `fix` - bug fix, error correction
- `docs` - documentation, comments, README
- `style` - formatting, whitespace, semicolons
- `refactor` - code restructure, no behavior change
- `perf` - performance optimization
- `test` - tests, test data, test scripts
- `chore` - build, deps, tooling, configs
- `revert` - rollback previous commit

## Scopes
- `core` - patches, base classes
- `models` - model definitions
- `data` - data loading, operators
- `train` - training loop
- `infer` - inference
- `api` - public interface
- `cli` - command-line tools
- `utils` - helpers
- `deps` - dependencies
- `build` - packaging
- `ci` - workflows
- `test` - test framework
- `docs` - documentation files
- `repo` - gitignore, license

## Rules
- Max 50 chars, lowercase, no period, imperative mood
- Single line only, no body
- No Co-Authored-By
- Use subject templates when they fit the change; refer to Examples for tone and granularity

## Subject Templates
- `docs(docs): add <topic> notes` - new explanatory notes or study notes
- `docs(docs): update <topic> docs` - changes to existing documentation
- `fix(<scope>): handle <case>` - bug fix for an edge case or invalid input
- `refactor(<scope>): simplify <component>` - simplify structure without behavior change
- `test(<scope>): add <case> coverage` - add focused test coverage
- `chore(<scope>): update <tooling>` - update config, tooling, or maintenance files

## Examples
- `feat(models): add wan video action encoder with noise injection`
- `fix(data): handle None action_emb in batch collation`
- `refactor(core): simplify patch application logic`
- `perf(ops): vectorize action normalization`
- `test(models): add action encoder coverage`
- `docs(docs): add public export notes`
- `chore(build): update packaging config`
