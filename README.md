# Arabic LLM Lab

**A 12-week AI research and engineering project focused on building an Arabic language model from the ground up.**

Arabic LLM Lab is a hands-on project designed to explore the engineering behind modern language models, from tokenization and Transformer architectures to training, post-training, evaluation, and deployment.

Rather than only using existing LLM APIs, the goal is to understand and implement the core components of a language-model pipeline, with particular attention to Arabic text.

> **Project status:** Week 1 — Repository foundation and tokenizer development. Features listed in the roadmap are planned deliverables and should not be considered implemented until completed and tested.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [12-Week Roadmap](#12-week-roadmap)
- [Week 1: Repository Foundation and Tokenization](#week-1-repository-foundation-and-tokenization)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Development Environment](#development-environment)
- [Code Quality and Testing](#code-quality-and-testing)
- [Development Workflow](#development-workflow)
- [Evaluation and Reporting](#evaluation-and-reporting)
- [Project Principles](#project-principles)
- [License](#license)

## Project Overview

Large Language Models (LLMs) rely on multiple interconnected components. Their behavior depends not only on the neural network architecture but also on text tokenization, data preparation, training procedures, post-training, evaluation, and deployment.

Arabic LLM Lab aims to implement and study these components incrementally within a single, organized Python repository.

The project follows a staged development approach:

1. Establish a reproducible software engineering environment.
2. Build a tokenizer and evaluate its behavior on Arabic and English text.
3. Implement and train a small decoder-only Transformer.
4. Develop a post-training pipeline.
5. Evaluate model performance systematically.
6. Serve the model through an application interface.
7. Explore vision-language models (VLMs).
8. Develop a Model Context Protocol (MCP) server.

The repository is intended to evolve throughout the 12-week program, with each stage building on the previous work.

## Objectives

### Technical objectives

- Understand how text is transformed into tokens and token IDs.
- Implement a byte-level Byte Pair Encoding (BPE) tokenizer.
- Investigate tokenization behavior for Arabic and English.
- Understand the architecture and training process of decoder-only Transformers.
- Explore the stages of language-model training and post-training.
- Develop reproducible evaluation procedures.
- Build a serving interface for model inference.
- Explore multimodal AI and tool integration through VLMs and MCP.

### Software engineering objectives

- Maintain a consistent project structure.
- Manage dependencies and the Python environment reproducibly.
- Apply automated formatting and linting.
- Use static type checking.
- Write automated tests.
- Introduce continuous integration (CI).
- Follow version control and pull-request workflows.
- Document experiments, results, and engineering decisions.

## 12-Week Roadmap

The following roadmap summarizes the intended progression. The exact scope of each week should follow the internship program's prescribed tasks.

| Stage | Main focus | Intended outcome |
|---|---|---|
| Week 1 | Repository foundation and tokenization | Development infrastructure and a tested tokenizer |
| Weeks 2–3 | Model architecture and training foundations | A small decoder-only Transformer and initial training workflow |
| Weeks 4–5 | Data and model training | A more complete, reproducible training pipeline |
| Weeks 6–7 | Post-training | A pipeline for adapting the pretrained model |
| Weeks 8–9 | Evaluation | Evaluation procedures, metrics, and documented results |
| Week 10 | Serving | A model inference interface |
| Week 11 | Vision-language models | An initial VLM experiment or integration |
| Week 12 | MCP and integration | An MCP server and a consolidated project deliverable |

*This is a high-level roadmap, not a claim that each component is already implemented. Detailed tasks and deliverables should be tracked against the official internship plan.*

## Week 1: Repository Foundation and Tokenization

Week 1 establishes the engineering practices needed to develop the remaining components reliably.

### 1. Development environment

The project uses `uv` to manage Python dependencies, the virtual environment, and reproducible dependency installation.

### 2. Code quality

The initial quality-control setup includes:

- **Ruff:** Python linting and code formatting.
- **mypy:** Static type checking.
- **pytest:** Automated testing.
- **pre-commit:** Running configured checks before Git commits.

### 3. Continuous integration

GitHub Actions is intended to automate code-quality checks and tests when changes are pushed or pull requests are opened.

### 4. Byte-level BPE tokenizer

The first major AI component is a byte-level Byte Pair Encoding tokenizer.

Its planned functionality includes:

- `train`: Learn merge rules from a training corpus.
- `encode`: Convert text into token IDs.
- `decode`: Reconstruct text from token IDs.

Byte-level tokenization begins with the UTF-8 byte representation of text and learns merges between frequently occurring byte sequences.

This provides a foundation for studying how tokenizers handle Arabic characters, English text, punctuation, and mixed-language input.

### 5. Arabic and English token-efficiency comparison

The tokenizer will be evaluated against two production tokenizers.

The comparison will examine:

- Tokens per word.
- Token counts for equivalent Arabic and English text samples.
- Differences in tokenization behavior.
- The implications of token counts for context-window usage.

The results will be documented in a report rather than relying on assumptions about which tokenizer performs best.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| `uv` | Dependency and environment management |
| Ruff | Linting and formatting |
| mypy | Static type checking |
| pytest | Automated testing |
| pytest-cov | Test coverage reporting |
| pre-commit | Git commit hooks |
| Git and GitHub | Version control and collaboration |
| GitHub Actions | Continuous integration |
| PyTorch | Planned model implementation and training |
| Jupyter notebooks | Planned experiments and analysis |

Some technologies are part of the initial setup, while others will be introduced when their corresponding project stages begin.

## Repository Structure

The repository is designed to keep implementation code, configuration, tests, documentation, and experimental results separate.

The following is the **target structure** as the project develops. Directories and modules should be added as their corresponding stages begin.

```text
arabic-llm-lab/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
│       └── ci.yml
├── configs/
├── docs/
├── infra/
│   ├── docker/
│   ├── k8s/
│   └── monitoring/
├── notebooks/
│   └── kaggle/
├── reports/
├── src/
│   └── lab/
│       ├── __init__.py
│       ├── cli.py
│       ├── tokenizer/
│       ├── model/
│       ├── train/
│       ├── posttrain/
│       ├── data/
│       ├── eval/
│       ├── serve/
│       ├── vlm/
│       └── mcp_server/
├── tests/
│   └── test_cli.py
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CODEOWNERS
├── Makefile
├── README.md
├── pyproject.toml
└── uv.lock
```

### Directory responsibilities

| Path | Responsibility |
|---|---|
| `src/lab/cli.py` | Command-line entry point |
| `src/lab/tokenizer/` | Tokenizer training, encoding, and decoding |
| `src/lab/model/` | Model architecture and related components |
| `src/lab/train/` | Model-training procedures |
| `src/lab/posttrain/` | Post-training workflows |
| `src/lab/data/` | Data loading and preparation |
| `src/lab/eval/` | Evaluation procedures and metrics |
| `src/lab/serve/` | Model inference and serving |
| `src/lab/vlm/` | Vision-language model experiments |
| `src/lab/mcp_server/` | MCP server implementation |
| `configs/` | Configuration files |
| `tests/` | Automated tests |
| `reports/` | Experiment results and evaluation reports |
| `docs/` | Technical documentation |
| `notebooks/kaggle/` | Notebook-based experiments |
| `infra/` | Infrastructure and deployment configurations |
| `.github/workflows/` | GitHub Actions automation |
| `pyproject.toml` | Project metadata, dependencies, and tool configuration |
| `uv.lock` | Locked dependency resolution |
| `.pre-commit-config.yaml` | Pre-commit hook configuration |
| `CHANGELOG.md` | Record of notable changes |
| `CODEOWNERS` | Code ownership rules for supported GitHub workflows |
| `Makefile` | Shortcuts for common development commands |

## Development Environment

### Prerequisites

- Python 3.13 or a compatible version allowed by `pyproject.toml`.
- Git.
- `uv`.
- A GitHub account for repository hosting and CI.

### Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/OmarZabadani/arabic-llm-lab.git
cd arabic-llm-lab
```

Synchronize the project environment:

```bash
uv sync
```

This installs the project and its declared dependencies into the managed environment.

Run the command-line entry point:

```bash
uv run lab
```

The initial CLI prints:

```text
Arabic LLM Lab
```

This confirms that the installed command-line entry point is accessible. It does not, by itself, confirm that the tokenizer or model components have been implemented.

## Code Quality and Testing

Run these commands from the repository root.

### Format Python files

```bash
uv run ruff format .
```

Formats code according to the project's Ruff configuration.

### Check code quality

```bash
uv run ruff check .
```

Detects configured code-quality issues, including common errors, import problems, and selected style violations.

### Verify formatting

```bash
uv run ruff format --check .
```

Checks whether files are formatted correctly without modifying them.

### Run static type checking

```bash
uv run mypy src
```

Checks source code for type inconsistencies that mypy can detect.

### Run tests

```bash
uv run pytest
```

Runs the automated test suite and reports the results.

The project also configures pytest-cov to report test coverage. Coverage indicates which code was exercised by tests; it does not prove that the code is correct.

### Run pre-commit checks

Install the Git hook:

```bash
uv run pre-commit install
```

Run the configured hooks across tracked files:

```bash
uv run pre-commit run --all-files
```

Pre-commit runs the configured checks before commits. It complements, rather than replaces, automated testing and CI.

## Development Workflow

The intended workflow is:

1. Create or update a feature on a development branch.
2. Implement the change in the appropriate module.
3. Add or update automated tests.
4. Run formatting, linting, type checking, and tests.
5. Commit the change after the configured hooks pass.
6. Push the branch to GitHub.
7. Open a pull request and review the changes.
8. Confirm that CI passes before merging.
9. Document meaningful changes and experimental results.

Branch protection, pull-request templates, issue templates, ownership rules, and project-management integration are repository setup tasks to introduce according to the internship plan.

## Evaluation and Reporting

The project aims to make its AI experiments reproducible and measurable.

For the initial tokenizer stage, reports should identify:

- The datasets or text samples used.
- The languages represented.
- The tokenizer configurations.
- The evaluation methodology.
- The measured token counts and derived metrics.
- Relevant limitations and observations.

Later model stages should similarly document their datasets, training configurations, evaluation methods, and results.

No performance numbers are claimed in this README because they must come from actual experiments.

## Project Principles

### Reproducibility

Use explicit configurations, a dependency lockfile, and documented commands so that experiments can be repeated.

### Testability

Add automated tests for important functionality, including tokenization, model components, data processing, and serving interfaces as they are implemented.

### Measurable progress

Track deliverables through working implementations, passing tests, and documented experimental results rather than relying solely on completed tasks.

### Incremental development

Build the repository in stages. Avoid implementing later components before the required foundations are in place.

### Transparency

Clearly distinguish implemented functionality from planned work, and report experimental results without inventing performance claims.

## License

No license has been specified yet. Add a `LICENSE` file once the project's licensing terms have been decided.

---

**Arabic LLM Lab** is intended to serve as a practical learning and engineering project: starting with reliable Python tooling and tokenization, then progressing toward a more complete language-model development pipeline.
