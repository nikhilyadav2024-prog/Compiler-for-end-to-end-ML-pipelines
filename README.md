# Compiler for End-to-End ML Pipelines — Review 1

A small domain-specific language (DSL) for describing ML workflows
(load → preprocess → split → model → train → evaluate), compiled
using standard front-end and semantic-analysis compiler techniques.

**Review 1 scope (50% of the project):** DSL design, grammar, lexer,
parser/AST, and semantic analysis. Optimization and code generation
are planned for Review 2.

## Project structure

```
ml-pipeline-compiler/
├── src/
│   ├── lexer.py               # tokenizer
│   ├── parser.py               # recursive-descent parser -> AST
│   ├── ast_nodes.py            # AST node dataclasses + pretty printer
│   ├── semantic_analyzer.py    # semantic checks over the AST
│   └── main.py                  # demo driver (run this)
├── samples/
│   ├── valid_churn.pipeline
│   ├── invalid_missing_semicolon.pipeline
│   ├── edge_split_values.pipeline
│   ├── invalid_semantic_bad_split.pipeline
│   └── invalid_semantic_unknown_model.pipeline
├── tests/
│   ├── test_lexer.py
│   ├── test_parser.py
│   └── test_semantic.py
└── README.md
```

## Requirements

Python 3.8+. No third-party packages needed (standard library only).

## Running the demo

```bash
cd src
python3 main.py ../samples/valid_churn.pipeline
python3 main.py ../samples/invalid_missing_semicolon.pipeline
python3 main.py ../samples/edge_split_values.pipeline
python3 main.py ../samples/invalid_semantic_bad_split.pipeline
python3 main.py ../samples/invalid_semantic_unknown_model.pipeline
```

Each run prints the source, the token stream, the AST (or a syntax
error with line/column), and the result of semantic analysis (or a
semantic error with a line number).

## Running the tests

```bash
python3 -m unittest discover -s tests -v
```

32 tests: lexer (keywords, identifiers, strings, numbers, line/col
tracking, bad-character errors), parser (a full valid-program parse —
statement order, preprocess steps, model params, evaluate metrics —
plus missing semicolon, unmatched brace, unknown keyword, and two
grammar-level edge cases), and semantic analysis (required-stage
checks, split-percentage validation, model-type/parameter validation,
metric validation, and statement-ordering rules).

## Example program

```
pipeline churn_prediction {
    load data "customer.csv";

    preprocess {
        fill_missing age;
        scale income;
        encode city;
    }

    split train 80 test 20;

    model random_forest {
        trees = 100;
    }

    train;
    evaluate metrics [accuracy, f1];
}
```

## What semantic analysis checks

The parser only enforces *syntax* — a program like `split train 100
test 0;` is grammatically legal even though it is a useless split.
`semantic_analyzer.py` walks the accepted AST and additionally checks:

- **Required stages** — every pipeline needs exactly one `load`, one
  `split`, and one `train` statement.
- **Statement ordering** — `load` before `split`, `split` before
  `train`, `model` (if present) before `train`, `evaluate` (if
  present) after `train`.
- **Split validity** — `train` and `test` percentages must both be
  positive and add up to exactly 100.
- **Model validity** — the model type must be one of the supported
  types (`random_forest`, `logistic_regression`, `decision_tree`,
  `svm`, `neural_network`), and each parameter must be a supported,
  non-duplicated, positive value for that model type.
- **Metric validity** — `evaluate metrics [...]` must list at least
  one supported, non-duplicated metric.

Every semantic error message includes the source line it was raised
from.

## Roadmap

| Review | Scope |
|---|---|
| **1 (this repo) — 50%** | Problem/scope, DSL + grammar, lexer, parser/AST, semantic analysis, full test suite |
| 2 — 100% | Optimization, code generation, execution of the generated workflow, final testing |
