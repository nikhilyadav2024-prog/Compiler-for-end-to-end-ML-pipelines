# Compiler for End-to-End ML Pipelines — Review 1

A small domain-specific language (DSL) for describing ML workflows
(load → preprocess → split → model → train → evaluate), compiled
using standard front-end compiler techniques.

**Review 1 scope:** DSL design, grammar, lexer, parser/AST.
Semantic analysis, optimization and code generation are planned for
Reviews 2 and 3 (see `Compiler_ML_Pipeline_Review1.pptx`).

## Project structure

```
ml-pipeline-compiler/
├── src/
│   ├── lexer.py        # tokenizer
│   ├── parser.py        # recursive-descent parser -> AST
│   ├── ast_nodes.py     # AST node dataclasses + pretty printer
│   └── main.py           # demo driver (run this)
├── samples/
│   ├── valid_churn.pipeline
│   ├── invalid_missing_semicolon.pipeline
│   └── edge_split_values.pipeline
├── tests/
│   ├── test_lexer.py
│   └── test_parser.py
└── README.md
```

## Requirements

Python 3.8+. No third-party packages needed for Review 1 (standard
library only).

## Running the demo

```bash
cd src
python3 main.py ../samples/valid_churn.pipeline
python3 main.py ../samples/invalid_missing_semicolon.pipeline
python3 main.py ../samples/edge_split_values.pipeline
```

Each run prints the source, the token stream, and either the AST
(`Result: ACCEPTED`) or a syntax error with line/column.

## Running the tests

```bash
python3 -m unittest discover -s tests -v
```

17 tests, covering keyword/identifier/string/number tokenization,
line/column tracking, lexer errors, a full valid-program parse
(statement order, preprocess steps, model params, evaluate metrics),
and three invalid/edge cases (missing semicolon, unmatched brace,
unknown keyword, 100/0 split, empty preprocess block).

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

## Roadmap

| Review | Scope |
|---|---|
| **1 (this repo)** | Problem/scope, DSL + grammar, lexer, parser/AST, initial tests |
| 2 | Semantic analysis, IR, ML-specific validation, optimization |
| 3 | Code generation, execution of generated workflow, final testing |
