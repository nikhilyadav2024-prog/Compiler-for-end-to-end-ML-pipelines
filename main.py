"""
main.py
-------
Demo driver for the ML Pipeline compiler front end + semantic analysis.

Usage:
    python3 main.py ../samples/valid_churn.pipeline
    python3 main.py ../samples/invalid_missing_semicolon.pipeline
    python3 main.py ../samples/invalid_semantic_bad_split.pipeline
    python3 main.py ../samples/invalid_semantic_unknown_model.pipeline
    python3 main.py ../samples/edge_split_values.pipeline

Prints:
    1. The token stream produced by the lexer.
    2. The AST produced by the parser (or a clear syntax error).
    3. The result of semantic analysis (or a clear semantic error).
"""

import sys
from lexer import tokenize_source, LexerError
from parser import parse_source, ParserError
from semantic_analyzer import analyze_program, SemanticError
from ast_nodes import pretty_print


def run(path: str):
    with open(path, "r") as f:
        source = f.read()

    print(f"=== SOURCE: {path} ===")
    print(source)

    print("=== TOKENS ===")
    try:
        tokens = tokenize_source(source)
    except LexerError as e:
        print(e)
        return
    for tok in tokens:
        print(tok)

    print("\n=== AST ===")
    try:
        program = parse_source(tokens)
    except ParserError as e:
        print(e)
        return
    pretty_print(program)
    print("Syntax: ACCEPTED (parsed with no errors)")

    print("\n=== SEMANTIC ANALYSIS ===")
    try:
        analyze_program(program)
    except SemanticError as e:
        print(e)
        return
    print("Semantics: ACCEPTED (all checks passed)")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "../samples/valid_churn.pipeline"
    run(target)
