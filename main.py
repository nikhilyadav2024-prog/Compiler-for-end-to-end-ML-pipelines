"""
main.py
-------
Review 1 demo driver.

Usage:
    python src/main.py samples/valid_churn.pipeline
    python src/main.py samples/invalid_missing_semicolon.pipeline

Prints:
    1. The token stream produced by the lexer.
    2. The AST produced by the parser (or a clear syntax error).
"""

import sys
from lexer import tokenize_source, LexerError
from parser import parse_source, ParserError
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
    print("\nResult: ACCEPTED (parsed with no errors)")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "samples/valid_churn.pipeline"
    run(target)
