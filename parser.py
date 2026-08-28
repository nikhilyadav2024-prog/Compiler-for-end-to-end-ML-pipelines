"""
parser.py
---------
Recursive-descent parser for the ML Pipeline DSL.

Responsibility (Review 1 scope):
    Consume the token stream produced by lexer.py, check it against
    the grammar, and build the AST (ast_nodes.py). Reports a clear
    syntax error with line/column on the first token that doesn't fit.

Grammar implemented (superset of the Review-1 slide grammar, extended
to cover preprocess/split/model/evaluate so the sample program parses
end to end):

    <program>     ::= <pipeline>
    <pipeline>    ::= "pipeline" ID "{" <statement>* "}"
    <statement>   ::= <load> | <preprocess> | <split>
                     | <model> | "train" ";" | <evaluate>
    <load>        ::= "load" "data" STRING ";"
    <preprocess>  ::= "preprocess" "{" <prep_step>* "}"
    <prep_step>   ::= "fill_missing" ID ";"
                     | "scale" ID ";"
                     | "encode" ID ";"
    <split>       ::= "split" "train" NUMBER "test" NUMBER ";"
    <model>       ::= "model" ID "{" <param>* "}"
    <param>       ::= ID "=" NUMBER ";"
    <evaluate>    ::= "evaluate" "metrics" "[" ID_LIST "]" ";"
"""

from lexer import Token, TokenType
from ast_nodes import (
    Program, Pipeline, LoadStmt, PreprocessBlock, FillMissingStep,
    ScaleStep, EncodeStep, SplitStmt, ModelBlock, ModelParam,
    TrainStmt, EvaluateStmt,
)


class ParserError(Exception):
    def __init__(self, message, token: Token):
        super().__init__(
            f"ParserError: {message} (got {token.type.name} {token.value!r} "
            f"at line {token.line}, col {token.col})"
        )
        self.token = token


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    # ---- token helpers ------------------------------------------------
    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _at_end(self) -> bool:
        return self._current().type == TokenType.EOF

    def _check(self, ttype, value=None) -> bool:
        tok = self._current()
        if tok.type != ttype:
            return False
        return value is None or tok.value == value

    def _advance(self) -> Token:
        tok = self._current()
        if not self._at_end():
            self.pos += 1
        return tok

    def _expect(self, ttype, value=None) -> Token:
        if self._check(ttype, value):
            return self._advance()
        expected = value if value is not None else ttype.name
        raise ParserError(f"expected {expected!r}", self._current())

    # ---- grammar rules --------------------------------------------------
    def parse_program(self) -> Program:
        pipeline = self._parse_pipeline()
        self._expect(TokenType.EOF)
        return Program(pipeline=pipeline)

    def _parse_pipeline(self) -> Pipeline:
        self._expect(TokenType.KEYWORD, "pipeline")
        name_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.SYMBOL, "{")

        statements = []
        while not self._check(TokenType.SYMBOL, "}"):
            if self._at_end():
                raise ParserError("unmatched '{' in pipeline block", self._current())
            statements.append(self._parse_statement())

        self._expect(TokenType.SYMBOL, "}")
        return Pipeline(name=name_tok.value, statements=statements)

    def _parse_statement(self):
        tok = self._current()
        if tok.type != TokenType.KEYWORD:
            raise ParserError("expected a statement keyword", tok)

        if tok.value == "load":
            return self._parse_load()
        if tok.value == "preprocess":
            return self._parse_preprocess()
        if tok.value == "split":
            return self._parse_split()
        if tok.value == "model":
            return self._parse_model()
        if tok.value == "train":
            self._advance()
            self._expect(TokenType.SYMBOL, ";")
            return TrainStmt()
        if tok.value == "evaluate":
            return self._parse_evaluate()

        raise ParserError(f"unexpected keyword {tok.value!r} in statement position", tok)

    def _parse_load(self) -> LoadStmt:
        self._expect(TokenType.KEYWORD, "load")
        self._expect(TokenType.KEYWORD, "data")
        path_tok = self._expect(TokenType.STRING)
        self._expect(TokenType.SYMBOL, ";")
        return LoadStmt(path=path_tok.value)

    def _parse_preprocess(self) -> PreprocessBlock:
        self._expect(TokenType.KEYWORD, "preprocess")
        self._expect(TokenType.SYMBOL, "{")
        steps = []
        while not self._check(TokenType.SYMBOL, "}"):
            if self._at_end():
                raise ParserError("unmatched '{' in preprocess block", self._current())
            steps.append(self._parse_preprocess_step())
        self._expect(TokenType.SYMBOL, "}")
        return PreprocessBlock(steps=steps)

    def _parse_preprocess_step(self):
        tok = self._current()
        if tok.type != TokenType.KEYWORD or tok.value not in (
            "fill_missing", "scale", "encode",
        ):
            raise ParserError(
                "expected 'fill_missing', 'scale' or 'encode'", tok
            )
        self._advance()
        col_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.SYMBOL, ";")
        if tok.value == "fill_missing":
            return FillMissingStep(column=col_tok.value)
        if tok.value == "scale":
            return ScaleStep(column=col_tok.value)
        return EncodeStep(column=col_tok.value)

    def _parse_split(self) -> SplitStmt:
        self._expect(TokenType.KEYWORD, "split")
        self._expect(TokenType.KEYWORD, "train")
        train_tok = self._expect(TokenType.NUMBER)
        self._expect(TokenType.KEYWORD, "test")
        test_tok = self._expect(TokenType.NUMBER)
        self._expect(TokenType.SYMBOL, ";")
        return SplitStmt(train_pct=int(train_tok.value), test_pct=int(test_tok.value))

    def _parse_model(self) -> ModelBlock:
        self._expect(TokenType.KEYWORD, "model")
        type_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.SYMBOL, "{")
        params = []
        while not self._check(TokenType.SYMBOL, "}"):
            if self._at_end():
                raise ParserError("unmatched '{' in model block", self._current())
            params.append(self._parse_param())
        self._expect(TokenType.SYMBOL, "}")
        return ModelBlock(model_type=type_tok.value, params=params)

    def _parse_param(self) -> ModelParam:
        name_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.OPERATOR, "=")
        value_tok = self._expect(TokenType.NUMBER)
        self._expect(TokenType.SYMBOL, ";")
        return ModelParam(name=name_tok.value, value=int(value_tok.value))

    def _parse_evaluate(self) -> EvaluateStmt:
        self._expect(TokenType.KEYWORD, "evaluate")
        self._expect(TokenType.KEYWORD, "metrics")
        self._expect(TokenType.SYMBOL, "[")
        metrics = []
        if not self._check(TokenType.SYMBOL, "]"):
            metrics.append(self._expect(TokenType.IDENTIFIER).value)
            while self._check(TokenType.SYMBOL, ","):
                self._advance()
                metrics.append(self._expect(TokenType.IDENTIFIER).value)
        self._expect(TokenType.SYMBOL, "]")
        self._expect(TokenType.SYMBOL, ";")
        return EvaluateStmt(metrics=metrics)


def parse_source(tokens) -> Program:
    """Convenience wrapper used by main.py and the tests."""
    return Parser(tokens).parse_program()
