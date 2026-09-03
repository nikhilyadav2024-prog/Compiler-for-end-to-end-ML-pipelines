"""
lexer.py
--------
Lexical analyzer (tokenizer) for the ML Pipeline DSL.

Responsibility (Review 1 scope):
    Scan the raw source text character-by-character and emit a flat
    stream of Token objects for the parser to consume. The lexer does
    NOT understand grammar/structure -- that is the parser's job.
"""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    KEYWORD = auto()
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    SYMBOL = auto()
    OPERATOR = auto()
    EOF = auto()


# Every reserved word in the DSL. Anything else that looks like a word
# is treated as an IDENTIFIER.
KEYWORDS = {
    "pipeline", "load", "data", "preprocess", "fill_missing",
    "scale", "encode", "split", "train", "test", "model",
    "evaluate", "metrics",
}

SINGLE_CHAR_SYMBOLS = {"{", "}", "[", "]", ";", ","}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.col})"


class LexerError(Exception):
    """Raised for any character the DSL does not recognize."""
    def __init__(self, message, line, col):
        super().__init__(f"LexerError: {message} (line {line}, col {col})")
        self.line = line
        self.col = col


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1

    # ---- low level character helpers -------------------------------
    def _peek(self, offset=0):
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else ""

    def _advance(self):
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _skip_whitespace_and_comments(self):
        while self.pos < len(self.source):
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
            elif ch == "/" and self._peek(1) == "/":
                # line comment, not required by the grammar but useful
                # for the demo sample files.
                while self.pos < len(self.source) and self._peek() != "\n":
                    self._advance()
            else:
                break

    # ---- token producers ---------------------------------------------
    def _read_string(self):
        start_line, start_col = self.line, self.col
        self._advance()  # opening quote
        chars = []
        while True:
            if self.pos >= len(self.source):
                raise LexerError("unterminated string literal", start_line, start_col)
            ch = self._advance()
            if ch == '"':
                break
            chars.append(ch)
        return Token(TokenType.STRING, "".join(chars), start_line, start_col)

    def _read_number(self):
        start_line, start_col = self.line, self.col
        chars = []
        while self._peek().isdigit():
            chars.append(self._advance())
        return Token(TokenType.NUMBER, "".join(chars), start_line, start_col)

    def _read_word(self):
        start_line, start_col = self.line, self.col
        chars = []
        while self._peek().isalnum() or self._peek() == "_":
            chars.append(self._advance())
        word = "".join(chars)
        ttype = TokenType.KEYWORD if word in KEYWORDS else TokenType.IDENTIFIER
        return Token(ttype, word, start_line, start_col)

    # ---- public API ----------------------------------------------------
    def tokenize(self):
        tokens = []
        while True:
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.source):
                tokens.append(Token(TokenType.EOF, "", self.line, self.col))
                break

            ch = self._peek()
            line, col = self.line, self.col

            if ch == '"':
                tokens.append(self._read_string())
            elif ch.isdigit():
                tokens.append(self._read_number())
            elif ch.isalpha() or ch == "_":
                tokens.append(self._read_word())
            elif ch == "=":
                self._advance()
                tokens.append(Token(TokenType.OPERATOR, "=", line, col))
            elif ch in SINGLE_CHAR_SYMBOLS:
                self._advance()
                tokens.append(Token(TokenType.SYMBOL, ch, line, col))
            else:
                raise LexerError(f"unexpected character {ch!r}", line, col)

        return tokens


def tokenize_source(source: str):
    """Convenience wrapper used by main.py and the tests."""
    return Lexer(source).tokenize()
