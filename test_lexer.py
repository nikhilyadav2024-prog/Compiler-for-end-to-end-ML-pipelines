import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lexer import tokenize_source, TokenType, LexerError


class TestLexer(unittest.TestCase):
    def test_keywords_and_identifier(self):
        tokens = tokenize_source("pipeline churn_prediction {")
        types = [t.type for t in tokens[:-1]]  # drop EOF
        self.assertEqual(
            types,
            [TokenType.KEYWORD, TokenType.IDENTIFIER, TokenType.SYMBOL],
        )

    def test_string_literal(self):
        tokens = tokenize_source('load data "customer.csv";')
        string_tok = tokens[2]
        self.assertEqual(string_tok.type, TokenType.STRING)
        self.assertEqual(string_tok.value, "customer.csv")

    def test_number_literal(self):
        tokens = tokenize_source("split train 80 test 20;")
        numbers = [t.value for t in tokens if t.type == TokenType.NUMBER]
        self.assertEqual(numbers, ["80", "20"])

    def test_operator_and_symbols(self):
        tokens = tokenize_source("trees = 100;")
        types = [t.type for t in tokens[:-1]]
        self.assertEqual(
            types,
            [TokenType.IDENTIFIER, TokenType.OPERATOR, TokenType.NUMBER, TokenType.SYMBOL],
        )

    def test_unterminated_string_raises(self):
        with self.assertRaises(LexerError):
            tokenize_source('load data "oops;')

    def test_unexpected_character_raises(self):
        with self.assertRaises(LexerError):
            tokenize_source("pipeline foo # bar")

    def test_line_and_col_tracking(self):
        tokens = tokenize_source("pipeline foo\n{\n}")
        brace_open = tokens[2]
        self.assertEqual(brace_open.line, 2)


if __name__ == "__main__":
    unittest.main()
