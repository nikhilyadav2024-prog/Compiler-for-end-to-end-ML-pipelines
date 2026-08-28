import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lexer import tokenize_source
from parser import parse_source, ParserError
from ast_nodes import (
    LoadStmt, PreprocessBlock, SplitStmt, ModelBlock, TrainStmt, EvaluateStmt,
)


VALID_PROGRAM = """
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
"""


class TestParserValidProgram(unittest.TestCase):
    def setUp(self):
        tokens = tokenize_source(VALID_PROGRAM)
        self.program = parse_source(tokens)

    def test_pipeline_name(self):
        self.assertEqual(self.program.pipeline.name, "churn_prediction")

    def test_statement_count_and_order(self):
        stmt_types = [type(s) for s in self.program.pipeline.statements]
        self.assertEqual(
            stmt_types,
            [LoadStmt, PreprocessBlock, SplitStmt, ModelBlock, TrainStmt, EvaluateStmt],
        )

    def test_preprocess_steps(self):
        block = self.program.pipeline.statements[1]
        self.assertEqual(len(block.steps), 3)

    def test_model_params(self):
        model = self.program.pipeline.statements[3]
        self.assertEqual(model.model_type, "random_forest")
        self.assertEqual(model.params[0].name, "trees")
        self.assertEqual(model.params[0].value, 100)

    def test_evaluate_metrics(self):
        ev = self.program.pipeline.statements[5]
        self.assertEqual(ev.metrics, ["accuracy", "f1"])


class TestParserInvalidProgram(unittest.TestCase):
    def test_missing_semicolon_raises_clear_error(self):
        source = """
        pipeline broken_example {
            load data "customer.csv"
            train;
        }
        """
        tokens = tokenize_source(source)
        with self.assertRaises(ParserError) as ctx:
            parse_source(tokens)
        # the parser should point at the 'train' keyword, since the
        # missing ';' after the load statement is only discoverable there
        self.assertIn("line", str(ctx.exception))

    def test_unmatched_brace_raises(self):
        source = 'pipeline p { load data "a.csv";'
        tokens = tokenize_source(source)
        with self.assertRaises(ParserError):
            parse_source(tokens)

    def test_unknown_keyword_raises(self):
        source = 'pipeline p { fly data "a.csv"; }'
        tokens = tokenize_source(source)
        with self.assertRaises(ParserError):
            parse_source(tokens)


class TestParserEdgeCase(unittest.TestCase):
    def test_edge_split_values_still_parses(self):
        # Grammatically legal even though 100/0 is a questionable split;
        # rejecting that is a semantic-analysis concern (Review 2).
        source = """
        pipeline edge_case_split {
            load data "data.csv";
            split train 100 test 0;
            train;
        }
        """
        tokens = tokenize_source(source)
        program = parse_source(tokens)
        split_stmt = program.pipeline.statements[1]
        self.assertEqual((split_stmt.train_pct, split_stmt.test_pct), (100, 0))

    def test_empty_preprocess_block_parses(self):
        source = """
        pipeline edge_empty_preprocess {
            load data "data.csv";
            preprocess { }
            train;
        }
        """
        tokens = tokenize_source(source)
        program = parse_source(tokens)
        block = program.pipeline.statements[1]
        self.assertEqual(block.steps, [])


if __name__ == "__main__":
    unittest.main()
