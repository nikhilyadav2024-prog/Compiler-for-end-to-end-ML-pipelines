import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lexer import tokenize_source
from parser import parse_source
from semantic_analyzer import analyze_program, SemanticError


def analyze(source: str):
    tokens = tokenize_source(source)
    program = parse_source(tokens)
    return analyze_program(program)


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


class TestSemanticValidProgram(unittest.TestCase):
    def test_valid_program_accepted(self):
        self.assertTrue(analyze(VALID_PROGRAM))

    def test_program_without_optional_stages_accepted(self):
        source = """
        pipeline minimal {
            load data "data.csv";
            split train 80 test 20;
            train;
        }
        """
        self.assertTrue(analyze(source))


class TestSemanticRequiredStages(unittest.TestCase):
    def test_missing_load_rejected(self):
        source = """
        pipeline no_load {
            split train 80 test 20;
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)

    def test_missing_split_rejected(self):
        source = """
        pipeline no_split {
            load data "data.csv";
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)

    def test_missing_train_rejected(self):
        source = """
        pipeline no_train {
            load data "data.csv";
            split train 80 test 20;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)

    def test_duplicate_load_rejected(self):
        source = """
        pipeline double_load {
            load data "data.csv";
            load data "other.csv";
            split train 80 test 20;
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)


class TestSemanticSplit(unittest.TestCase):
    def test_split_not_summing_to_100_rejected(self):
        source = """
        pipeline bad_split {
            load data "data.csv";
            split train 70 test 50;
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)

    def test_zero_split_rejected(self):
        # Grammatically legal (see the parser's edge-case tests) but
        # semantically meaningless -- a 0% test split trains on
        # everything and evaluates on nothing.
        source = """
        pipeline edge_case_split {
            load data "data.csv";
            split train 100 test 0;
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)


class TestSemanticModel(unittest.TestCase):
    def test_unsupported_model_type_rejected(self):
        source = """
        pipeline unknown_model {
            load data "data.csv";
            split train 80 test 20;
            model gradient_boost {
                trees = 50;
            }
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)

    def test_unknown_param_rejected(self):
        source = """
        pipeline bad_param {
            load data "data.csv";
            split train 80 test 20;
            model random_forest {
                learning_rate = 1;
            }
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)

    def test_duplicate_param_rejected(self):
        source = """
        pipeline dup_param {
            load data "data.csv";
            split train 80 test 20;
            model random_forest {
                trees = 50;
                trees = 100;
            }
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)


class TestSemanticEvaluate(unittest.TestCase):
    def test_unsupported_metric_rejected(self):
        source = """
        pipeline bad_metric {
            load data "data.csv";
            split train 80 test 20;
            train;
            evaluate metrics [made_up_metric];
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)

    def test_duplicate_metric_rejected(self):
        source = """
        pipeline dup_metric {
            load data "data.csv";
            split train 80 test 20;
            train;
            evaluate metrics [accuracy, accuracy];
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)


class TestSemanticOrdering(unittest.TestCase):
    def test_split_before_load_rejected(self):
        source = """
        pipeline wrong_order {
            split train 80 test 20;
            load data "data.csv";
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)

    def test_evaluate_before_train_rejected(self):
        source = """
        pipeline wrong_order {
            load data "data.csv";
            split train 80 test 20;
            evaluate metrics [accuracy];
            train;
        }
        """
        with self.assertRaises(SemanticError):
            analyze(source)


if __name__ == "__main__":
    unittest.main()
