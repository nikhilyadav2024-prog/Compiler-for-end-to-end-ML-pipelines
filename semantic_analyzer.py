"""
semantic_analyzer.py
---------------------
Semantic analysis pass for the ML Pipeline DSL.

Responsibility:
    Walk the AST produced by parser.py and check the rules that sit
    outside the grammar -- a program can be syntactically perfect and
    still be meaningless (a 70/50 split, an unsupported model type, a
    pipeline with no 'train' stage at all). The parser cannot catch
    any of this because it only checks token shape, not meaning.

    This pass runs strictly after parsing succeeds and does not modify
    the AST -- it only validates it and raises SemanticError with a
    clear, line-numbered message on the first problem found.

Not in scope here: optimization and code generation.
"""

from ast_nodes import (
    Program, LoadStmt, PreprocessBlock, SplitStmt, ModelBlock,
    TrainStmt, EvaluateStmt,
)

# Supported model types and the parameter names each one accepts.
# (Parameter values in this DSL are always whole numbers -- see the
# lexer's NUMBER token -- so every allowed parameter here is an int.)
SUPPORTED_MODELS = {
    "random_forest": {"trees", "max_depth"},
    "logistic_regression": {"max_iter"},
    "decision_tree": {"max_depth"},
    "svm": {"c"},
    "neural_network": {"layers", "epochs"},
}

SUPPORTED_METRICS = {"accuracy", "precision", "recall", "f1", "auc", "rmse", "mae"}


class SemanticError(Exception):
    def __init__(self, message, line=None):
        loc = f" (line {line})" if line else ""
        super().__init__(f"SemanticError: {message}{loc}")
        self.line = line


class SemanticAnalyzer:
    """Single-pass semantic checker.

    analyze() raises SemanticError on the first violation it finds and
    otherwise returns True. It does not attempt error recovery -- for
    a student compiler project, one clear error at a time is more
    useful than a partial error list.
    """

    def __init__(self, program: Program):
        self.program = program

    def analyze(self) -> bool:
        pipeline = self.program.pipeline
        stmts = pipeline.statements

        loads = [s for s in stmts if isinstance(s, LoadStmt)]
        splits = [s for s in stmts if isinstance(s, SplitStmt)]
        trains = [s for s in stmts if isinstance(s, TrainStmt)]
        models = [s for s in stmts if isinstance(s, ModelBlock)]
        evaluates = [s for s in stmts if isinstance(s, EvaluateStmt)]

        self._check_required_stage(loads, "load", pipeline.name)
        self._check_required_stage(splits, "split", pipeline.name)
        self._check_required_stage(trains, "train", pipeline.name)

        if len(models) > 1:
            raise SemanticError(
                f"pipeline {pipeline.name!r} defines more than one 'model' block",
                models[1].line,
            )
        if len(evaluates) > 1:
            raise SemanticError(
                f"pipeline {pipeline.name!r} defines more than one 'evaluate' statement",
                evaluates[1].line,
            )

        self._check_order(stmts)

        for split in splits:
            self._check_split(split)
        for model in models:
            self._check_model(model)
        for ev in evaluates:
            self._check_evaluate(ev)

        return True  # semantically accepted

    # ---- individual checks --------------------------------------------
    def _check_required_stage(self, found, stage_name, pipeline_name):
        if not found:
            raise SemanticError(
                f"pipeline {pipeline_name!r} is missing a required {stage_name!r} stage"
            )
        if len(found) > 1:
            raise SemanticError(
                f"pipeline {pipeline_name!r} defines the {stage_name!r} stage more than once",
                found[1].line,
            )

    def _check_order(self, stmts):
        # load must precede split; split must precede train;
        # model (if present) must precede train; evaluate (if present)
        # must follow train.
        first_index = {}
        for i, s in enumerate(stmts):
            first_index.setdefault(type(s), i)

        load_i = first_index.get(LoadStmt)
        split_i = first_index.get(SplitStmt)
        train_i = first_index.get(TrainStmt)
        model_i = first_index.get(ModelBlock)
        eval_i = first_index.get(EvaluateStmt)

        if load_i is not None and split_i is not None and load_i > split_i:
            raise SemanticError("'load' must appear before 'split'")
        if split_i is not None and train_i is not None and split_i > train_i:
            raise SemanticError("'split' must appear before 'train'")
        if model_i is not None and train_i is not None and model_i > train_i:
            raise SemanticError("'model' must appear before 'train'")
        if eval_i is not None and train_i is not None and eval_i < train_i:
            raise SemanticError("'evaluate' must appear after 'train'")

    def _check_split(self, split: SplitStmt):
        if split.train_pct <= 0 or split.test_pct <= 0:
            raise SemanticError(
                f"split percentages must be positive, got train={split.train_pct}, "
                f"test={split.test_pct}",
                split.line,
            )
        total = split.train_pct + split.test_pct
        if total != 100:
            raise SemanticError(
                f"split percentages must add up to 100, got {total} "
                f"(train={split.train_pct}, test={split.test_pct})",
                split.line,
            )

    def _check_model(self, model: ModelBlock):
        if model.model_type not in SUPPORTED_MODELS:
            supported = ", ".join(sorted(SUPPORTED_MODELS))
            raise SemanticError(
                f"unsupported model type {model.model_type!r} (supported: {supported})",
                model.line,
            )
        allowed_params = SUPPORTED_MODELS[model.model_type]
        seen = set()
        for param in model.params:
            if param.name in seen:
                raise SemanticError(
                    f"duplicate parameter {param.name!r} in model {model.model_type!r}",
                    param.line,
                )
            seen.add(param.name)
            if param.name not in allowed_params:
                raise SemanticError(
                    f"unknown parameter {param.name!r} for model {model.model_type!r} "
                    f"(allowed: {', '.join(sorted(allowed_params))})",
                    param.line,
                )
            if param.value <= 0:
                raise SemanticError(
                    f"parameter {param.name!r} must be positive, got {param.value}",
                    param.line,
                )

    def _check_evaluate(self, ev: EvaluateStmt):
        if not ev.metrics:
            raise SemanticError("'evaluate' must list at least one metric", ev.line)
        seen = set()
        for m in ev.metrics:
            if m not in SUPPORTED_METRICS:
                supported = ", ".join(sorted(SUPPORTED_METRICS))
                raise SemanticError(
                    f"unsupported metric {m!r} (supported: {supported})", ev.line
                )
            if m in seen:
                raise SemanticError(f"duplicate metric {m!r} in 'evaluate'", ev.line)
            seen.add(m)


def analyze_program(program: Program) -> bool:
    """Convenience wrapper used by main.py and the tests."""
    return SemanticAnalyzer(program).analyze()
