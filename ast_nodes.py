"""
ast_nodes.py
------------
AST node definitions produced by parser.py.

Each node is a plain dataclass -- no behaviour yet. Semantic analysis,
optimization and code generation (Review 2 & 3) will walk this tree.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Program:
    pipeline: "Pipeline"


@dataclass
class Pipeline:
    name: str
    statements: List["Statement"] = field(default_factory=list)


# Statement is a type alias for readability; any of the classes below
# can appear inside Pipeline.statements
Statement = object


@dataclass
class LoadStmt:
    path: str


@dataclass
class PreprocessBlock:
    steps: List["PreprocessStep"] = field(default_factory=list)


@dataclass
class FillMissingStep:
    column: str


@dataclass
class ScaleStep:
    column: str


@dataclass
class EncodeStep:
    column: str


PreprocessStep = object  # FillMissingStep | ScaleStep | EncodeStep


@dataclass
class SplitStmt:
    train_pct: int
    test_pct: int


@dataclass
class ModelParam:
    name: str
    value: int


@dataclass
class ModelBlock:
    model_type: str
    params: List[ModelParam] = field(default_factory=list)


@dataclass
class TrainStmt:
    pass


@dataclass
class EvaluateStmt:
    metrics: List[str] = field(default_factory=list)


def pretty_print(node, indent=0):
    """Small recursive printer used by main.py's demo output.
    Not part of the compiler pipeline itself -- just makes the AST
    readable on the terminal / in the review demo.
    """
    pad = "  " * indent
    if isinstance(node, Program):
        print(f"{pad}Program")
        pretty_print(node.pipeline, indent + 1)
    elif isinstance(node, Pipeline):
        print(f"{pad}Pipeline(name={node.name!r})")
        for stmt in node.statements:
            pretty_print(stmt, indent + 1)
    elif isinstance(node, LoadStmt):
        print(f"{pad}LoadStmt(path={node.path!r})")
    elif isinstance(node, PreprocessBlock):
        print(f"{pad}PreprocessBlock")
        for step in node.steps:
            pretty_print(step, indent + 1)
    elif isinstance(node, FillMissingStep):
        print(f"{pad}FillMissingStep(column={node.column!r})")
    elif isinstance(node, ScaleStep):
        print(f"{pad}ScaleStep(column={node.column!r})")
    elif isinstance(node, EncodeStep):
        print(f"{pad}EncodeStep(column={node.column!r})")
    elif isinstance(node, SplitStmt):
        print(f"{pad}SplitStmt(train={node.train_pct}, test={node.test_pct})")
    elif isinstance(node, ModelBlock):
        print(f"{pad}ModelBlock(type={node.model_type!r})")
        for p in node.params:
            pretty_print(p, indent + 1)
    elif isinstance(node, ModelParam):
        print(f"{pad}ModelParam({node.name}={node.value})")
    elif isinstance(node, TrainStmt):
        print(f"{pad}TrainStmt")
    elif isinstance(node, EvaluateStmt):
        print(f"{pad}EvaluateStmt(metrics={node.metrics})")
    else:
        print(f"{pad}{node!r}")
