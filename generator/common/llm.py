import ast
import inspect
import textwrap
from typing import Callable

from langchain_community.callbacks import OpenAICallbackHandler


class _AttributeVisitor(ast.NodeVisitor):
    """Visits AST nodes to collect attributes accessed via 'self'."""

    def __init__(self):
        """
        Initialize the visitor.
        """
        self.attributes = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Visit an attribute node in the AST
        and add the attribute name to the set
        if accessed via 'self'.

        Args:
            node: The attribute node being visited
        """
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            self.attributes.add(node.attr)
        self.generic_visit(node)


def _get_attr_names_from_callable(_callable: Callable) -> set[str]:
    """
    Extracts the names of attributes accessed via 'self'
    in the given callable.

    Args:
        _callable: The callable from which to extract attribute names

    Returns:
        A set of attribute names accessed via 'self' within the callable

    Raises:
        ValueError: If the source code cannot be retrieved or parsed
        TypeError: If the provided object is not a callable
    """
    try:
        source_code = inspect.getsource(_callable)
        # Dedent the source code to fix indentation issues
        source_code = textwrap.dedent(source_code)
    except OSError as e:
        raise ValueError(
            "Failed to retrieve source code for the method. "
            f"Original error: {e}"
        ) from e

    try:
        tree = ast.parse(source_code)
        visitor = _AttributeVisitor()
        visitor.visit(tree)
        return visitor.attributes
    except SyntaxError as e:
        raise SyntaxError(
            f"Failed to parse the method source code: " f"{e}"
        ) from e


class LLMCostHandler:
    __repr__ = OpenAICallbackHandler.__repr__
    cost_types = _get_attr_names_from_callable(__repr__)

    def __init__(self):
        for cost_type in self.cost_types:
            setattr(self, cost_type, 0)

    def update_costs(self, cb_handler: OpenAICallbackHandler) -> None:
        for cost_type in self.cost_types:
            setattr(
                self,
                cost_type,
                (getattr(self, cost_type) + getattr(cb_handler, cost_type)),
            )
