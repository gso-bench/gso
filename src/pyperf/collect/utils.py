import ast


def prepare_prob_script(tests: list[str]) -> str:
    """Clean and prepare a problem script for a problem instance"""
    # pick the first test
    prob_script = tests[0]

    # remove all docstrings
    tree = RemoveDocstrings().visit(ast.parse(prob_script))
    ast.fix_missing_locations(tree)

    # remove harness code
    tree = RemoveHarness().visit(tree)
    ast.fix_missing_locations(tree)

    # remove comments and return prob_script
    prob_script = ast.unparse(tree)

    return prob_script


class RemoveHarness(ast.NodeTransformer):
    def visit_Module(self, node):
        # Filter out the timeit.template assignment
        node.body = [
            n
            for n in node.body
            if not (
                isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Attribute)
                and isinstance(n.targets[0].value, ast.Name)
                and n.targets[0].value.id == "timeit"
                and n.targets[0].attr == "template"
            )
        ]

        # Filter out the main() function definition
        node.body = [
            n
            for n in node.body
            if not (isinstance(n, ast.FunctionDef) and n.name == "main")
        ]

        # Filter out the if __name__ == '__main__' block
        node.body = [
            n
            for n in node.body
            if not (
                isinstance(n, ast.If)
                and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "__name__"
                and isinstance(n.test.ops[0], ast.Eq)
                and isinstance(n.test.comparators[0], ast.Constant)
                and n.test.comparators[0].value == "__main__"
            )
        ]

        return node


class RemoveDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            # When using Python 3.8+, string constants are of type ast.Constant.
            node.body.pop(0)
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            node.body.pop(0)
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            node.body.pop(0)
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Str, ast.Constant))
        ):
            node.body.pop(0)
        return node
