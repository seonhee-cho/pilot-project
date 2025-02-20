from constants import *
from expressions import *
from utils import *
from parser import *

def test_lexer():
    exp = "log(3)+2"
    print(exp)
    tokens = Lexer.tokenize(exp)
    for i, token in enumerate(tokens): 
        print(f"{i}: {token.value} '{token.type}'")

    print("pi * 2 + e")
    tokens = Lexer.tokenize("pi * 2 + e")
    for i, token in enumerate(tokens):
        print(f"{i}: {token.value} '{token.type}'")

def test_canonicalize():
    test_expressions = [
        # "diff(3 * x, x)",
        # "diff(x*3, x)",
        # "log(2) + e",
        # "diff(x^2 + y^2, x)",
        # "diff(x^2 + y^2, y)",
        # "diff(x^2 + y^2, x, y)",
        "sin(diff(x^2 + y^2, x)))",
    ]
    for exp in test_expressions:
        print(f"Original: {exp}")
        result, ast = calculate_expression(exp, verbose=True)
        # print(f"## Canonicalized\n{ast.canonicalize()}")
        # print(f"## Result: {result}")
        print()


if __name__ == "__main__":
    # test_lexer()
    # main()
    test_canonicalize()