from decimal import Decimal
import sympy as sp
import math

VARIABLES = {"x", "y", "z"}
OPERATORS = {"+", "-", "*", "/", "^", "'"}
MATH_CONSTANTS = {
    "pi": Decimal("3.141592653589793238462643383279502884197"),
    "e": Decimal("2.718281828459045235360287471352662497757"),
    "phi": Decimal((1 + math.sqrt(5)) / 2),
    "gamma": Decimal("0.577215664901532860606512090082402431042"),
} # tau = pi * 2

FUNCTIONS_math = {
    "sin": math.sin,
    "sin'": math.cos,
    "cos": math.cos,
    "cos'": lambda x: -math.sin(x),
    "tan": math.tan,
    "tan'": lambda x: 1 / (math.cos(x) ** 2),
    "exp": math.exp,
    "exp'": math.exp,
    "log": math.log,
    "log'": lambda x: 1 / x,
    "ln": math.log,
    "ln'": lambda x: 1 / x,
    "arcsin": math.asin,
    "arcsin'": lambda x: 1 / math.sqrt(1 - x ** 2),
    "arccos": math.acos,
    "arccos'": lambda x: -1 / math.sqrt(1 - x ** 2),
    "arctan": math.atan,
    "arctan'": lambda x: 1 / (1 + x ** 2),
    "sqrt": math.sqrt,
    "sqrt'": lambda x: 1 / (2 * math.sqrt(x)),
    "abs": math.fabs,
    "abs'": lambda x: 1 if x > 0 else -1,
    "diff": lambda x: x.derivative(),
}

FUNCTIONS = {
    "sin": sp.sin,
    "sin'": sp.cos,
    "cos": sp.cos,
    "cos'": lambda x: -sp.sin(x),
    "tan": sp.tan,
    "tan'": lambda x: 1 / (sp.cos(x) ** 2),
    "exp": sp.exp,
    "exp'": sp.exp,
    "log": sp.log,
    "log'": lambda x: 1 / x,
    "ln": sp.ln,
    "ln'": lambda x: 1 / x,
    "arcsin": sp.asin,
    "arcsin'": lambda x: 1 / sp.sqrt(1 - x ** 2),
    "arccos": sp.acos,
    "arccos'": lambda x: -1 / sp.sqrt(1 - x ** 2),
    "arctan": sp.atan,
    "arctan'": lambda x: 1 / (1 + x ** 2),
    "sqrt": sp.sqrt,
    "sqrt'": lambda x: 1 / (2 * sp.sqrt(x)),
    "abs": sp.Abs,
    "abs'": lambda x: 1 if x > 0 else -1,
    "diff": lambda x: x.derivative(),
}

INVERSE_FUNCTIONS = {
    "ln": "exp",
    "exp": "ln",
    "sin": "arcsin",
    "cos": "arccos",
    "tan": "arctan",
    "arcsin": "sin",
    "arccos": "cos",
    "arctan": "tan",
}