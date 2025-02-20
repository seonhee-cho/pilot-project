from decimal import Decimal
import math

VARIABLES = {'x', 'y', 'z'}
OPERATORS = {'+', '-', '*', '/', '^', "'"}
MATH_CONSTANTS = {
    'pi': Decimal('3.141592653589793238462643383279502884197'),
    'e': Decimal('2.718281828459045235360287471352662497757'),
    'phi': Decimal((1 + math.sqrt(5)) / 2),
    'gamma': Decimal('0.577215664901532860606512090082402431042'),
} # tau = pi * 2

FUNCTIONS = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'exp': math.exp,
    'log': math.log,
    'arcsin': math.asin,
    'arccos': math.acos,
    'arctan': math.atan,
    'sqrt': math.sqrt,
    'abs': math.fabs,
    'ceil': math.ceil,
    'floor': math.floor,
    'round': round,
    'diff': lambda x: x.derivative(),
}