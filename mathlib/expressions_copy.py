from decimal import Decimal
from abc import ABC, abstractmethod
from typing import Union

import sympy
from utils import apply_operator
from constants import FUNCTIONS

class Expression(ABC):
    """
    수학식 (함수, 상수, 변수, 연산 등)을 추상적으로 표현하는 클래스
    단변수/다변수 함수 표현 가능.
    """
    @abstractmethod
    def evaluate(self, env: dict = None) -> Decimal:
        pass
    
    def __repr__(self):
        return self._repr_tree(0)
    
    def canonicalize(self):
        pass

    def derivative(self, var: str) -> 'Expression':
        # 단변수
        pass

class NumExpression(Expression):
    def __init__(self, value: Decimal):
        self.value = value

    def evaluate(self, env: dict = None) -> Decimal:
        return Decimal(self.value)

    def derivative(self, var: str) -> 'Expression':
        return NumExpression(0)
    
    def canonicalize(self):
        return self
    
    def __str__(self):
        return str(self.value)
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}NumExpression({self.value})"
    

class UnaryOpExpression(Expression):
    def __init__(self, op: str, expr: Expression):
        self.op = op
        self.expr = expr

    def evaluate(self, env: dict = None) -> Decimal:
        val = self.expr.evaluate(env)
        if self.op == '+':
            return Decimal(val)
        elif self.op == '-':
            return -Decimal(val)
        else:
            raise ValueError(f"Invalid operator: {self.op}")
        
    def derivative(self, var: str) -> 'Expression':
        if self.op == '+':
            return self.expr.derivative(var)
        elif self.op == '-':
            return -self.expr.derivative(var)
        else:
            raise ValueError(f"Invalid operator: {self.op}")
        
    def canonicalize(self):
        canonical_expr = self.expr.canonicalize()
        if self.op == '+':
            return canonical_expr
        else: # '-'
            if isinstance(canonical_expr, NumExpression):
                return NumExpression(-canonical_expr.value)
            
            elif isinstance(canonical_expr, UnaryOpExpression):
                if canonical_expr.op == '+':
                    return UnaryOpExpression(self.op, canonical_expr.expr)
                else:
                    return canonical_expr.expr
            else:
                return UnaryOpExpression(self.op, canonical_expr)
            
    def __str__(self):
        return f"({self.op}{self.expr})"
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}UnaryOpExpression({self.op})\n{self.expr._repr_tree(level + 2)}"
    

class BinOpExpression(Expression):
    def __init__(self, left: Expression, op: str, right: Expression):
        self.left = left
        self.op = op
        self.right = right
        
    def evaluate(self, env: dict = None) -> Decimal:
        left_val = self.left.evaluate(env)
        right_val = self.right.evaluate(env)

        if isinstance(left_val, Decimal) and isinstance(right_val, Decimal):
            return Decimal(apply_operator(self.op, left_val, right_val))
        else:
            return BinOpExpression(left_val, self.op, right_val)
    
    def differentiate(self, var: str = "") -> 'Expression':
        if var:
            out = self.derivative(var)
        else:
            out = self.gradient()
        
        # simplify
        return out.canonicalize()
    
    def derivative(self, var: str) -> 'Expression':
        left_deriv = self.left.derivative(var)
        right_deriv = self.right.derivative(var)

        if self.op in ('+', '-'):
            return BinOpExpression(left_deriv, self.op, right_deriv)
                
        elif self.op == '*':
            return BinOpExpression(
                BinOpExpression(left_deriv, '*', self.right),
                '+',
                BinOpExpression(self.left, '*', right_deriv)
            )
        
        elif self.op == '/':
            # (u / v)' = (u'v - uv') / (v^2)
            numerator = BinOpExpression(
                BinOpExpression(left_deriv, '*', self.right),
                '-',
                BinOpExpression(self.left, '*', right_deriv)
            )
            denominator = BinOpExpression(self.right, '^', NumExpression(2))
            return BinOpExpression(numerator, '/', denominator)
        
        elif self.op == '^':
            # (x^n)' = n * x^(n-1)
                # if node.base is Variable and node.exponent is Number:
                #     return node.exponent * (node.base ^ (node.exponent - 1))
            if isinstance(self.left, VarExpression) and self.left.name == var and isinstance(self.right, NumExpression):
                return BinOpExpression(
                    self.right,
                    '*',
                    BinOpExpression(
                        self.left,
                        '^',
                        NumExpression(self.right.value - 1)
                    ).canonicalize()
                )
            # else:
            #     raise NotImplementedError("Exponentiation derivative is not implemented.")
        
    def canonicalize(self) -> 'Expression':
        left = self.left.canonicalize()
        right = self.right.canonicalize()
        # ((0*x) + (3*1))
        # 좌항과 우항이 모두 숫자일 경우, 계산하여 단순화
        if isinstance(left, NumExpression) and isinstance(right, NumExpression):
            return NumExpression(apply_operator(self.op, left.value, right.value))
        
        # 덧셈이나 뺄셈에서 0을 제거
        if self.op in ('+', '-') and isinstance(left, NumExpression) and left.value == 0:
            return right
        if self.op in ('+', '-') and isinstance(right, NumExpression) and right.value == 0:
            return left
        
        # 곱셈에서 1을 제거
        if self.op == '*' and isinstance(left, NumExpression) and left.value == 1:
            return right
        if self.op == '*' and isinstance(right, NumExpression) and right.value == 1:
            return self.left.canonicalize()
        
        # 곱셈에서 0을 반환
        if self.op == '*' and (isinstance(left, NumExpression) and left.value == 0 or
                               isinstance(right, NumExpression) and right.value == 0):
            return NumExpression(0)
        
        # 나눗셈에서 1을 제거
        if self.op == '/' and isinstance(right, NumExpression) and right.value == 1:
            return left
        
        # 거듭제곱에서 1을 제거 (x^1 = x)
        if self.op == '^' and isinstance(right, NumExpression) and right.value == 1:
            return left
        
        # 거듭제곱에서 0을 처리 (x^0 = 1, 단 x != 0)
        if self.op == '^' and isinstance(right, NumExpression) and right.value == 0:
            return NumExpression(1)
        
        # 기본적으로 자기 자신을 반환
        return self
    
    def __str__(self):
        return f"({self.left} {self.op} {self.right})"
    
    def _repr_tree(self, level=0):
        return (
            f"{'  ' * level}BinOpExpression({self.op})\n"
            f"{self.left._repr_tree(level + 2)}\n"
            f"{self.right._repr_tree(level + 2)}"
        )


class VarExpression(Expression):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, env: dict = None) -> Decimal:
        if env is None or self.name not in env or isinstance(env[self.name], sympy.Symbol):
            return self
        
        return Decimal(env[self.name])
    
    def derivative(self, var: str) -> 'Expression':
        if self.name == var:
            return NumExpression(1)
        else:
            return NumExpression(0)

    def canonicalize(self):
        return self
    
    def __str__(self):
        return self.name

    def _repr_tree(self, level=0):
        return f"{'  ' * level}VarExpression({self.name})"
    

class SingleVarFunction(Expression):
    def __init__(self, func: str, expr: Expression):
        if func not in FUNCTIONS:
            raise ValueError(f"Invalid function: {func}")
        self.func = func
        self.expr = expr

    def evaluate(self, env: dict = None) -> Union[Decimal, 'Expression', None]:
        if self.func == 'diff':
            return self.expr.derivative(self.expr.name)
        
        elif isinstance(self.expr, VarExpression) and isinstance(env[self.expr.name], sympy.Symbol):
            return self
        
        val = self.expr.evaluate(env)
        return Decimal(FUNCTIONS[self.func](val))
    
    def derivative(self, var: str) -> 'Expression':
        if self.func == 'diff':
            return self.expr.derivative(var)
        
        derivative_expr = self.expr.derivative(var)
        if isinstance(derivative_expr, NumExpression):
            return NumExpression(0)
        else:
            return SingleVarFunction(self.func, derivative_expr)
        
    def canonicalize(self):
        canonical_expr = self.expr.canonicalize()
        return SingleVarFunction(self.func, canonical_expr)
    
    def __str__(self):
        return f"{self.func}({self.expr})"
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}SingleVarFunction({self.func})\n{self.expr._repr_tree(level + 2)}"
    

class MultiVarFunction(Expression):
    def __init__(self, func: str, *args: Expression):
        if func not in FUNCTIONS:
            raise ValueError(f"Invalid function: {func}")
        self.func = func
        self.args = args

    def evaluate(self, env: dict = None) -> Decimal:
        
        if self.func == 'diff':
            _func = self.args[0] # BinOpExpression(x^2 + y^2)
            _var = self.args[1:] # [VarExpression(x)]
            if len(_var) == 1:
                return _func.differentiate(var=_var[0].name)
            elif len(_var) > 1:
                return _func.differentiate(direction=[arg.name for arg in _var])
            
        if self.func == 'grad':
            # grad 함수의 경우 재귀적으로 미분
            # 1. 모든 변수에 대해 편미분
            # 2. 모든 변수에 대해 편미분한 결과를 Gradient 형태로 반환
            pass
            
        arg_values = [arg.evaluate(env) for arg in self.args]
        return Decimal(FUNCTIONS[self.func](*arg_values))
    
    def differentiate(self, var: str = "", direction: list[Decimal]=None) -> 'Expression':
        # diff 함수의 경우 재귀적으로 미분
            # 1. 하나의 변수에 대해 편미분
            # 2. 모든 변수에 대해 편미분한 결과를 Gradient 형태로 반환
            # 3. 방향 벡터로 미분
        if var:
            return self.func.derivative(var)
        elif direction:
            return [self.func.derivative(var) for var in direction]
        else: # gradient 반환
            return self.func.gradient()

    def derivative(self, var: str) -> 'Expression':
        
        return MultiVarFunction(self.func, *[arg.derivative(var) for arg in self.args])
    
    def canonicalize(self):
        return MultiVarFunction(self.func, *[arg.canonicalize() for arg in self.args])

    def __str__(self):
        return f"{self.func}({', '.join(str(arg) for arg in self.args)})"

    def _repr_tree(self, level=0):
        return f"{'  ' * level}MultiVarFunction({self.func})\n" + \
            f"{'  ' * (level + 2)}" + '\n,\n'.join(arg._repr_tree(0) for arg in self.args)
        

def get_sort_key(expr: Expression) -> tuple:
    """
    Node를 (rank, value) 형태의 튜플로 변환
    - FuncNode -> (0, func_name)
    - VarNode -> (1, var_name)
    - NumNode -> (3, num_value)
    - UnaryOpNode, BinOpNode -> (2, str(node))
    """
    if isinstance(expr, SingleVarFunction):
        return (0, expr.func)
    elif isinstance(expr, MultiVarFunction):
        return (0, expr.func)
    elif isinstance(expr, VarExpression):
        return (1, expr.name)
    elif isinstance(expr, NumExpression):
        return (3, expr.value)
    else:
        return (2, str(expr))

def compare_expressions(expr1: Expression, expr2: Expression) -> int:
    """
    -1: expr1 < expr2
     0: expr1 == expr2
     1: expr1 > expr2

    - 숫자 vs. 숫자 : 크기 비교
    - 숫자 vs. 변수 : 변수가 먼저
    - 변수 vs. 변수 : 사전순 비교
    - 함수 vs. 함수 : 함수 이름 사전순 비교
    - 함수 vs. 숫자/변수 : 함수가 먼저
    """
    k1, k2 = get_sort_key(expr1), get_sort_key(expr2)
    if k1 == k2:
        return 0
    elif k1 < k2:
        return -1
    else:
        return 1