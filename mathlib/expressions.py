
from abc import ABC
from typing import Union
from fractions import Fraction
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP

from constants import *
from interval import *


def apply_operator(op, a, b):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0:
            raise ZeroDivisionError("Division by zero.")
        return a / b
    elif op == '^':
        return a ** b
    else:
        raise ValueError(f"Invalid operator '{op}'")


class Expression(ABC):
    """
    수학식 (함수, 상수, 변수, 연산 등)을 추상적으로 표현하는 클래스
    
    초기화 시:
    vars_ : 변수 이름 집합
    domain : 변수 이름과 해당 변수의 정의역 집합
    
    # 메서드:
    (계산 관련)
    canonicalize() -> Expression: 수식 정규화
    evaluate(env: dict) -> Expression: 수식 계산
    
    (미분 관련)
    derivative(var: str) -> Expression: 미분
    gradient() -> Expression: 기울기
    
    (정의역 관련)
    is_continuous_at(var: str, point: Decimal) -> bool: 연속성 확인
    is_differentiable_at(var: str, point: Decimal) -> bool: 미분가능성 확인
    
    (상수 관련)
    is_constant() -> bool: 상수 여부 확인
    is_integer() -> bool: 정수 여부 확인
    
    (magic methods)
    __eq__(other: Expression) -> bool: 동등성 확인
    __hash__() -> int: 해시 값 반환
    __str__() -> str: 문자열 표현
    _repr_tree(level: int) -> str: 트리 표현

    __add__(other: Expression) -> Expression: 덧셈
    __sub__(other: Expression) -> Expression: 뺄셈
    __mul__(other: Expression) -> Expression: 곱셈
    __truediv__(other: Expression) -> Expression: 나눗셈
    __pow__(other: Expression) -> Expression: 거듭제곱

    """
    def __init__(self,
                  vars_: set[str],
                  domain: dict[str, Interval] = None,
                ):
        self.vars_ = vars_ if vars_ else set()
        self.domain = domain if domain else {}
        self._intrinsic_domain = self.domain

    def evaluate(self, env: dict = None) -> 'Expression':
        pass

    def canonicalize(self):
        pass

    def derivative(self, var: str) -> 'Expression':
        pass

    def directional_derivative(self, direction: dict) -> 'Expression':
        pass

    def gradient(self) -> 'Expression':
        if self.vars_:
            return UnionExpression( *[self.derivative(var) for var in self.vars_] )
        else:
            return NumExpression(0)
    
    def is_constant(self) -> bool:
        pass

    def is_integer(self) -> bool:
        pass
    
    def __str__(self):
        # 문자열 표현
        pass
    
    def __repr__(self):
        # 객체 타입을 포함한 트리 구조 표현
        return self._repr_tree()
    
    def _repr_tree(self, level=0):
        # 객체 타입을 포함한 트리 구조 표현
        pass
    
    def _domain_str(self):
        # 정의역 문자열 표현
        if not self.domain:
            return ""
        
        pairs = []
        for var, interval in self.domain.items():
            pairs.append(f"{var}: {interval}")
        return "\n".join(pairs)

    def __add__(self, other: 'Expression') -> 'Expression':
        return BinOpExpression(self, '+', other).canonicalize()
    
    def __sub__(self, other: 'Expression') -> 'Expression':
        return BinOpExpression(self, '-', other).canonicalize()
    
    def __mul__(self, other: 'Expression') -> 'Expression':
        if isinstance(other, Decimal):
            other = NumExpression(other)
        return BinOpExpression(self, '*', other).canonicalize()
    
    def __truediv__(self, other: 'Expression') -> 'Expression':
        if isinstance(other, Decimal):
            other = NumExpression(other)
        return BinOpExpression(self, '/', other).canonicalize()
    
    def __pow__(self, other: 'Expression') -> 'Expression':
        if isinstance(other, Decimal):
            other = NumExpression(other)
        return BinOpExpression(self, '^', other).canonicalize()
    
    def __eq__(self, other: 'Expression') -> bool:
        pass

    def __hash__(self):
        pass


class NumExpression(Expression):
    def __init__(self, value: Decimal, name: str = ""):
        super().__init__(
            vars_=set(),
            domain={},
            # domain={"constant": Interval.all_real()},
            # range_=Interval(name, value, value, closed_start=True, closed_end=True)
        )
        self.value = value if isinstance(value, Decimal) else Decimal(value)
        try:
            self.value = self.value.quantize(Decimal("1.0000000000"), rounding=ROUND_HALF_UP).normalize()
        except Exception as e:
            print(f"Error quantizing value: {e}")
        if self.value == 0: # -0 처리
            self.value = Decimal(0)
        self.name = name

    def evaluate(self, env: dict = None) -> 'Expression':
        return self

    def derivative(self, var: str) -> 'Expression':
        return NumExpression(0)
    
    def directional_derivative(self, direction: dict) -> 'Expression':
        return NumExpression(0)
    
    def is_constant(self) -> bool:
        return True
    
    def is_integer(self) -> bool:
        return self.value.is_integer()
    
    def canonicalize(self):
        return self
    
    def __eq__(self, other: 'Expression') -> bool:
        if not isinstance(other, NumExpression):
            return False
        return self.value == other.value
    
    def __le__(self, other: 'Expression') -> bool:
        if not isinstance(other, NumExpression):
            return False
        return self.value <= other.value
    
    def __ge__(self, other: 'Expression') -> bool:
        if not isinstance(other, NumExpression):
            return False
        return self.value >= other.value
    
    def __lt__(self, other: 'Expression') -> bool:
        if not isinstance(other, NumExpression):
            return False
        return self.value < other.value
    
    def __gt__(self, other: 'Expression') -> bool:
        if not isinstance(other, NumExpression):
            return False
        return self.value > other.value
    
    def __hash__(self):
        return hash(self.value)
    
    def __str__(self):
        if self.name:
            return self.name
        return str(self.value)
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}NumExpression({self.name if self.name else self.value})"
    
    def to_sympy(self) -> sp.Expr:
        return sp.Integer(self.value)


class UnaryOpExpression(Expression):
    def __init__(self, op: str, expr: Expression):
        self.op = op
        self.expr = expr
        super().__init__(vars_=expr.vars_, domain=expr.domain)


    def evaluate(self, env: dict = None) -> 'Expression':
        val = self.expr.evaluate(env)
        if isinstance(val, NumExpression):
            if self.op == '+':
                return val
            elif self.op == '-':
                return NumExpression( -Decimal(val.value) )
            else:
                raise ValueError(f"Invalid operator: {self.op}")
        else:
            output = UnaryOpExpression(self.op, val)
            output.domain = self.domain
            return output
        
    def canonicalize(self):
        # 부호 정리
        canonical_expr = self.expr.canonicalize()
        simplified_expr = self._simplify(canonical_expr)
        simplified_expr.domain = self.domain
        return simplified_expr
    
    def _simplify(self, _expr):
        # canonicalized 된 식을 받아서 부호 정리
        if self.op == '+':
            return _expr
        else: # '-'
            if isinstance(_expr, NumExpression):
                return NumExpression(-_expr.value)
            
            elif isinstance(_expr, UnaryOpExpression):
                if _expr.op == '+':
                    return UnaryOpExpression(self.op, _expr.expr)
                else:
                    return _expr.expr
            else:
                return UnaryOpExpression(self.op, _expr)
    
        
    def derivative(self, var: str) -> 'Expression':
        if self.op == '+':
            output = self.expr.derivative(var).canonicalize()
        elif self.op == '-':
            output = -self.expr.derivative(var).canonicalize()
        else:
            raise ValueError(f"Invalid operator: {self.op}")
        output.domain = self.domain
        return output
        
    def directional_derivative(self, direction: dict):
        deriv = []
        for var in self.vars_:
            coef = NumExpression(direction.get(var, 0))  # 해당 변수의 방향 성분; 없으면 0으로 간주
            deriv.append(coef * self.derivative(var))
        output = UnionExpression(*deriv)
        output.domain = self.domain
        return output

    def __eq__(self, other: Expression) -> bool:
        if not isinstance(other, UnaryOpExpression):
            return False
        return (self.op == other.op) and (self.expr == other.expr) and (self.domain == other.domain)

    def __hash__(self):
        return hash(str(self))
    
    def __str__(self):
        return f"({self.op}{self.expr})"
    
    def is_constant(self) -> bool:
        return self.expr.is_constant()
    
    def is_integer(self) -> bool:
        return self.expr.is_integer()
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}UnaryOpExpression({self.op})\n{self.expr._repr_tree(level + 2)}"


class BinOpExpression(Expression):
    def __init__(self, left: Expression, op: str, right: Expression):
        # 메타데이터 초기화 (정의역)
        combined_vars = left.vars_ | right.vars_
        merged_domains = merge_domains(left.domain, right.domain)
        if op == '/':
            for v in right.vars_:
                merged_domains[v].conditions.add( (right, "neq", NumExpression(0)) )
            
        if op == '^':
            # x^y
            # x==0 -> y>0
            if left.is_constant() and left.value == 0:
                if right.is_constant() and right.value <= 0:
                    raise ValueError("Domain error")
                for v in right.vars_:
                    merged_domains[v].conditions.add((right, "gt", NumExpression(0)))

            # x<0 -> y is integer (음수의 N제곱근은 존재하지 않기 때문) 혹은 y가 유리수이고, 분모가 홀수
            if left.is_constant() and left.value < 0:
                if right.is_constant() and (right.value % 1 != 0):
                    frac = Fraction(right.value).limit_denominator()
                    if frac.numerator % 2 == 0:
                        raise ValueError("Domain error")
                for v in right.vars_:
                    merged_domains[v].conditions.add((right, "integer|odd_nominator", True)) # TODO
            
            # 밑이 변수를 포함할 때
            for v in left.vars_:
                if right.is_constant():
                    if isinstance(right, NumExpression):
                        right_val = right.value
                    else:
                        right_val = right.evaluate().value

                    if right_val == 0.5: # sqrt(x) 일 때
                        merged_domains[v].conditions.add((left, "geq", NumExpression(0)))
                    
                    if right_val % 1 != 0:  # 비정수일 경우: x<0이면 안됨
                        frac = Fraction(right_val).limit_denominator()
                        if frac.numerator % 2 == 0:
                            merged_domains[v].conditions.add((left, "geq", NumExpression(0)))

                    if right_val <= 0: # 0이하일 경우: x==0이면 안됨
                        merged_domains[v].conditions.add((left, "neq", NumExpression(0)))

                else: # 둘다 변수 일 때 (e.g. x^y) -> 따로 처리 안함.
                    pass

        super().__init__(vars_=combined_vars, domain=merged_domains)
        self.left = left
        self.op = op
        self.right = right
        
    def evaluate(self, env: dict = None) -> Union['Expression', Decimal]:
        left_val = self.left.evaluate(env)
        right_val = self.right.evaluate(env)
        if isinstance(left_val, NumExpression) and isinstance(right_val, NumExpression):
            return apply_operator(self.op, left_val, right_val).canonicalize()
        else:
            output = BinOpExpression(left_val, self.op, right_val).canonicalize()
            output.domain = self.domain
            return output
    
    def canonicalize(self) -> 'Expression':
        """
        식 정규화
        1. 각 인자 정규화
        2. 로그 함수 정리 e.g. log(x^n) -> n*log(x)
        3. 분배 법칙 적용 e.g. (A+B) * C = A*C + B*C
        4. 동류항 합치기
        5. 식 재구성
        """
        left = self.left.canonicalize()
        right = self.right.canonicalize()
        if isinstance(left, NumExpression) and isinstance(right, NumExpression):
            if self.op == '+':
                return NumExpression(left.value + right.value)
            elif self.op == '-':
                return NumExpression(left.value - right.value)
            elif self.op == '*':
                return NumExpression(left.value * right.value)
            elif self.op == '/':
                return NumExpression(left.value / right.value)
            elif self.op == '^':
                return NumExpression(left.value ** right.value)
                
        output = simplify(BinOpExpression(left, self.op, right))
        output.domain = self.domain
        return output
    
    def derivative(self, var: str) -> 'Expression':
        self.left = self.left.canonicalize()
        self.right = self.right.canonicalize()
        left_deriv = self.left.derivative(var)
        right_deriv = self.right.derivative(var)

        if self.op in ('+', '-'):
            output = BinOpExpression(left_deriv, self.op, right_deriv).canonicalize()
                
        elif self.op == '*':
            output = left_deriv * self.right + self.left * right_deriv
        
        elif self.op == '/':
            # (u / v)' = (u'v - uv') / (v^2)
            numerator = left_deriv * self.right - self.left * right_deriv
            denominator = self.right ** NumExpression(2)
            output = BinOpExpression(numerator, '/', denominator).canonicalize()
        
        elif self.op == '^':
            if isinstance(self.left, VarExpression) and isinstance(self.right, NumExpression):
                if self.left.name == var:
                    output = self.right * (self.left ** NumExpression(self.right.value - 1))
                else:
                    output = NumExpression(0)

            else:
                # x^x, 3^x 같이 추가적인/복잡한 미분공식이 필요한 경우는 지원 x
                raise ValueError("Function derivative not available.")
        else:
            raise ValueError(f"Invalid operator: {self.op}")
        
        output.domain = self.domain
        return output.canonicalize()
    
    def directional_derivative(self, direction: dict) -> 'Expression':
        deriv = []
        for var in self.vars_:
            coef = NumExpression(direction.get(var, 0))  # 해당 변수의 방향 성분; 없으면 0으로 간주
            deriv.append(coef * self.derivative(var))
        output = UnionExpression(*deriv)
        output.domain = self.domain
        return output.canonicalize()
    
    def is_constant(self) -> bool:
        return self.left.is_constant() and self.right.is_constant()
    
    def is_integer(self) -> bool:
        return self.evaluate().is_integer()
    
    def __eq__(self, other: Expression) -> bool:
        if not isinstance(other, BinOpExpression):
            return False
        return (self.op == other.op) and (self.left == other.left) and (self.right == other.right) and (self.domain == other.domain)
    
    def __hash__(self):
        return hash(str(self))
    
    def __str__(self):
        if self.op == '^':
            return f"{self.left}^{{ {self.right} }}"
        else:
            return f"({self.left} {self.op} {self.right})"
    
    def _repr_tree(self, level=0):
        return (
            f"{'  ' * level}BinOpExpression({self.op})\n"
            f"{self.left._repr_tree(level + 2)}\n"
            f"{self.right._repr_tree(level + 2)}"
        )

class VarExpression(Expression):
    def __init__(self, name: str, interval: Interval = None):
        if name not in VARIABLES:
            raise ValueError(f"Invalid variable: {name}")
        domain_dict = {name: interval if interval is not None else Interval.all_real(name)}
        super().__init__(
            vars_={name},
            domain=domain_dict,
        )
        self.name = name

    def evaluate(self, env: dict = None) -> Decimal:
        try:
            if env is None or self.name not in env or isinstance(env[self.name], sp.Symbol):
                return self
            out = NumExpression( Decimal( env[self.name] ) )
        except:
            import pdb; pdb.set_trace()
        return NumExpression( Decimal( env[self.name] ) )
    
    def derivative(self, var: str) -> 'Expression':
        if self.name == var:
            output = NumExpression(1)
        else:
            output = NumExpression(0)

        output.domain = self.domain
        return output
        
    def directional_derivative(self, direction: dict) -> 'Expression':
        if self.name not in direction:
            return NumExpression(0)
        output = self.derivative(self.name) * direction[self.name]
        output.domain = self.domain
        return output
        
    def is_constant(self) -> bool:
        return False
    
    def is_integer(self) -> bool:
        return False

    def canonicalize(self):
        return self
    
    def __eq__(self, other: Expression) -> bool:
        if not isinstance(other, VarExpression):
            return False
        return (self.name == other.name)

    def __hash__(self):
        # domain_items = tuple(sorted(self.domain.items(), key=lambda x: x[0]))
        return hash(self.name)

    def __str__(self):
        return self.name

    def _repr_tree(self, level=0):
        return f"{'  ' * level}VarExpression({self.name})"
    
    def to_sympy(self) -> sp.Expr:
        return sp.symbols(self.name)


class SingleVarFunction(Expression):
    def __init__(self, func: str, expr: Expression):
        # 삼각함수: sin, cos, tan  /  역삼각함수: asin, acos, atan
        # 지수함수 : exp  /  (로그함수 : log, ln -> MultiVarFunction), (제곱근 : sqrt -> BinOpExpression)
        # 절대값 : abs  /  미분 : diff, grad
        if func.replace("'", "") not in FUNCTIONS:
            raise ValueError(f"Invalid function: {func}")
        self.func = func
        self.expr = expr

        # 메타 데이터 초기화 - 정의역 초기화 포함. 정의역 지정 시 이후에 변경
        combined_vars = expr.vars_
        domain = {v: expr.domain[v] for v in combined_vars}
        if func == "tan":
            condition = (SingleVarFunction("cos", expr), "neq", NumExpression(0))
            for v in combined_vars:
                domain[v].conditions.add(condition)

        if func in ("arcsin", "arccos"):
            # arcsin(x) 같은 꼴일 때는 바로 interval 객체로 처리
            if isinstance(expr, VarExpression):
                domain[expr.name] = domain[expr.name].intersects(Interval(expr.name, -1, 1, True, True))
            
            # arcsin(x^2) 같은 꼴일 때는 조건 추가
            else:
                condition = (expr, "leq", NumExpression(1))
                for v in combined_vars:
                    domain[v].conditions.add(condition)
                condition = (expr, "geq", NumExpression(-1))
                for v in combined_vars:
                    domain[v].conditions.add(condition)

        if func == "ln":
            condition = (expr, "gt", NumExpression(0))
            for v in combined_vars:
                domain[v].conditions.add(condition)

        super().__init__(vars_=expr.vars_, domain=domain)

    def evaluate(self, env: dict = None) -> Union[Decimal, 'Expression', None]:
        val_expr = self.expr.evaluate(env)
        # 함수 내부에 값이 아니라 또 다른 함수/연산자 등이 오는 경우
        if not isinstance(val_expr, NumExpression):
            output = SingleVarFunction(self.func, val_expr)
            output.domain = self.domain
            return output
        # 함수 값 계산이 가능한 경우
        try:
            return NumExpression( Decimal(str(FUNCTIONS[self.func](val_expr.value))) )
        except Exception as e:
            import pdb; pdb.set_trace()
            pass
    
    def canonicalize(self):
        canonical_expr = self.expr.canonicalize()
        if self.func in INVERSE_FUNCTIONS:
            if isinstance(canonical_expr, SingleVarFunction) and canonical_expr.func == INVERSE_FUNCTIONS[self.func]:
                canonical_expr.domain = self.domain
                return canonical_expr.expr
    
        expr = SingleVarFunction(self.func, simplify(canonical_expr))
        expr.domain = self.domain
        return expr
    
    
    def derivative(self, var: str) -> 'Expression':
        self.expr = self.expr.canonicalize()
        derivative_expr = self.expr.derivative(var)
        if self.func == "ln":
            output = BinOpExpression(NumExpression(1), "/", self.expr) * derivative_expr
        
        if self.func == "sin":
            output = SingleVarFunction("cos", self.expr) * derivative_expr
        
        if self.func == "cos":
            output = SingleVarFunction("sin", self.expr) * NumExpression(-1) * derivative_expr
        
        if self.func == "tan": # sec^2(x) = 1 + tan^2(x)
            output = BinOpExpression(
                    BinOpExpression(
                        NumExpression(1), "+", BinOpExpression(self.expr, "^", NumExpression(2))
                    ), '*', derivative_expr
                )
        
        if self.func == "arcsin": # (1-x^2)^(-0.5)
            output = BinOpExpression(
                BinOpExpression( 
                    NumExpression(1), '-', BinOpExpression(self.expr, "^", NumExpression(2))
                ), "^", NumExpression(-0.5)
            ) * derivative_expr
        
        if self.func == "arccos": # -(1-x^2)^(-0.5)
            output = BinOpExpression(
                BinOpExpression(
                    NumExpression(1), '-', BinOpExpression(self.expr, "^", NumExpression(2))
                ), "^", NumExpression(0.5)
            ) * NumExpression(-1) * derivative_expr
        
        if self.func == "arctan": # 1/(x^2+1)
            output = BinOpExpression(
                BinOpExpression(
                    BinOpExpression(self.expr, "^", NumExpression(2)), '+', NumExpression(1)
                ), "^", NumExpression(-1)
            ) * derivative_expr
        
        if self.func == 'exp':
            output = self * derivative_expr
        
        if self.func == 'sqrt':
            output = BinOpExpression(self, "^", NumExpression(-0.5)) * derivative_expr
        
        if self.func == 'abs':
            raise ValueError("Derivative of abs function is not supported.")
        output = output.canonicalize()
        output.domain = self.domain
        return output
    
    def directional_derivative(self, direction: dict) -> 'Expression':
        deriv = []
        for var in self.vars_:
            coef = NumExpression(direction.get(var, 0))  # 해당 변수의 방향 성분; 없으면 0으로 간주
            deriv.append(coef * self.derivative(var))
        output = UnionExpression(*deriv)
        output.domain = self.domain
        return output
        
    def is_constant(self) -> bool:
        return self.expr.is_constant()

    def is_integer(self) -> bool:
        return self.evaluate().is_integer()
    
    def __eq__(self, other: Expression) -> bool:
        if not isinstance(other, SingleVarFunction):
            return False
        return (self.func == other.func) and (self.expr == other.expr) and (self.domain == other.domain)
    
    def __hash__(self):
        return hash(str(self))
    
    def __str__(self):
        if self.func == 'abs':
            return f"|{str(self.expr)}|"
        return f"{self.func}({self.expr})"
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}SingleVarFunction({self.func}) (\n{self.expr._repr_tree(level + 2)}\n{'  ' * level})".replace("((", "(").replace("))", ")")


class MultiVarFunction(Expression):
    def __init__(self, func: str, *args: Expression):
        if func not in FUNCTIONS:
            raise ValueError(f"Invalid function: {func}")
        self.func = func
        self.args = args

        combined_vars = collect_var_names(self)
        merged_domain = {}
        for arg in self.args:
            if hasattr(arg, 'domain'):
                merged_domain = merge_domains(merged_domain, arg.domain)
        
        if self.func == 'log':
            base, value = args
            # base > 0, base != 1
            # value > 0
            if isinstance(base, VarExpression):
                merged_domain[base.name] = merged_domain[base.name].intersects(Interval(base.name, 0, Decimal("inf"), False, False))
                merged_domain[base.name].excluded_points.add(Decimal(1))
            elif isinstance(base, NumExpression):
                if base.value <= 0 or base.value == 1:
                    raise ValueError(f"Domain error: Log function - base value")
            if isinstance(value, VarExpression):
                merged_domain[value.name] = merged_domain[value.name].intersects(Interval(value.name, 0, Decimal("inf"), False, False))
            elif isinstance(value, NumExpression):
                if value.value <= 0:
                    raise ValueError(f"Domain error: Log function - argument value")

        super().__init__(vars_=combined_vars, domain=merged_domain)

    def evaluate(self, env: dict = None) -> Decimal:        
        if self.func == 'log':
            # argument에 변수가 있을 때 변수의 값이 정의된게 있으면 값을 넣어주고, 아니면 변수 째로 반환
            arg_values = [arg.evaluate(env).value if isinstance(arg.evaluate(env), NumExpression) else arg.evaluate(env) for arg in self.args]
            # 변수가 포함되지 않았을 경우에 함수 적용
            if all(isinstance(arg, Decimal) for arg in arg_values):
                if arg_values[0] <= 0:
                    raise ValueError(f"Base of logarithm must be positive, but {arg_values[0]} is given.")
                
                if arg_values[1] <= 0:
                    raise ValueError(f"Value of logarithm must be positive, but {arg_values[1]} is given.")
                
                return NumExpression( Decimal( str(FUNCTIONS[self.func](*arg_values[::-1])) ) )
            else:
                return self
        else:
            raise NotImplementedError
        
    def canonicalize(self):
        # 내부 인자 정리
        self.args = [simplify(arg.canonicalize()) for arg in self.args]
        return self
    
    def derivative(self, var: str) -> 'Expression':
        if self.func == 'log':
            base = self.args[0].canonicalize()
            value = self.args[1].canonicalize()
            base_derivative = base.derivative(var)
            value_derivative = value.derivative(var)

            # ((value_deriv/value) * ln(base)) - ((base_deriv/base) * ln(value)) / ln(base)^2
            output = (
                ((value_derivative / value) * SingleVarFunction('ln', base)) - ((base_derivative / base) * SingleVarFunction('ln', value))
            ) / (SingleVarFunction('ln', base) ^ NumExpression(2))
                
        else:
            raise NotImplementedError(f"Derivative not implemented for function {self.func}")
        
        output.domain = self.domain
        return output.canonicalize()

    def directional_derivative(self, direction: dict) -> 'Expression':
        deriv = []
        for var in self.vars_:
            coef = NumExpression(direction.get(var, 0))
            deriv.append(coef * self.derivative(var))
        output = UnionExpression(*deriv)
        output.domain = self.domain
        return output

    def is_constant(self) -> bool:
        return all(arg.is_constant() for arg in self.args)
        
    def is_integer(self) -> bool:
        return all(arg.is_integer() for arg in self.args)
    
    def __eq__(self, other: Expression) -> bool:
        if not isinstance(other, MultiVarFunction):
            return False
        return (self.func == other.func) and (self.args == other.args) and (self.domain == other.domain)
    
    def __hash__(self):
        return hash(str(self))
    
    def __str__(self):
        if self.func == 'log':
            return f"{self.func}_{{{self.args[0]}}}{self.args[1]}"
        else:
            return f"{self.func}({', '.join(str(arg) for arg in self.args)})"

    def _repr_tree(self, level=0):
        return f"{'  ' * level}MultiVarFunction({self.func}) (\n".replace("((", "(").replace("))", ")") + \
            f"\n{'  ' * (level+2)},\n".join(arg._repr_tree(level+2) for arg in self.args) + \
            f"\n{'  ' * level})"
        

class UnionExpression(Expression):
    """
    abs 의 미분값 계산 시 활용
    Gradient 계산 시 활용 (다변수 함수의 도함수를 모두 담아두기 위한...)
    """
    def __init__(self, *args: Expression):
        self.args = args
        domain = {}
        combined_vars = set()
        for arg in self.args:
            combined_vars |= collect_var_names(arg)
            domain = merge_domains(domain, arg.domain)

        super().__init__(vars_=combined_vars, domain=domain)

    def evaluate(self, env: dict = None) -> Decimal:
        result = UnionExpression(*[a.evaluate(env) for a in self.args])
        result.domain = self.domain
        return result
    
    def canonicalize(self):
        return UnionExpression(*[simplify(arg.canonicalize()) for arg in self.args])
    
    def derivative(self, var: str) -> 'Expression':
        output = UnionExpression(*[arg.derivative(var).canonicalize() for arg in self.args])
        output.domain = self.domain
        return output
    
    def directional_derivative(self, direction: dict) -> 'Expression':
        return UnionExpression(*[arg.directional_derivative(direction) for arg in self.args])
    
    def gradient(self) -> 'Expression':
        return UnionExpression(*[arg.gradient() for arg in self.args])
    
    def is_constant(self) -> bool:
        return all(arg.is_constant() for arg in self.args)
    
    def is_integer(self) -> bool:
        return all(arg.is_integer() for arg in self.args)
    
    def __eq__(self, other: Expression) -> bool:
        if not isinstance(other, UnionExpression):
            return False
        return all( sa == oa for sa, oa in zip(self.args, other.args) )

    def __hash__(self):
        return hash(str(self))
    
    def __str__(self):
        return f"( {', '.join(str(arg) for arg in self.args)} )"
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}UnionExpression(\n".replace("((", "(").replace("))", ")") + \
            f"\n{'  ' * (level+2)},\n".join(arg._repr_tree(level+2) for arg in self.args) + \
            f"\n{'  ' * level})"


################
# 정규화 관련 함수 #
################

def collect_var_names(expr: 'Expression'):
    # AST에서 변수 이름을 추출
    if isinstance(expr, VarExpression):
        return {expr.name}
    elif isinstance(expr, UnaryOpExpression):
        return collect_var_names(expr.expr)
    elif isinstance(expr, BinOpExpression):
        return collect_var_names(expr.left) | collect_var_names(expr.right)
    elif isinstance(expr, SingleVarFunction):
        return collect_var_names(expr.expr)
    elif isinstance(expr, MultiVarFunction):
        return set().union(*[collect_var_names(arg) for arg in expr.args])
    elif isinstance(expr, UnionExpression):
        return set().union(*[collect_var_names(arg) for arg in expr.args])
    return set()
    

def simplify(expr: Expression) -> Expression:
    if isinstance(expr, NumExpression) or isinstance(expr, VarExpression):
        return expr
    # 정규화 시 필요한 기능들
    log_expr = rewrite_log_powers(expr) # log(x^n) -> n*log(x)
    expanded_expr = distribute(log_expr) # 분배
    terms = collect_terms(expanded_expr) # 항 분해
    combined = combine_like_terms(terms) # 동류항 정리
    simplified = rebuild_expression(combined) # 항 재구성
    return simplified


def rewrite_log_powers(expr: Expression) -> Expression:
    if isinstance(expr, BinOpExpression):
        new_left = rewrite_log_powers(expr.left)
        new_right = rewrite_log_powers(expr.right)
        return BinOpExpression(new_left, expr.op, new_right)
    
    elif isinstance(expr, UnaryOpExpression):
        new_sub = rewrite_log_powers(expr.expr)
        return UnaryOpExpression(expr.op, new_sub)
    
    elif isinstance(expr, MultiVarFunction):
        if expr.func == 'log':
            base_expr = rewrite_log_powers(expr.args[0]) # 밑
            val_expr = rewrite_log_powers(expr.args[1]) # 진수

            if isinstance(val_expr, BinOpExpression) and val_expr.op == "^":
                if isinstance(val_expr.right, NumExpression):
                    n_exponent = val_expr.right.value
                    output = BinOpExpression(NumExpression(n_exponent), "*", MultiVarFunction('log', base_expr, val_expr.left))
                    output.domain = expr.domain
                    return output

        return expr # 지수가 숫자가 아닐 시, 그대로 유지 (rewrite_log_powers 처리 x)
    
    elif isinstance(expr, SingleVarFunction):
        expr.expr = rewrite_log_powers(expr.expr)
        return expr
    
    else:
        return expr
    

def distribute(expr: Expression) -> Expression:
    """
    1. BinOpExpression 중 (A + B) * C 형태로 나타날 때 분배법칙을 이용해
    A*C + B*C 로 바꾼다. 내부적으로 재귀적으로 전개한 다음 반환.
    2. 거듭제곱 전개
    (x + 1)^2 → (x + 1)*(x + 1) → ...
    """

    # 자식들을 먼저 전개
    if isinstance(expr, BinOpExpression):
        left = distribute(expr.left)
        right = distribute(expr.right)
        op = expr.op

        # (A + B) * C → A*C + B*C
        if op == '*':
            # 왼쪽이 덧셈이면 분배
            if isinstance(left, BinOpExpression) and left.op == '+':
                return BinOpExpression(
                    distribute(BinOpExpression(left.left, '*', right)),
                    '+',
                    distribute(BinOpExpression(left.right, '*', right))
                )

            # 오른쪽이 덧셈이면 분배
            if isinstance(right, BinOpExpression) and right.op == '+':
                return BinOpExpression(
                    distribute(BinOpExpression(left, '*', right.left)),
                    '+',
                    distribute(BinOpExpression(left, '*', right.right))
                )

        # ^ 등 거듭제곱 전개가 필요한 경우 (x + 1)^2 → (x + 1)*(x + 1) → ...
        # n 이 정수일 때만 전개
        if op == '^':
            if (isinstance(right, NumExpression) 
                and right.value == int(right.value)  # 정수 여부
                and right.value >= 2
                and isinstance(left, BinOpExpression)):
                exponent = int(right.value)
                base_expanded = distribute(left)
                # (base)^n = base * base^(n-1) 재귀
                result = base_expanded
                for _ in range(exponent - 1):
                    result = distribute(BinOpExpression(result, '*', base_expanded))
                return result

        # 분배할 게 없으면(더 이상 적용할 규칙이 없으면) 그대로 BinOpExpression 반환
        return BinOpExpression(left, op, right)

    elif isinstance(expr, UnaryOpExpression):
        expr_inside = distribute(expr.expr)
        return UnaryOpExpression(expr.op, expr_inside)

    elif isinstance(expr, SingleVarFunction) or isinstance(expr, MultiVarFunction):
        # 내부식도 분배
        new_args = []
        for arg in expr.args if isinstance(expr, MultiVarFunction) else [expr.expr]:
            new_args.append(distribute(arg))

        if isinstance(expr, MultiVarFunction):
            return MultiVarFunction(expr.func, *new_args)
        else:
            return SingleVarFunction(expr.func, new_args[0])
        
    elif isinstance(expr, UnionExpression):
        return UnionExpression(*[distribute(arg) for arg in expr.args])

    # NumExpression, VarExpression이면 그대로 반환
    return expr

def collect_terms(expr):
    """
    모든 항을 (계수, ({변수, 차수}, {변수2, 차수2})) 의 형태로 분해
    e.g. 3*x^2 + 2*x + 1 -> [(3, ({x: 2})), (2, ({x: 1})), (1, ())]
    e.g. 3*x*y^2 + 2*x*y + 1 -> [(3, ({x: 1, y: 2}), (2, ({x: 1, y: 1})), (1, ())]
    """
    terms = []

    def collect(node):
        if isinstance(node, NumExpression):
            terms.append((node.value, {}))
        
        elif isinstance(node, VarExpression):
            terms.append((1, {node.name: 1}))

        elif isinstance(node, UnaryOpExpression):
            if node.op == '-':
                sub_terms = collect_terms(node.expr)
                sub_terms = [(-c, p) for (c, p) in sub_terms]
                terms.extend(sub_terms)
            else:  # + 는 그냥 내부만
                collect(node.expr)

        elif isinstance(node, BinOpExpression):
            if node.op in ("+", "-"):
                collect(node.left)
                right_terms = collect_terms(node.right)
                if node.op == "-":
                    right_terms = [(-coeff, var_dict) for (coeff, var_dict) in right_terms]
                terms.extend(right_terms)

            elif node.op == "*":
                # 계수와 변수를 분리
                left_terms = collect_terms(node.left)
                right_terms = collect_terms(node.right)

                for (c1, var_dict_left) in left_terms:
                    for (c2, var_dict_right) in right_terms:
                        new_dict = var_dict_left.copy()
                        for var, exp in var_dict_right.items():
                            new_dict[var] = new_dict.get(var, 0) + exp
                        terms.append((c1 * c2, new_dict))
            elif node.op == "/":
                # A / B -> A * B^(-1)
                terms.append((1, {('div', node.left, node.right): 1}))

            elif node.op == "^":
                # node.left ^ node.right
                # coeff, {key: exponent}
                base_terms = collect_terms(node.left)
                exp = node.right
                if len(base_terms) != 1:
                    # raise ValueError(f"거듭제곱 내부 이슈 - distribute 처리가 안됨 {base_terms}")
                    # (x+1)/(x-1) 와 같은 경우, 분모^(-1) 형태로 변환되기 때문에 이 if 문이 호출됨.
                    # 이 경우 전체 분모를 그냥 하나의 항으로 처리
                    terms.append((1, {('pow', node): 1}))

                else:
                    (c, p) = base_terms[0]
                    if not isinstance(exp, NumExpression):
                        if isinstance(node.left, NumExpression):
                            terms.append((1, {('pow', node): 1}))
                        else:
                            # 지수가 숫자가 아닌 경우 (e.g. x^y)
                            terms.append((c, {('pow', node): 1}))
                    else:
                        n = exp.value
                        if not isinstance(c, Decimal):
                            c = Decimal(str(c))
                        new_coeff = c ** n # base 계수 ^ 지수
                        new_dict = {k: v*n for k,v in p.items()}
                        terms.append((new_coeff, new_dict))

        elif isinstance(node, SingleVarFunction):
            # sin(x), cos(x) 등의 함수 형태
            func_name = node.func
            inner_expr = node.expr
            terms.append((1, {('func', func_name, inner_expr): 1}))

        elif isinstance(node, MultiVarFunction):
            func_name = node.func
            key = ('func', func_name) + tuple(node.args)
            terms.append((1, {key: 1}))
        else:
            terms.append((1, {('unknown', node): 1}))
    
    collect(expr)
    return terms


def combine_like_terms(terms):
    # collect_terms로 뽑아낸 리스트에서 동일한 변수를 갖는 계수를 합산
    combined = defaultdict(Decimal)

    for coeff, var_dict in terms:
        key = tuple( sorted(var_dict.items(), key=lambda x: str(x[0])) )
        combined[key] += Decimal(coeff)

    result = []
    for key, coeff in combined.items():
        # 계수가 0이면 무시
        if coeff != 0:
            var_dict = dict(key)
            result.append((coeff, var_dict))

    return result


def sort_terms_for_rebuild(terms: list[tuple[Decimal, dict]]) -> list[tuple[Decimal, dict]]:
    def term_sort_key(term):
        coeff, var_dict = term
        # 변수 딕셔너리가 비었으면(상수) 맨 뒤로
        if not var_dict:
            return (999, 999, float(coeff))
        
        # 가장 앞의 변수 이름 (사전순)
        numerator_keys = []
        denominator_keys = []
        for k, exp in var_dict.items():
            if exp < 0:
                # k가 튜플이면, 대표값으로 str(k[0])을 사용하고, 아니라면 str(k)
                # tuple인 경우: ^ 혹은 / 연산에서 여러 개의 항이 하나의 항으로 묶인 경우.
                denominator_keys.append(str(k[0]) if isinstance(k, tuple) else str(k))
            else:
                numerator_keys.append(str(k[0]) if isinstance(k, tuple) else str(k))

        if numerator_keys:
            min_num = min(numerator_keys)
            selected_exp = None
            for k, exp in var_dict.items():
                key_name = k if isinstance(k, str) else k[0]
                if key_name == min_num:
                    selected_exp = exp
            if selected_exp is None:
                selected_exp = 0
            return (0, min_num, -float(selected_exp))
        else:
            # 분모가 있는 경우, (-1)승의 곱으로 표현됨. 가장 뒤에서 곱해주기.
            min_den = min(denominator_keys)
            return (1, "zz_" + min_den, 0) # 변수가 x, y, z인데, 사전순서로 정렬 시 그 뒤에 올 수 있도록 zz prefix

    return sorted(terms, key=term_sort_key)


def rebuild_expression(terms):
    # 항(단항) 하나를 expression으로 만드는 헬퍼 함수
    def build_term(coeff: Decimal, var_dict: dict) -> 'Expression':
        # 계수
        if coeff == 1 and var_dict:
            term_expr = None
        else:
            term_expr = NumExpression(coeff)

        # 변수/지수
        for var_key, exponent in sorted(var_dict.items(), key=lambda x: len(str(x[0]))): # pow(, ^-1)이 오면 뒤로 가도록
            try:
                # exponent가 0일 경우, 생략
                if exponent == 0:
                    continue
                
                # (1) 일반 변수 var_key = 'x'
                if isinstance(var_key, str):
                    base_expr = VarExpression(var_key)
                    factor_expr = base_expr if exponent == 1 else BinOpExpression(base_expr, "^", NumExpression(exponent))

                elif isinstance(var_key, tuple):
                    # functions
                    if var_key[0] == 'func':
                        func_name = var_key[1]
                        args = var_key[2:]
                        if len(args) == 1:
                            base_expr = SingleVarFunction(func_name, args[0])
                        else:
                            base_expr = MultiVarFunction(func_name, *args)
                        factor_expr = base_expr if exponent == 1 else BinOpExpression(base_expr, "^", NumExpression(exponent))
                    elif var_key[0] == 'pow':
                        base_expr = var_key[1]
                        factor_expr = base_expr if exponent == 1 else BinOpExpression(base_expr, "^", NumExpression(exponent))
    
                    elif var_key[0] == 'div':
                        # 새로 추가: division 연산 재구성
                        left_expr = var_key[1]
                        right_expr = var_key[2]
                        base_expr = BinOpExpression(left_expr, "/", right_expr)
                        factor_expr = base_expr if exponent == 1 else BinOpExpression(base_expr, "^", NumExpression(exponent))

                    else:
                        raise ValueError(f"Unknown variable key while rebuilding expression: {var_key}")
                else:
                    factor_expr = NumExpression(1)

                if term_expr is None:
                    term_expr = factor_expr
                else:                    
                    term_expr = BinOpExpression(term_expr, "*", factor_expr)
            
            except Exception as e:
                print(f"Error building term: {e}")
                import pdb; pdb.set_trace()
                raise e
            
        return term_expr if term_expr else NumExpression(coeff)
    
    sorted_terms = sort_terms_for_rebuild(terms)
    expr_sum = None
    for (coeff, var_dict) in sorted_terms:
        single_expr = build_term(coeff, var_dict)
        if expr_sum is None:
            expr_sum = single_expr
        else:
            expr_sum = BinOpExpression(expr_sum, '+', single_expr)
    return expr_sum if expr_sum else NumExpression(0)


def get_sort_key(expr: Expression) -> tuple:
    """
    1) 변수 이름 사전순 (x < y < z)
    2) 지수(내림차순) (x^(n+1) > x^n)
    3) 함수(Function)의 경우, 인자에 변수가 있으면 그 인자(변수) 우선 비교 -> 
       그 뒤 함수 이름 사전순
    4) BinOpExpression도 (op-rank, left_key, right_key) 식으로 재귀 비교
    """
    # (A) 숫자
    if isinstance(expr, NumExpression):
        # 예) (type_rank=0, 값) -> 숫자는 맨 앞쪽으로(오름차순) 놓게 됨.
        return (0, float(expr.value))

    # (B) 변수
    elif isinstance(expr, VarExpression):
        return (1, expr.name)

    # (C) 단항 함수: SingleVarFunction
    elif isinstance(expr, SingleVarFunction):
        # 인자가 변수가 있는지 확인하여, 인자 key를 먼저 두고, 그 다음 함수 이름
        # 예: (2, (인자키), func_name)
        inner_key = get_sort_key(expr.expr)
        return (2, inner_key, expr.func)

    # (D) 다항 함수: MultiVarFunction
    elif isinstance(expr, MultiVarFunction):
        # 인자 여러 개 -> 각 arg의 key를 튜플로
        # (2, (arg1_key), (arg2_key), ..., func_name)
        arg_keys = tuple(get_sort_key(a) for a in expr.args)
        return (2,) + arg_keys + (expr.func,)

    # (E) 단항 부정/단항 연산: UnaryOpExpression
    elif isinstance(expr, UnaryOpExpression):
        # 내부식 key만 반환
        # 예: -(x+1)
        sub_key = get_sort_key(expr.expr)
        return sub_key

    # (F) 2항 연산: BinOpExpression
    elif isinstance(expr, BinOpExpression):
        # op 별 rank: ^ < * < +, - < / 등
        op_priority = {'^': 1, '*': 2, '+': 3, '-': 3, '/': 4}
        rank_op = op_priority.get(expr.op, 99)

        left_key = get_sort_key(expr.left)
        right_key = get_sort_key(expr.right)

        if expr.op == '^':
            # right가 NumExpression이면 지수=int(...) 가능
            if isinstance(expr.right, NumExpression):
                e = expr.right.value
                # exponent 내림차순 정렬을 위해 -e
                return (rank_op, left_key, -float(e))
            else:
                # 지수가 변수나 함수인 경우
                return (rank_op, left_key) + get_sort_key(expr.right)
        
        else:
            # ^ 아니면 그냥 left_key, right_key 연결
            # 예: +, -, *, / => (rank_op, left_key, right_key)
            return (rank_op, left_key, right_key)

    else:
        # Unknown type
        return (999, str(expr))
    

######################
# 연속성, 미분가능성 확인 #
######################

def approximate_limit(expr: Expression, point: dict[str, Decimal], direction: str, tol=1e-10, max_iter=10) -> float:
    """
    주어진 점(point)에 대해, direction 방향('+' 또는 '-')으로 극한을 수치 근사합니다.
    초기 epsilon 값에서 시작해 반복적으로 10으로 나누며 f(x)를 평가하여 수렴하는지 확인합니다.
    """
    # 초기 epsilon 값 설정 (예: 0.001)
    epsilon = Decimal(1e-11)
    sign = -1 if direction == '-' else 1
    prev_value = None
    
    for _ in range(max_iter):
        new_point = {v: point[v] + sign * epsilon for v in point}
        try:
            val_expr = expr.evaluate(env=new_point)
            
            if isinstance(val_expr, NumExpression):
                curr_value = float(val_expr.value)
            else:
                # 숫자로 평가되지 않는 경우 수렴 판정 불가
                return None
        except Exception:
            return None
        
        if prev_value is not None and abs(curr_value - prev_value) >= tol:
            return None
        
        prev_value = curr_value
        epsilon /= 10  # epsilon 감소시켜 점점 더 가까운 값 평가
    
    return prev_value


def check_continuity_at(expr: Expression, point: dict[str, Decimal], tol=1e-8) -> bool:
    """
    수치 근사로 좌우 극한을 구한 후, 해당 점에서 함수값과 비교하여 연속성 여부를 확인
    
    1. 해당 변수의 정의역에 점이 포함되는지 검사
    2. 주어진 점에서의 함수값을 evaluate
    3. approximate_limit 함수를 통해 좌극한과 우극한을 계산
    4. 좌극한, 우극한, 함수값이 tol 내에서 일치하면 연속으로 판단
    """
    ##  다변수 함수일 때 각 변수에 대해 값을 지정받아야하고, 각각에 대해 연속성 확인. (모두 연속이어야 연속)
    if isinstance(expr, UnionExpression):
        for arg in expr.args:
            if not check_continuity_at(arg, point):
                return False
        return True

    variables = collect_var_names(expr)
    for var in variables:
    # 1. 정의역 검사
        if expr.domain and var in expr.domain:
            if not expr.domain[var].contains(point[var]):
                return False

    # 2. 주어진 점에서 함수값 평가
    try:
        val_expr = expr.evaluate(env=point)
        
        if isinstance(val_expr, NumExpression):
            val = float(val_expr.value)
        # elif isinstance(val_expr, list) and all(isinstance(v, NumExpression) for v in val_expr):
            # val = [float(v.value) for v in val_expr]
        else:
            return False
    except Exception:
        return False

    # 3. 좌극한, 우극한 근사 계산
    left_lim = approximate_limit(expr, point, direction='-')
    right_lim = approximate_limit(expr, point, direction='+')
    if left_lim is None or right_lim is None:
        return False

    # 4. 극한값과 함수값 비교
    if abs(left_lim - val) < tol and abs(right_lim - val) < tol:
        return True
    else:
        return False
    

def check_differentiability_at(expr: Expression, point: dict[str, Decimal], tol=1e-8) -> bool:
    """
    1. 연속성 확인 (정의역 검사 + 연속성 확인 기능)
    2. 도함수 구하기
    3. 도함수의 좌극한 우극한 비교 (도함수의 연속성 확인 / 혹은 좌미분계수 우미분계수 따로)
    """
    if not check_continuity_at(expr, point):
        return False
    
    # TODO: 2. 도함수 구하기
    for var in point:
        try:
            diff_expr = expr.derivative(var)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
            return False

        # TODO: 3. 도함수의 좌극한 우극한 비교
        left_lim = approximate_limit(diff_expr, point, direction='-')
        right_lim = approximate_limit(diff_expr, point, direction='+')
        if left_lim is None or right_lim is None:
            return False
    
    return True

def plot_graph(expr: Expression):
    # expr.vars_ 는 수식에 사용된 변수들의 집합입니다.
    variables = list(expr.vars_)
    n = len(variables)

    # 각 변수별로 도메인 정보를 활용하여 100개의 균등한 점을 생성합니다.
    samples = {}
    for var in variables:
        interval = expr.domain[var]
        start, end = interval.start, interval.end
        # 도메인이 무한대인 경우 적절한 구간 [-10, 10]으로 대체합니다.
        if start == -np.inf:
            start = -10
        if end == np.inf:
            end = 10
        samples[var] = np.linspace(start, end, 30)
    
    if n == 1:
        # 1차원 함수: 한 변수에 대해 선 그래프로 그립니다.
        var = variables[0]
        x_vals = samples[var]
        y_vals = [expr.evaluate({var: xi}).value for xi in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel(var)
        plt.ylabel(f"f({var})")
        plt.title("Plot of the Expression")
        plt.grid(True)
        plt.show()
    
    elif n == 2:
        # 2차원 함수: 두 변수에 대해 meshgrid로 격자를 만들고, 3D surface plot을 그립니다.
        X, Y = np.meshgrid(samples[variables[0]], samples[variables[1]])
        # meshgrid로 만든 각 점에 대해 수식을 평가합니다.
        points_list = []
        for idx in range(X.size):
            point = {variables[0]: X.flat[idx], variables[1]: Y.flat[idx]}
            points_list.append(point)
        values = np.array([expr.evaluate(point).value for point in points_list])
        Z = values.reshape(X.shape)

        from mpl_toolkits.mplot3d import Axes3D  # 3D plotting


        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel(f"f({variables[0]}, {variables[1]})")
        ax.set_title("Surface Plot of the Expression")
        plt.show()
    
    else:
        # 3개 이상의 변수가 있을 경우, 도메인을 전부 격자로 샘플링하기 어렵기 때문에
        # 100개의 무작위 점을 선택하여 함수값을 계산하고, 샘플 인덱스에 따른 값으로 플롯합니다.
        points_list = []
        values = []
        for _ in range(100):
            point = {}
            for var in variables:
                interval = expr.domain[var]
                start, end = interval.start, interval.end
                if start == -np.inf:
                    start = -10
                if end == np.inf:
                    end = 10
                point[var] = np.random.uniform(start, end)
            points_list.append(point)
            values.append(expr.evaluate(point).value)
        values = np.array(values)
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(values)), values, 'o-')
        plt.xlabel("Sample Index")
        plt.ylabel("Function Value")
        plt.title("Plot of the Expression on Sample Points")
        plt.grid(True)
        plt.show()
