from decimal import Decimal
from abc import ABC, abstractmethod
from typing import Union, Optional

import sympy as sp
from utils import apply_operator, get_function_domain
from constants import *

class Interval:
    def __init__(
            self,
            start: Decimal,
            end: Decimal,
            closed_start: bool = True,
            closed_end: bool = True,
            excluded_points: Optional[set[Decimal]] = None
            # TODO: excluded_subintervals 추가 고려
        ):
        # 함수의 정의역 (혹은 치역) 을 정의하기 위한 객체
        # all_real() 함수를 통해 실수 전체 집합을 간단히 표현 가능
        # excluded_points 변수를 통해 제외할 점들을 지정할 수 있음

        if start > end:
            raise ValueError("Start must be less than end")
        
        self.start = start
        self.end = end
        self.closed_start = closed_start
        self.closed_end = closed_end
        self.excluded_points = excluded_points if excluded_points else set()
        
    @classmethod
    def all_real(cls):
        return cls(-math.inf, math.inf, closed_start=False, closed_end=False)
    
    @classmethod
    def parse(cls, s: str) -> 'Interval':
        s = s.strip().replace(" ", "")
        assert s.startswith("[") or s.startswith("(")
        assert s.endswith("]") or s.endswith(")")
        
        sbracket = s[0]
        ebracket = s[-1]
        
        closed_start = True if sbracket == "[" else False
        closed_end = True if ebracket == "]" else False
        
        start, end = s[1:-1].split(',')
        start = Decimal(start)
        end = Decimal(end)

        return cls(start, end, closed_start, closed_end)
    
    @classmethod
    def from_sympy_conditions(cls, conditions):
        start = -Decimal('Infinity')
        end = Decimal('Infinity')
        closed_start = True
        closed_end = True
        excluded_points = set()

        result = {}

        for condition, variable in conditions:
            if not isinstance(variable, sp.Symbol):
                continue
            if isinstance(condition, sp.Gt):
                if condition.lhs == variable:
                    start = max(start, Decimal(float(condition.rhs.evalf())))
                    closed_start = False
            elif isinstance(condition, sp.Ge):
                if condition.lhs == variable:
                    start = max(start, Decimal(float(condition.rhs.evalf())))
                    closed_start = True
            elif isinstance(condition, sp.Lt):
                if condition.lhs == variable:
                    end = min(end, Decimal(float(condition.rhs.evalf())))
                    closed_end = False
            elif isinstance(condition, sp.Le):
                if condition.lhs == variable:
                    end = min(end, Decimal(float(condition.rhs.evalf())))
                    closed_end = True
            elif isinstance(condition, sp.Ne):
                excluded_points.add(Decimal(float(condition.rhs.evalf())))
            
            result[str(variable)] = cls(start, end, closed_start, closed_end, excluded_points=excluded_points)

        return result

    def __repr__(self):
        sbracket = "[" if self.closed_start else "("
        ebracket = "]" if self.closed_end else ")"
        main_str = f"{sbracket}{self.start}, {self.end}{ebracket}"

        if self.excluded_points:
            ex_pts_str = f"excluded_points={self.excluded_points}"
            return f"Interval({main_str}, {ex_pts_str})"
        
        return f"Interval({main_str})"
    
    def contains(self, value: Decimal) -> bool:
        # 1) 제외할 점이 있는 경우
        if self.excluded_points and value in self.excluded_points:
            return False
        
        # 2) 정의된 범위 내에 들어가는지
        if value < self.start or value > self.end:
            return False
        elif (value == self.start and not self.closed_start) or (value == self.end and not self.closed_end):
            return False
        else:
            return True

    def intersects(self, other: 'Interval') -> 'Interval':
        # 구간 간의 교차범위 구하기
        new_start = max(self.start, other.start)
        new_end = min(self.end, other.end)
        
        if new_start > new_end:
            return None
            
        # 시작점이 같을 때는 둘 다 닫힌 구간이어야 포함
        new_closed_start = (self.closed_start and other.closed_start) if new_start == self.start == other.start \
            else (self.closed_start if new_start == self.start else other.closed_start)
            
        # 끝점이 같을 때는 둘 다 닫힌 구간이어야 포함  
        new_closed_end = (self.closed_end and other.closed_end) if new_end == self.end == other.end \
            else (self.closed_end if new_end == self.end else other.closed_end)
        
        new_excluded_points = self.excluded_points | other.excluded_points
            
        return Interval(new_start, new_end, new_closed_start, new_closed_end, new_excluded_points)
    
    def issubset(self, other: 'Interval') -> bool:
        # 구간 간의 포함 관계 확인. self가 other 에 포함되는지 확인
        # other : -inf, inf # self : -1, 1
        if self.start >= self.end:
            raise ValueError("Interval is not valid: start must be less than end")
        
        if (other.start < self.start < other.end) or (other.start == self.start and other.closed_start):
            if (other.start < self.end < other.end) or (other.end == self.end and other.closed_end):
                for ex_pt in self.excluded_points:
                    if other.contains(ex_pt):
                        return False
                return True
        
        return False
    
def merge_domains(domainA: dict[str, Interval], domainB: dict[str, Interval]) -> dict[str, Interval]:
    merged = {}
    all_vars = set(domainA.keys()) | set(domainB.keys())

    for var in all_vars:
        intervalA = domainA.get(var, Interval.all_real())
        intervalB = domainB.get(var, Interval.all_real())

        intersected = intervalA.intersects(intervalB)
        if intersected is None:
            return {}
        else:
            merged[var] = intersected
            
    return merged


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
    return set()


class Expression(ABC):
    """
    수학식 (함수, 상수, 변수, 연산 등)을 추상적으로 표현하는 클래스
    단변수/다변수 함수 표현 가능.
    """
    def __init__(self,
                  vars_: set[str],
                  domain: dict[str, Interval] = None,
                #   range_: dict[str, Interval] = None
                ):
        # vars, range는 이미 표준/내장함수가 있음
        self.vars_ = vars_ if vars_ else set()
        self.domain = domain if domain else {}
        self._intrinsic_domain = self.domain
        # self.range_ = range_

    def update_domain(self, var, new_domain: dict[str, Interval]):
        # new_domain이 함수의 원 정의역인 self._intrinsic_domain 에 포함되는지 확인
        # 포함될 시 self.domain 을 new_domain 으로 업데이트
        # 포함되지 않을 시 오류 발생
        if var not in self._intrinsic_domain:
            self.domain[var] = new_domain
        else:
            if not new_domain.issubset(self._intrinsic_domain[var]):
                # import pdb; pdb.set_trace()
                raise ValueError(f"Interval {new_domain} is not a subset of the intrinsic domain")            
            else:
                for points in self._intrinsic_domain[var].excluded_points:
                    if new_domain.contains(points):
                        raise ValueError(f"Interval {new_domain} contains excluded points")
                    
                self.domain[var] = new_domain

    @abstractmethod
    def evaluate(self, env: dict = None) -> 'Expression':
        pass
    
    def __repr__(self):
        return self._repr_tree(0)
    
    def _domain_str(self):
        if not self.domain:
            return ""
        
        pairs = []
        for var, interval in self.domain.items():
            pairs.append(f"{var}: {interval}")
        return "\n".join(pairs)
    
    
    def canonicalize(self):
        pass

    def derivative(self, var: str) -> 'Expression':
        pass

    def is_continuous_at(self, var: str, point: Decimal) -> bool:
        if self.domain and self.domain.get(var) is not None:
            if self.domain[var].contains(point):
                return True
        return False

    def is_differentiable_at(self, var: str, point: Decimal) -> bool:
        if not self.is_continuous_at(var, point):
            return False
        return True

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

    @abstractmethod
    def to_sympy(self) -> sp.Expr:
        pass


class NumExpression(Expression):
    def __init__(self, value: Decimal, name: str = ""):
        super().__init__(
            vars_=set(),
            domain={},
            # range_=Interval(value, value, closed_start=True, closed_end=True)
        )
        self.value = value
        self.name = name

    def evaluate(self, env: dict = None) -> 'Expression':
        return self

    def derivative(self, var: str) -> 'Expression':
        return NumExpression(0)
    
    def is_continuous_at(self, var: str) -> bool:
        return True
    
    def is_differentiable_at(self, var: str) -> bool:
        return True
    
    def canonicalize(self):
        return self
    
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
        # TODO
        self.op = op
        self.expr = expr

        # 메타데이터 초기화
        combined_vars = expr.vars_
        merged_domains = dict(expr.domain) # shallow copy
        super().__init__(vars_=combined_vars, domain=merged_domains)

    def evaluate(self, env: dict = None) -> 'Expression':
        val = self.expr.evaluate(env)
        if isinstance(val, NumExpression):
            if self.op == '+':
                return NumExpression( Decimal(val) )
            elif self.op == '-':
                return NumExpression( -Decimal(val) )
            else:
                raise ValueError(f"Invalid operator: {self.op}")
        else:
            return UnaryOpExpression(self.op, val)
        
    def derivative(self, var: str) -> 'Expression':
        if self.op == '+':
            return self.expr.derivative(var)
        elif self.op == '-':
            return -self.expr.derivative(var)
        else:
            raise ValueError(f"Invalid operator: {self.op}")
        
    def is_continuous_at(self, var: str, point: Decimal) -> bool:
        return self.expr.is_continuous_at(var, point)
    
    def is_differentiable_at(self, var: str, point: Decimal) -> bool: ## 확인
        return self.expr.is_differentiable_at(var, point)
        
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
    
    def to_sympy(self) -> sp.Expr:
        expr_sympy = self.expr.to_sympy()
        if self.op == '+':
            return expr_sympy
        elif self.op == '-':
            return -expr_sympy
        else:
            raise ValueError(f"Invalid operator: {self.op}")


class BinOpExpression(Expression):
    def __init__(self, left: Expression, op: str, right: Expression):
        # 메타데이터 초기화
        combined_vars = left.vars_ | right.vars_

        merged_domains = merge_domains(left.domain, right.domain)
        if op == '/':
            # TODO: 분모가 0이 되는 부분 처리
            pass

        if op == '^' or '**':
            # TODO: self.left가 0 이상? 초과? 여야 함.
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
            return BinOpExpression(left_val, self.op, right_val).canonicalize()
    
    def differentiate(self, var: str = "", direction: list[Decimal]=None) -> 'Expression':
        """ (x)
        2 + 3 -> 0 + 0 => 0
        x + 3 -> 1 + 0 => 1
        x^2 + 3 -> 2*x^(2-1) + 0 => 2x

        INPUT : x^2 - 2x + 1 = 0
        OUTPUT : (x-1)^2 = 0 => x = 1
        변수의 값을 계산하는 기능 (구현 x)

        sin(x)/cos(x) -> canonicalize를 먼저 하고 미분
        --
        parsing > canonicalize > 그 다음작업
        """
        canonical_expr = self.canonicalize()
        if var:
            return canonical_expr.derivative(var).canonicalize()
        elif direction:
            return [canonical_expr.derivative(var).canonicalize() for var in direction]
        else: # gradient 반환
            return canonical_expr.gradient().canonicalize()
        

    def is_continuous_at(self, var: str, point: Decimal) -> bool:
        try:
            # evaluate 시 오류가 발생하지 않으면 연속임
            _ = self.evaluate(env={var: point})
            # self.left.is_continuous_at(var, point) and self.right.is_continuous_at(var, point)
            return True
        
        except:
            return False
    
    def is_differentiable_at(self, var: str, point: Decimal) -> bool:
        return self.left.is_differentiable_at(var, point) and self.right.is_differentiable_at(var, point)
    
    
    def derivative(self, var: str) -> 'Expression':
        left_deriv = self.left.derivative(var)
        right_deriv = self.right.derivative(var)

        if self.op in ('+', '-'):
            return BinOpExpression(left_deriv, self.op, right_deriv)
                
        elif self.op == '*':
            return left_deriv * self.right + self.left * right_deriv
        
        elif self.op == '/':
            # (u / v)' = (u'v - uv') / (v^2)
            numerator = left_deriv * self.right - self.left * right_deriv
            denominator = self.right ** NumExpression(2)
            return BinOpExpression(numerator, '/', denominator)
        
        elif self.op == '^':
            if isinstance(self.left, VarExpression) and self.left.name == var and isinstance(self.right, NumExpression):
                return self.right * (self.left ** NumExpression(self.right.value - 1))
            
            return NumExpression(0)
        
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
        
        # 원소 정렬
        if self.op in ('+', '*'):
            if compare_expressions(self.left, self.right) > 0:
                self.left, self.right = self.right, self.left
        
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

    def to_sympy(self) -> sp.Expr:
        left_sympy = self.left.to_sympy()
        right_sympy = self.right.to_sympy()
        return apply_operator(self.op, left_sympy, right_sympy)


class VarExpression(Expression):
    def __init__(self, name: str, interval: Interval = None):
        if name not in VARIABLES:
            raise ValueError(f"Invalid variable: {name}")
        domain_dict = {name: interval if interval else Interval.all_real()}
        super().__init__(
            vars_={name},
            domain=domain_dict,
        )
        self.name = name

    def evaluate(self, env: dict = None) -> Decimal:
        if env is None or self.name not in env or isinstance(env[self.name], sp.Symbol):
            return self
        
        return NumExpression( Decimal( env[self.name] ) )
    
    def derivative(self, var: str) -> 'Expression':
        if self.name == var:
            return NumExpression(1)
        else:
            return NumExpression(0)
        
    def is_continuous_at(self, var: str, point: Decimal) -> bool:
        return True
    
    def is_differentiable_at(self, var: str, point: Decimal) -> bool:
        return True

    def canonicalize(self):
        return self
    
    def __str__(self):
        return self.name

    def _repr_tree(self, level=0):
        return f"{'  ' * level}VarExpression({self.name})"
    
    def to_sympy(self) -> sp.Expr:
        return sp.symbols(self.name)


class SingleVarFunction(Expression):
    def __init__(self, func: str, expr: Expression):
        # 삼각함수: sin, cos, tan  /  역삼각함수: asin, acos, atan
        # 지수함수 : exp  /  로그함수 : log, ln  /  제곱근 : sqrt
        # 절대값 : abs  /  미분 : diff, grad
        if func.replace("'", "") not in FUNCTIONS:
            raise ValueError(f"Invalid function: {func}")
        self.func = func
        self.expr = expr

        # 메타 데이터 초기화 - 정의역 초기화 포함. 정의역 지정 시 이후에 변경
        combined_vars = expr.vars_

        if len(combined_vars) > 1:
            print("DEBUG FOR SINGLE VAR FUNCTION - MULTIPLE VARIABLES")
            import pdb; pdb.set_trace()


        merged_domain = dict(expr.domain)
        function_domain = Interval.from_sympy_conditions(get_function_domain(func, expr.to_sympy()))
        all_vars = set(merged_domain.keys()) | set(function_domain.keys())
        for _v in all_vars:
            if _v in merged_domain and _v in function_domain:
                merged_domain[_v] = merged_domain[_v].intersects(function_domain[_v])
            elif _v in function_domain:
                merged_domain[_v] = function_domain[_v]
        print(merged_domain)
        super().__init__(vars_=expr.vars_, domain=merged_domain)


    def evaluate(self, env: dict = None) -> Union[Decimal, 'Expression', None]:
        # import pdb; pdb.set_trace()
        if self.func == 'diff':
            return self.expr.derivative(self.expr.name)
        
        elif isinstance(self.expr, VarExpression) and env and isinstance(env[self.expr.name], sp.Symbol):
            return self
        
        ## 함수 내부에 값이 아니라 또 다른 함수/연산자 등이 오는 경우
        val = self.expr.evaluate(env)

        if not isinstance(val, NumExpression):
            return SingleVarFunction(self.func, val)
        
        return NumExpression( Decimal(FUNCTIONS[self.func](val.value)) )
    

    def derivative(self, var: str) -> 'Expression':
        derivative_expr = self.expr.derivative(var)

        if isinstance(derivative_expr, SingleVarFunction) or isinstance(derivative_expr, MultiVarFunction):
            # 합성함수: 체인 룰 적용
            outer_derivative = SingleVarFunction(self.func + "'", derivative_expr)
            if outer_derivative is None:
                raise ValueError(f"Derivative not defined for function {self.func}")
            return outer_derivative * derivative_expr
        
        elif isinstance(derivative_expr, NumExpression):
            return NumExpression(0)
        
        else:
            return SingleVarFunction(self.func, derivative_expr)
        
    def is_continuous_at(self, var: str, point: Decimal) -> bool:
        if self.func == 'abs':
            try:
                val = self.expr.evaluate(env={var: point})
                if val.value == 0:
                    return False
            except:
                return False
        return self.expr.is_continuous_at(var, point)
    
    def is_differentiable_at(self, var: str, point: Decimal) -> bool:
        return self.expr.is_differentiable_at(var, point)
        
    def canonicalize(self):
        canonical_expr = self.expr.canonicalize()
        return SingleVarFunction(self.func, canonical_expr)
    
    def __str__(self):
        if self.func == 'abs':
            return f"|{str(self.expr).strip('()')}|"
        return f"{self.func}({self.expr})".replace("((", "(").replace("))", ")")
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}SingleVarFunction({self.func}) (\n{self.expr._repr_tree(level + 2)}\n{'  ' * level})".replace("((", "(").replace("))", ")")
    
    def to_sympy(self) -> sp.Expr:
        expr_sympy = self.expr.to_sympy()
        if self.func == 'sin':
            return sp.sin(expr_sympy)
        elif self.func == 'cos':
            return sp.cos(expr_sympy)
        elif self.func == 'tan':
            return sp.tan(expr_sympy)
        elif self.func == 'exp':
            return sp.exp(expr_sympy)
        elif self.func == 'log':
            return sp.log(expr_sympy)
        elif self.func == 'sqrt':
            return sp.sqrt(expr_sympy)
        elif self.func == 'abs':
            return sp.Abs(expr_sympy)
        else:
            raise ValueError(f"Invalid function: {self.func}")


class MultiVarFunction(Expression):
    def __init__(self, func: str, *args: Expression):
        if func not in FUNCTIONS:
            raise ValueError(f"Invalid function: {func}")
        self.func = func
        self.args = args

        # 메타 데이터 초기화
        """
        arg : Union[NumExpression, VarExpression, FuncExpression, ..]
        """
        combined_vars = collect_var_names(self)
        try:
            merged_domain = Interval.from_sympy_conditions(get_function_domain(self.func, self.to_sympy()))
        except:
            print("FAILED to get domain interval from sympy")
            import pdb; pdb.set_trace()
        for arg in args:
            if isinstance(arg, Expression):
                merged_domain = merge_domains(merged_domain, arg.domain)

        print(merged_domain)
        super().__init__(vars_=combined_vars, domain=merged_domain)


    def evaluate(self, env: dict = None) -> Decimal:
        if self.func == 'diff':
            _func = self.args[0] # BinOpExpression(x^2 + y^2)
            _var = self.args[1:] # [VarExpression(x)]
            if len(_var) == 1:
                return _func.differentiate(var=_var[0].name).evaluate(env)
            elif len(_var) > 1:
                return _func.differentiate(direction=[arg.name for arg in _var]).evaluate(env)
            
        if self.func == 'grad':
            # grad 함수의 경우 재귀적으로 미분
            # 1. 모든 변수에 대해 편미분
            # 2. 모든 변수에 대해 편미분한 결과를 Gradient 형태로 반환
            pass
        
        if self.func == 'log':
            # argument에 변수가 있을 때 변수의 값이 정의된게 있으면 값을 넣어주고, 아니면 변수 째로 반환
            arg_values = [arg.evaluate(env).value if isinstance(arg.evaluate(env), NumExpression) else arg.evaluate(env) for arg in self.args]
            # 변수가 포함되지 않았을 경우에 함수 적용
            if all(isinstance(arg, Decimal) for arg in arg_values):
                if arg_values[0] <= 0:
                    raise ValueError(f"Base of logarithm must be positive, but {arg_values[0]} is given.")
                
                if arg_values[1] <= 0:
                    raise ValueError(f"Value of logarithm must be positive, but {arg_values[1]} is given.")
                
                return NumExpression( Decimal(FUNCTIONS[self.func](*arg_values)) )
            else:
                return self
            
        else:
            raise NotImplementedError
    
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
        if self.func == 'log':
            base, value = self.args
            return NumExpression(1) / (value * MultiVarFunction('log', NumExpression(MATH_CONSTANTS['e']), base))
        
        elif self.func == 'diff':
            return MultiVarFunction(self.func, *[arg.derivative(var) for arg in self.args])
        
        else:
            raise NotImplementedError(f"Derivative not implemented for function {self.func}")
        

    def is_continuous_at(self, var: str, point: Decimal) -> bool:
        # diff, log, grad
        ## diff, grad에서 호출될 때
            # args[0] : func, args[1:] : args
        # args[0].is_continuous_at(var, point) ??
        func = self.args[0]
        args = self.args[1:]
        if func.func == 'diff':
            return args[0].is_continuous_at(var, point)
        
        elif func.func == 'log':
            return args[0].is_continuous_at(var, point) and args[1].is_continuous_at(var, point)
        elif func.func == 'grad':
            pass
        


    def is_differentiable_at(self, var: str, point: Decimal) -> bool:
        return self.args[0].is_differentiable_at(var, point) and self.args[1].is_differentiable_at(var, point)
    

    
    def canonicalize(self):
        return MultiVarFunction(self.func, *[arg.canonicalize() for arg in self.args])

    def __str__(self):
        return f"{self.func}({', '.join(str(arg) for arg in self.args)})"

    def _repr_tree(self, level=0):
        return f"{'  ' * level}MultiVarFunction({self.func}) (\n".replace("((", "(").replace("))", ")") + \
            f"\n{'  ' * (level+2)},\n".join(arg._repr_tree(level+2) for arg in self.args) + \
            f"\n{'  ' * level})"
        
    def to_sympy(self) -> sp.Expr:
        args_sympy = [arg.to_sympy() for arg in self.args]
        if self.func in FUNCTIONS:
            func = FUNCTIONS[self.func]
            if callable(func):
                return func(*args_sympy)
            else:
                raise ValueError(f"Function {self.func} is not callable.")
        else:
            raise NotImplementedError(f"SymPy conversion not implemented for function {self.func}")


def get_sort_key(expr: Expression) -> tuple:
    """
    Node를 (rank, value) 형태의 튜플로 변환
    - FuncNode -> (0, func_name)
    - VarNode -> (1, var_name)
    - NumNode -> (3, num_value)
    - UnaryOpNode, BinOpNode -> (2, str(node))
    """
    def get_degree(expr: Expression) -> int:
        if isinstance(expr, VarExpression):
            return 1
        elif isinstance(expr, BinOpExpression) and expr.op == '^' and isinstance(expr.right, NumExpression):
            return int(expr.right.value)
        return 0

    degree = get_degree(expr)
    
    if isinstance(expr, SingleVarFunction):
        return (0, expr.func)
    elif isinstance(expr, MultiVarFunction):
        return (0, expr.func)
    elif isinstance(expr, VarExpression):
        return (1, expr.name, degree)
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