import sympy as sp

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


class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return f"Token({self.value}, type:{self.value})"
    
    def __repr__(self):
        return f"Token({self.value}, type:{self.value})"
    

def get_function_domain(function: str, expr: sp.Expr):
    if function == 'log':
        ## log( argument, base ) 형태로 들어옴
        # 1. 수식에서 로그의 피연산자와 밑을 추출
        argument = expr.args[1].args[0]      # 로그의 밑 (sin(x^2.7))
        base = expr.args[0].args[0].args[0]  # 로그의 피연산자/진수 (y)
        
        # 2. 정의역 조건 생성
        conditions = []
        
        # 조건 1: 로그의 피연산자 > 0
        conditions.append((sp.Gt(argument, 0), argument))
        
        # 조건 2: 로그의 밑 > 0 및 밑 ≠ 1
        conditions.append((sp.Gt(base, 0), base))
        conditions.append((sp.Ne(base, 1), base))

    elif function in ['arcsin', 'arccos']:
        argument = expr.args[0]
        conditions = [
            (sp.Ge(argument, -1), argument),
            (sp.Le(argument, 1), argument)
        ]

    elif function in ['sqrt']:
        argument = expr.args[0]
        conditions = [(sp.Ge(argument, 0), argument)]

    else:
        conditions = [(True, expr.args[0])]

    return conditions