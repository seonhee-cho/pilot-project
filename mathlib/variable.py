from decimal import Decimal

class SingleVariableFunction:
    def __init__(self, expression: str):
        self.expression = expression
    
    def evaluate(self, x: Decimal) -> Decimal:
        pass
    
    def derivative(self) -> 'SingleVariableFunction':
        pass
    

class MultiVariableFunction:
    def __init__(self, expression: str):
        self.expression = expression
    
    def evaluate(self, **kwargs) -> Decimal:
        pass
    
    def partial_derivative(self, var: str) -> 'SingleVariableFunction':
        # 편미분 expression 반환
        pass
    
    def gradient(self) -> list['SingleVariableFunction']:
        # 모든 변수에 대한 편미분 반환
        pass
    
    def directional_derivative(self, direction: list[Decimal]) -> Decimal:
        # 방향 미분 반환
        pass
    
    # is_continuous, is_differentiable, compose 등 구현 가능
