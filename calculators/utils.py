from abc import ABC, abstractmethod
from decimal import Decimal

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
    else:
        raise ValueError(f"Invalid operator '{op}'")


class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

class Node(ABC):
    """
    ABC: abstract base class
    java 에서의 Object 클래스와 같은 역할. (모든 클래스의 최상위 클래스)
    e.g. Object 클래스의 메서드: id, to_string, hash_code, .. 
    
    """
    @abstractmethod
    def evaluate(self):
        # 계산 결과를 반환하는 메서드
        pass

class NumNode(Node):
    def __init__(self, value):
        self.value = Decimal(value)
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return self._repr_tree(0)
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}NumNode({self.value})"
    
    def evaluate(self):
        return self.value
    

class UnaryOpNode(Node):
    def __init__(self, op, node):
        self.op = op
        self.node = node
    
    def __str__(self): # (-3)
        return f"({self.op}{self.node})"
    
    def __repr__(self):
        return self._repr_tree(0)
    
    def _repr_tree(self, level=0): 
        return f"{'  ' * level}UnaryOpNode({self.op})\n{self.node._repr_tree(level + 2)}"
    
    def evaluate(self):
        val = self.node.evaluate()
        if self.op == '+':
            return val
        elif self.op == '-':
            return -val
        else:
            raise ValueError(f"Invalid operator '{self.op}'")


class BinOpNode(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    
    def __str__(self): # (3 + 2)
        return f"({self.left} {self.op} {self.right})"
    
    def __repr__(self):
        return self._repr_tree(0)
    
    def _repr_tree(self, level=0): 
        return (
            f"{'  ' * level}BinOpNode({self.op})\n"
            f"{self.left._repr_tree(level + 2)}\n"
            f"{self.right._repr_tree(level + 2)}"
        )
    
    def evaluate(self):
        left_val = self.left.evaluate()
        right_val = self.right.evaluate()
        return apply_operator(self.op, left_val, right_val)