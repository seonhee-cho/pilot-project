from abc import ABC, abstractmethod
from decimal import Decimal
import math

###################
# 변수, 상수, 함수 정의
###################
VARIABLES = {'x', 'y', 'z'}
OPERATORS = {'+', '-', '*', '/', '^', '(', ')'}
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
}


###################
# 함수 정의
################### 

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

def print_token(tokens):
    print("Token: ", end="")
    print(" / ".join(f"{token.value} \"{token.type}\"" for token in tokens))
    print()

###################
# 파서 클래스 정의
###################

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return f"{self.value}<{self.type}>"
    
    def __repr__(self):
        return f"{self.value}<{self.type}>"


class Node(ABC):
    """
    ABC: abstract base class
    java 에서의 Object 클래스와 같은 역할. (모든 클래스의 최상위 클래스)
    e.g. Object 클래스의 메서드: id, to_string, hash_code, .. 
    
    """
    @abstractmethod
    def evaluate(self, env=None):
        # 계산 결과를 반환하는 메서드
        pass
    
    def canonicalize(self):
        return self
    
    def __repr__(self):
        return self._repr_tree(0)
    

def get_sort_key(node: Node) -> tuple:
    """
    Node를 (rank, value) 형태의 튜플로 변환
    - FuncNode -> (0, func_name)
    - VarNode -> (1, var_name)
    - NumNode -> (2, num_value)
    - UnaryOpNode, BinOpNode -> (3, str(node))
    """
    if isinstance(node, FuncNode):
        return (0, node.func)
    elif isinstance(node, VarNode):
        return (1, node.name)
    elif isinstance(node, NumNode):
        return (2, node.value)
    else:
        return (3, str(node))

def compare_nodes(node1: Node, node2: Node) -> int:
    """
    -1: node1 < node2
     0: node1 == node2
     1: node1 > node2

    - 숫자 vs. 숫자 : 크기 비교
    - 숫자 vs. 변수 : 변수가 먼저
    - 변수 vs. 변수 : 사전순 비교
    - 함수 vs. 함수 : 함수 이름 사전순 비교
    - 함수 vs. 숫자/변수 : 함수가 먼저
    """
    k1, k2 = get_sort_key(node1), get_sort_key(node2)
    if k1 == k2:
        return 0
    elif k1 < k2:
        return -1
    else:
        return 1


class NumNode(Node):
    def __init__(self, value):
        self.value = Decimal(value)
    
    def evaluate(self, env=None):
        return self.value
    
    def canonicalize(self):
        return self
    
    def __str__(self):
        return str(self.value)
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}NumNode({self.value})"
    

class UnaryOpNode(Node):
    def __init__(self, op, node):
        self.op = op
        self.node = node
    
    def evaluate(self, env=None):
        val = self.node.evaluate()
        if self.op == '+':
            return val
        elif self.op == '-':
            return -val
        else:
            raise ValueError(f"Invalid operator '{self.op}'")
    
    def canonicalize(self):
        canonical_node = self.node.canonicalize()
        if self.op == '+':
            return canonical_node
        else: # '-'
            if isinstance(canonical_node, NumNode):
                return NumNode(-canonical_node.value)
            
            elif isinstance(canonical_node, UnaryOpNode):
                if canonical_node.op == '+':
                    return UnaryOpNode(self.op, canonical_node.node)
                else:
                    return canonical_node.node
            else:
                return UnaryOpNode(self.op, canonical_node)

    def __str__(self): # (-3)
        return f"({self.op}{self.node})"
    
    def _repr_tree(self, level=0): 
        return f"{'  ' * level}UnaryOpNode({self.op})\n{self.node._repr_tree(level + 2)}"


class BinOpNode(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    
    def evaluate(self, env=None):
        left_val = self.left.evaluate(env)
        right_val = self.right.evaluate(env)
        return apply_operator(self.op, left_val, right_val)
    
    def canonicalize(self):
        left_canonical = self.left.canonicalize()
        right_canonical = self.right.canonicalize()

        if self.op in ('+', '*'):
            # 교환법칙 성립 - 노드 정렬
            if compare_nodes(left_canonical, right_canonical) > 0:
                left_canonical, right_canonical = right_canonical, left_canonical

        return BinOpNode(left_canonical, self.op, right_canonical)
            
    
    def __str__(self): # (3 + 2)
        return f"({self.left} {self.op} {self.right})"
    
    def _repr_tree(self, level=0): 
        return (
            f"{'  ' * level}BinOpNode({self.op})\n"
            f"{self.left._repr_tree(level + 2)}\n"
            f"{self.right._repr_tree(level + 2)}"
        )

class VarNode(Node):
    def __init__(self, name):
        self.name = name
    
    def evaluate(self, env=None):
        raise NotImplementedError("Variable evaluation is not implemented.")
    
    def canonicalize(self):
        return self
    
    def __str__(self):
        return self.name
    
    def _repr_tree(self, level=0):
        return f"{'  ' * level}VarNode({self.name})"

class FuncNode(Node):
    SUPPORTED_FUNCTIONS = {
        "sin": (math.sin, 1),
        "cos": (math.cos, 1),
        "tan": (math.tan, 1),
        "exp": (math.exp, 1),
        "ln": (math.log, 1),
        "log": (math.log, (1, 2)),  # log(x) = log10(x) or log_b(x)
        "sqrt": (math.sqrt, 1)
    }
    def __init__(self, func, nodes):
        if func not in self.SUPPORTED_FUNCTIONS:
            raise ValueError(f"Unsupported function: {func}")
        self.func = func
        self.nodes = nodes if isinstance(nodes, list) else [nodes]
    
    def evaluate(self, env=None):
        arg_values = [arg.evaluate(env) for arg in self.nodes]
        func, param_count = self.SUPPORTED_FUNCTIONS[self.func]
        
        if len(arg_values) == 1:
            return Decimal(func(arg_values[0]))
        else:
            assert isinstance(param_count, tuple)
            return Decimal(func(*arg_values))

    def canonicalize(self):
        canonical_nodes = [arg.canonicalize() for arg in self.nodes]
        return FuncNode(self.func, canonical_nodes)
    
    def __str__(self):
        return f"{self.func}({', '.join(map(str, self.nodes))})"
    
    def _repr_tree(self, level=0):
        args_repr = "\n".join(arg._repr_tree(level + 2) for arg in self.nodes)
        return f"{'  ' * level}FuncNode({self.func})\n{args_repr}"
