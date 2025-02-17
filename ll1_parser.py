import re
import math
import sympy
from collections import deque
from utils import *

class Lexer:
    TOKEN_REGEX = re.compile(r'\s*(\d+\.\d+|\d+|[()+\-*/^]|[a-zA-Z]+)') # 실수 | 정수 | 상수 | 연산자

    @staticmethod
    def tokenize(expression: str, verbose: bool = False) -> deque[Token]:
        tokens = []
        position = 0

        while position < len(expression):
            match = Lexer.TOKEN_REGEX.match(expression, position)
            if match:
                token_value = match.group(0).strip()
                if token_value:
                    # 기본 연산자
                    if token_value in OPERATORS:
                        tokens.append(Token(token_value, token_value))
                    # 주요 수학 상수
                    elif token_value in MATH_CONSTANTS:
                        tokens.append(Token('const', token_value))
                    # 수학 함수
                    elif token_value in FUNCTIONS:
                        tokens.append(Token('func', token_value))
                    # 숫자
                    elif re.fullmatch(r'\d+\.\d+|\d+', token_value):  # 숫자 검증 강화 (소수점 두개 방지)
                        tokens.append(Token('num', token_value))
                    # 변수
                    elif token_value.isalpha():
                        tokens.append(Token('var', token_value))
                    else:
                        raise ValueError(f"Invalid token: {token_value}")

                position = match.end()
            else:
                raise ValueError(f"Invalid character at position {position}: '{expression[position]}'")
        
        if verbose:
            print_token(tokens)
        return deque(tokens)
        

class Parser:
    def __init__(self):
        self.tokens = None
        self.current_token = None

    
    def consume(self, expected: str):
        if self.current_token and self.current_token.type == expected:
            self.current_token = self.tokens.popleft() if self.tokens else None
        else:
            raise ValueError(f"Unexpected token: {self.current_token} (expected: {expected})")
        

    def parse(self) -> Node:
        self.tokens = deque(self.tokens)
        self.current_token = self.tokens.popleft() if self.tokens else None
        return self.parse_expression()
    
    def parse_expression(self) -> Node:
        # E (expression, 다항식) 문법 규칙: E -> T ( (+ T) | (- T) )*
        node = self.parse_term()
        while self.current_token and self.current_token.type in ('+', '-'):
            op = self.current_token.value
            self.consume(op)
            node = BinOpNode(node, op, self.parse_term())
        return node
    
    def parse_term(self) -> Node:
        # T (Term, 단항식) 문법 규칙: T -> F ( (* F) | (/ F) )*
        node = self.parse_power()
        while self.current_token and self.current_token.type in ('*', '/'):
            op = self.current_token.value
            self.consume(op)
            node = BinOpNode(node, op, self.parse_power())
        return node
    
    def parse_power(self) -> Node:
        # P (Power, 거듭제곱) 문법 규칙: P -> F (^ F)
        node = self.parse_factor()
        while self.current_token and self.current_token.type == '^':
            op = self.current_token.value
            self.consume(op)
            node = BinOpNode(node, op, self.parse_factor())
        return node
        
    
    def parse_factor(self) -> Node:
        # F (Factor, 인자) 문법 규칙: F -> num | (E) | (+ F) | (- F)
        if self.current_token is None:  
            raise ValueError("Unexpected end of input.")
        
        if self.current_token.type == 'num':
            val = self.current_token.value
            self.consume('num')
            return NumNode(val)
        
        elif self.current_token.type == 'const':
            val = self.current_token.value
            self.consume('const')
            return NumNode(MATH_CONSTANTS[val])
        
        elif self.current_token.type == '+':
            self.consume('+')
            sub_node = self.parse_factor()
            return UnaryOpNode('+', sub_node)
        
        elif self.current_token.type == '-':
            self.consume('-')
            sub_node = self.parse_factor()
            return UnaryOpNode('-', sub_node)
        
        elif self.current_token.type == '(':
            self.consume('(')
            node = self.parse_expression()
            self.consume(')')
            return node
        
        elif self.current_token.type == 'func':
            func_name = self.current_token.value
            self.consume('func')
            self.consume('(')
            
            arguments = [self.parse_expression()]  # 첫 번째 인자 파싱
            # 다변수 함수의 경우 추가 인자 파싱
            while self.current_token and self.current_token.value == ',':
                self.consume(',')
                arguments.append(self.parse_expression())
            
            self.consume(')')

            return FuncNode(func_name, arguments)
        
        elif self.current_token.type == 'var':
            var_name = self.current_token.value
            self.consume('var')
            return VarNode(var_name)
        
        else:
            raise ValueError(f"Unexpected token: {self.current_token.value}")


class Evaluator:
    @staticmethod
    def evaluate(ast: Node, env: dict = None, verbose: bool = False) -> Decimal:
        # 결과 계산 뿐 아니라 계산 과정을 출력하는 기능도 추가.
        if verbose:
            print("\n## Equation (str):")
            print(str(ast)[1:-1]) # 바깥 괄호 제거
            print("\n## AST (repr):")
            print(ast.__repr__())
            print("\n## Result:", ast.evaluate(env))

        return ast.evaluate()
    
def calculate_expression(expression: str, verbose: bool = False) -> Decimal:
    tokens = Lexer.tokenize(expression, verbose)
    parser = Parser()
    parser.tokens = tokens
    ast = parser.parse()
    return Evaluator.evaluate(ast, verbose), ast

def main():
    while True:
        try:
            expression = input("Enter expression (or 'q' to quit): ")
            if expression.lower() == 'q':
                break
            
            result, ast = calculate_expression(expression, verbose=True)
            print("Result:", result)
        except Exception as e:
            print("Error:", e)

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
        # 교환법칙 (숫자+숫자)
        "2 + 3",
        "3 + 2",  
        # 교환법칙 (변수+함수)
        # "x + sin(y)",
        # "sin(y) + x",
        # 중첩 단항 연산자
        "-(-3)",
        "-(-(-3))",
        # 교환법칙 (변수*숫자)
        # "3 * x",
        # "x * 3",
        # 괄호식
        # "(x + 3) * 2",
        # "(3 + x) * 2",
        # 혼합
        # "-(+x)",
        "log(2) + e",
        # "cos(x) * y + 2",
    ]
    for exp in test_expressions:
        print(f"Original: {exp}")
        result, ast = calculate_expression(exp, verbose=True)
        print(f"## Canonicalized\n{ast.canonicalize()}")
        print(f"## Result: {result}")
        print()

if __name__ == "__main__":
    # test_lexer()
    # main()
    test_canonicalize()