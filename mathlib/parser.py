import re
from decimal import Decimal
from collections import deque

from constants import *
from expressions import *

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return f"Token({self.value}, type:{self.value})"
    
    def __repr__(self):
        return f"Token({self.value}, type:{self.value})"


class Lexer:
    TOKEN_REGEX = re.compile(r'\s*(\d+\.\d+|\d+|[()+\-*/^,]|[a-zA-Z]+)') # 실수 | 정수 | 연산자 | 수학 상수, 함수, 변수

    @staticmethod
    def tokenize(expression: str, verbose: bool = False) -> deque[Token]:
        tokens = []
        position = 0

        while position < len(expression):
            match = Lexer.TOKEN_REGEX.match(expression, position)
            if match:
                token_value = match.group(0).strip()
                if token_value:
                    # 1. 기본 연산자
                    if token_value in OPERATORS:
                        tokens.append(Token(token_value, token_value))
                    # 1-1. 기타 기호
                    elif token_value in (',', '(', ')'):
                        tokens.append(Token(token_value, token_value))
                    # 2. 주요 수학 상수 (pi, e 등)
                    elif token_value in MATH_CONSTANTS:
                        tokens.append(Token('const', token_value))
                    # 3. 수학 함수 (sin, cos 등)
                    elif token_value in FUNCTIONS:
                        tokens.append(Token('func', token_value))
                    # 4. 숫자
                    elif re.fullmatch(r'\d+\.\d+|\d+', token_value):  # 숫자 검증 강화 (소수점 두개 방지)
                        tokens.append(Token('num', token_value))
                    # 5. 그 외 알파벳 -> 변수
                    elif token_value.isalpha():
                        tokens.append(Token('var', token_value))
                    else:
                        raise ValueError(f"Invalid token: {token_value}")
                position = match.end()

            else:
                raise ValueError(f"Invalid character at position {position}: '{expression[position]}'")
        
        if verbose:
            Lexer._print_token(tokens)

        return deque(tokens)
    
    @staticmethod
    def _print_token(tokens):
        print("Tokens:")
        for token in tokens:
            print("   ", token)


class MathParser:
    def __init__(self):
        self.tokens = None
        self.current_token = None

    def consume(self, expected: str):
        if self.current_token and self.current_token.type == expected:
            self.current_token = self.tokens.popleft() if self.tokens else None
        else:
            raise ValueError(f"Unexpected token: {self.current_token} (expected: {expected})")

    def parse(self) -> 'Expression':
        self.tokens = deque(self.tokens)
        self.current_token = self.tokens.popleft() if self.tokens else None
        return self.parse_expression()
    
    def parse_expression(self) -> 'Expression':
        # E (expression, 다항식) 문법 규칙: E -> T ( (+ T) | (- T) )*
        node = self.parse_term()
        while self.current_token and self.current_token.type in ('+', '-'):
            op = self.current_token.value
            self.consume(op)
            node = BinOpExpression(node, op, self.parse_term())
        return node
    
    def parse_term(self) -> 'Expression':
        # T (Term, 단항식) 문법 규칙: T -> F ( (* F) | (/ F) )*
        node = self.parse_power()
        while self.current_token and self.current_token.type in ('*', '/'):
            op = self.current_token.value
            self.consume(op)
            node = BinOpExpression(node, op, self.parse_power())
        return node
    
    def parse_power(self) -> 'Expression':
        # P (Power, 거듭제곱) 문법 규칙: P -> F (^ F)
        node = self.parse_factor()
        while self.current_token and self.current_token.type == '^':
            op = self.current_token.value
            self.consume(op)
            node = BinOpExpression(node, op, self.parse_factor())

        if hasattr(node, 'left') and node.left == NumExpression(MATH_CONSTANTS['e']):
            return SingleVarFunction("exp", node.right)
        
        return node
        
    
    def parse_factor(self) -> 'Expression':
        # F (Factor, 인자) 문법 규칙: F -> num | (E) | (+ F) | (- F)
        if self.current_token is None:  
            raise ValueError("Unexpected end of input.")
        
        if self.current_token.type == 'num':
            val = self.current_token.value
            self.consume('num')
            return NumExpression(Decimal(val))
        
        elif self.current_token.type == 'const':
            val = self.current_token.value
            self.consume('const')
            return NumExpression(MATH_CONSTANTS[val], name=val)
        
        elif self.current_token.type == '+':
            self.consume('+')
            sub_node = self.parse_factor()
            return UnaryOpExpression('+', sub_node)
        
        elif self.current_token.type == '-':
            self.consume('-')
            sub_node = self.parse_factor()
            return UnaryOpExpression('-', sub_node)
        
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

            if len(arguments) == 1:
                if func_name == 'sqrt':
                    return BinOpExpression(arguments[0], '^', NumExpression(0.5))
                if func_name == 'log':
                    func_name = 'ln'
                return SingleVarFunction(func_name, arguments[0])
            
            else:
                return MultiVarFunction(func_name, *arguments)
        
        elif self.current_token.type == 'var':
            var_name = self.current_token.value
            self.consume('var')
            return VarExpression(var_name)
        
        else:
            raise ValueError(f"Unexpected token: {self.current_token.value}")
        

class Evaluator:
    @staticmethod
    def evaluate(ast: 'Expression', env: dict = None, verbose: bool = False) -> Decimal:
        result = ast.canonicalize().evaluate(env)

        if verbose:
            print("\n## Equation (str) ##")
            print(str(ast)) # 바깥 괄호 제거
            print("\n## AST (repr) ##")
            print(ast.__repr__())
            print("\n## Canonicalized:", ast.canonicalize())
            print("\n## Result ##")
            print(result)
            print("Domain:\n", ast._domain_str())
        return result

def calculate_expression(expression: str, verbose: bool = False) -> Decimal:
    tokens = Lexer.tokenize(expression)
    parser = MathParser()
    parser.tokens = tokens
    ast = parser.parse()
    result = Evaluator.evaluate(ast, verbose=verbose)
    return result, ast
