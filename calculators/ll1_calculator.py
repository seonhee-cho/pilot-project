import re
from collections import deque

from utils import *

class Lexer:
    TOKEN_REGEX = re.compile(r'\s*(\d+\.\d+|\d+|[()+\-*/^])')

    @staticmethod
    def tokenize(expression: str) -> deque[Token]:
        tokens = []
        position = 0

        while position < len(expression):
            match = Lexer.TOKEN_REGEX.match(expression, position)
            if match:
                token_value = match.group(0).strip()
                if token_value:
                    if token_value in ['(', ')', '+', '-', '*', '/']:
                        tokens.append(Token(token_value, token_value))
                    else:
                        tokens.append(Token('num', token_value))
                position = match.end()
            else:
                raise ValueError(f"Invalid character at position {position}: '{expression[position]}'")
        
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
        node = self.parse_factor()
        while self.current_token and self.current_token.type in ('*', '/'):
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
        
        else:
            raise ValueError(f"Unexpected token: {self.current_token.value}")


class Evaluator:
    def evaluate(self, ast: Node, verbose: bool = False) -> Decimal:
        # 결과 계산 뿐 아니라 계산 과정을 출력하는 기능도 추가.
        if verbose:
            print("\n## Equation (str):")
            print(str(ast)[1:-1]) # 바깥 괄호 제거
            print("\n## AST (repr):")
            print(ast.__repr__())
            print("\n## Result:", ast.evaluate())

        return ast.evaluate()
    


def main():
    parser = Parser()
    evaluator = Evaluator()
    while True:
        try:
            expression = input("Enter expression (or 'q' to quit): ")
            if expression.lower() == 'q':
                break
            
            tokens = Lexer.tokenize(expression)
            parser.tokens = tokens
            parser.current_token = parser.tokens.popleft() if parser.tokens else None
            
            ast = parser.parse()
            result = evaluator.evaluate(ast, verbose=True)
            print("Result:", result)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()