import re
from utils import *

class LL1Calculator:
    def __init__(self):
        self.tokens = []
        self.index = 0
        self.current_token = None


    def tokenize(self, expression: str) -> list:
        token_spec = [
            ('LPAREN',  r'\('),        # (
            ('RPAREN',  r'\)'),        # )
            ('PLUS',    r'\+'),        # +
            ('MINUS',   r'-'),         # -
            ('STAR',    r'\*'),        # *
            ('SLASH',   r'/'),         # /
            ('NUM',     r'\d+(\.\d+)?'),  # 숫자(부호 없이). 정수나 소수
            ('WS',      r'\s+'),       # 공백 (무시)
        ]
        tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
        get_token = re.compile(tok_regex).match

        pos = 0
        tokens = []
        while pos < len(expression):
            match = get_token(expression, pos)
            if match is None:
                raise ValueError(f"Invalid character at position {pos}: {expression[pos]}")
                
            tok = match.lastgroup
            tok_val = match.group(tok)
            if tok != 'WS':
                # Token type 설정 - 괄호와 연산자는 값과 동일하게 설정, 숫자는 num으로 설정.
                if tok in ['PLUS', 'MINUS', 'STAR', 'SLASH', 'LPAREN', 'RPAREN']:
                    tok = tok_val
                elif tok == 'NUM':
                    tok = 'num'
                tokens.append(Token(tok, tok_val))
                
            pos = match.end()
        
        return tokens
    

    def get_next_token(self):
        if self.index < len(self.tokens):
            self.current_token = self.tokens[self.index]
            self.index += 1
        else:
            self.current_token = None


    def match(self, token_type):
        # 현재 토큰이 expected token type과 같으면 토큰을 반환하고, 다음 토큰으로 이동.
        if self.current_token and self.current_token.type == token_type:
            value = self.current_token.value
            self.get_next_token()
            return value
        else:
            raise ValueError(
                f"Unexpected token: {self.current_token.value} (type: {self.current_token.type})"
                if self.current_token else "Unexpected end of input." + (f"Expected {token_type}" )
            )
        
    
    def parse(self):
        self.index = 0
        self.get_next_token()
        ast = self.parse_expression()

        # EOF 토큰이 남았는지 확인
        if self.current_token and self.current_token.type != 'EOF':
            raise ValueError("Unexpected token")
        return ast
    
    
    def parse_expression(self):
        # E (expression) 문법 규칙:
            # (1) E -> T E'
            # (2) E' -> + T E' | - T E' | ε (빈 문자열)
        # Term을 먼저 파싱한 후 E'에 해당하는 연산자(+, -)가 나오면 한 번 더 Term을 파싱하여 이항 연산.
        # 남은 입력 값에 E'가 없을 경우, Term을 바로 반환.

        node = self.parse_term()

        while self.current_token and self.current_token.type in ['+', '-']:
            op = self.current_token.type
            self.match(op)

            right_node = self.parse_term()
            node = BinOpNode(node, op, right_node)

        return node
    
    
    def parse_term(self):
        # T (Term) 문법 규칙:
            # (1) T -> F ( ('*' F) | ('/' F) )*
            # (2) T' -> * F T' | / F T' | ε (빈 문자열)
        # Factor를 먼저 파싱한 후 T'에 해당하는 연산자(*, /)가 나오면 한 번 더 Factor를 파싱하여 이항 연산.
        # 남은 입력 값에 T'가 없을 경우, Factor를 바로 반환.

        node = self.parse_factor()

        while self.current_token and self.current_token.type in ['*', '/']:
            op = self.current_token.type
            self.match(op)

            right_node = self.parse_factor()
            node = BinOpNode(node, op, right_node)

        return node
    
    
    def parse_factor(self):
        # F (Factor) 문법 규칙: F -> num | (E) | (+ F) | (- F)
        # 숫자(num), 괄호 내 표현식(E), 또는 단항 연산자('+', '-')로 시작하는 Factor를 파싱.
        if self.current_token is None:
            raise ValueError("Unexpected end of input.")

        if self.current_token.type == '(':
            self.match('(')
            node = self.parse_expression()
            self.match(')')
            return node
        
        elif self.current_token.type == '+':
            self.match('+')
            sub_node = self.parse_factor()
            return UnaryOpNode('+', sub_node)
        
        elif self.current_token.type == '-':
            self.match('-')
            sub_node = self.parse_factor()
            return UnaryOpNode('-', sub_node)
        
        elif self.current_token.type == 'num':
            val = self.match('num')
            return NumNode(val)
        
        else:
            raise ValueError(f"Unexpected token: {self.current_token.value} (type: {self.current_token.type})"
                             if self.current_token is not None else "Unexpected end of input.")
        
    
    
    def calculate(self, expression: str, debug: bool = False):
        print("## Input expression:", expression)

        self.tokens = self.tokenize(expression)
        if debug:
            print("\n## Tokens:")
            for token in self.tokens:
                print(repr(token.value), end=' ')
            print()

        ast = self.parse()

        if debug:
            print("\n## Equation (str):")
            print(str(ast)[1:-1]) # 바깥 괄호 제거
            print("\n## AST (repr):")
            print(ast.__repr__())

        if debug:
            print("\n## Result:", ast.evaluate())
        return ast.evaluate()

def main():
    calculator = LL1Calculator()

    print("LL(1) Calculator")
    expression = "(-3.2+2)/0.6*-1"
    print(f"Example: {expression}")

    try:
        result = calculator.calculate(expression, debug=True)
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Error: {e}")

    print("Enter an expression to calculate. (Enter 'q' to quit)")
    while True:
        expression = input("> ")
        if expression.lower() == 'q':
            print("Exiting the program.")
            break
        try:
            result = calculator.calculate(expression, debug=True)
            print(f"Result: {result}")
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()