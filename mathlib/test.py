from constants import *
from expressions import *
from utils import *
from parser import *

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
        # 초등함수 - 대수함수 (다항함수, 유리함수)
        # "log(x, 100)",
        # "log(sin(y^3), x)",
        # "log(y, sin(x^e))",
        # "x^2 + x*3 + 2",
        "x^(1/2)+ x^3 + 2",
        # "(x^2 + 3*x + 2) / (x + 1)", # x + 2
        # "log(diff(3 * x, x) / cos(x))",
        # "diff(x*3, x)",
        # "sin(log(2) + e)",
        # "diff(x^2 / (x + 1), x)",
        # "diff(x^2 + y^2, x)",
        # "diff(x^2 + y^2, y)",
        # "diff(x^2 + y^2, x, y)",
        # "diff((x^2 + 2x + 1) / (x + 1), x)", #### UNEXPECTED TOKEN x, EXPECTED )
        # "diff(cos(sin(x^2)) + y^2, x))",
    ]
    for exp in test_expressions:
        exp = exp.strip()
        print(f"Original: {exp}")
        result, ast = calculate_expression(exp, verbose=True)
        # print(result)
        variables = collect_var_names(result)
        while variables:
            _continue = False

            # domain 입력
            for var in variables:
                print(f"\nEnter the domain for {var} (or press Enter to skip): ")
                print("Format: [start, end] or [start, end) or (start, end] or (start, end)")
                new_domain = input()

                if new_domain:
                    _continue = True
                    domain = Interval.parse(new_domain)
                    ast.update_domain(var, domain)
                    print(f"Updated Domain - {ast._domain_str()}")
            
            
            values = {}
            for var in variables:
                value = input(f"Enter the value for {var} (or press Enter to skip): ")
                if value:
                    _continue = True
                    if ast.domain[var].contains(Decimal(value)):
                        values[var] = Decimal(value)
                    else:
                        raise ValueError(f"Value {value} is not in the domain of {var}")
                    
            result_with_values = ast.evaluate(values)
            print(f"Result: {result_with_values}")

            if not _continue: # 아무 변수의 값도 입력되지 않으면 반복문 종료
                break

if __name__ == "__main__":
    # test_lexer()
    # main()
    test_canonicalize()