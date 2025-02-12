from ll1_calculator import Lexer, Parser, Evaluator
from decimal import Decimal

def run_test_cases():
    test_cases = [
        # 📌 기본 기능 테스트 (Basic Functionality)
        ("1+2-3*6/2", -6),
        # ("4-2", 2),
        # ("6*3", 18),
        # ("8/2", 4),

        # 📌 연산자 우선순위 테스트 (Operator Precedence)
        ("2+3*4", 14),  # 곱셈이 먼저 계산됨
        ("(2+3)*4", 20),  # 괄호 안이 먼저 계산됨
        ("10-5/5", 9),  # 나눗셈이 먼저 계산됨

        # 📌 단항 연산자 테스트 (Unary Operators)
        ("-3+5", 2),  # 단항 - 처리
        ("+3+5", 8),  # 단항 + 처리
        ("-3*-2", 6),  # 두 개의 단항 - 처리
        ("--3", 3),  # 두 번 적용된 단항 - 처리

        # 📌 괄호 연산 테스트 (Parentheses Handling)
        ("(1+2)*3", 9),
        ("(10-5)/5", 1),
        ("((2+3)*4)-5", 15),
        ("-(3+2)", -5),

        # 📌 소수점 처리 테스트 (Floating Point Handling)
        ("3.5+2.5", 6.0),
        ("10.0/4", 2.5),
        ("(1.2+3.4)*2", 9.2),
        ("-0.5*2", -1.0),

        # 📌 예외 처리 테스트 (Error Handling)
        ("5/0", ZeroDivisionError),  # 0으로 나누기
        ("2++3", 5),  # 잘못된 연산자 사용
        ("abc+1", ValueError),  # 잘못된 문자 포함
        ("(2+3", ValueError),  # 닫히지 않은 괄호
        ("3+*", ValueError),  # 올바르지 않은 연산

        # 📌 극단적인 입력 테스트 (Edge Cases)
        ("9999999999+1", 10000000000),  # 큰 수 연산
        ("0.0000001*10000000", 1.0),  # 작은 수 연산
        ("0", 0),  # 단일 숫자
        ("-0", 0),  # 음수 0 처리
        ("", ValueError),

        # 📌 대규모 입력 테스트 (Performance Testing)
        # 시스템의 재귀 깊이를 초과하는 너무 긴 입력은 테스트 목적이 아니므로 주석 처리
        # ("+".join(["1"] * 1000), 1000),  # 1을 10000번 더하기
        # ("*".join(["1"] * 1000), 1),  # 1을 10000번 더하기
        # ("1" + "*" + "1" * 10000, int("1"*10000)),  # 1을 10000자리 수와 곱하기 (테스트 목적)
    ]

    passed = 0
    failed = 0

    parser = Parser()
    evaluator = Evaluator()

    for expr, expected in test_cases:
        # 긴 입력이면 출력 생략
        if len(expr) > 100:
            expr_display = expr[:50] + " ... (truncated)"
        else:
            expr_display = expr
        try:
            tokens = Lexer.tokenize(expr)
            parser.tokens = tokens
            parser.current_token = parser.tokens.popleft() if parser.tokens else None
            
            ast = parser.parse()
            result = evaluator.evaluate(ast)


            if isinstance(expected, type) and issubclass(expected, Exception):
                print(f"❌ Test failed (Expected exception): {expr_display}")
                failed += 1
            elif result == Decimal(str(expected)):
                print(f"✅ Passed: {expr_display} = {result}")
                passed += 1
            else:
                print(f"❌ Failed: {expr_display} (Expected {expected}, Got {result})")
                import pdb; pdb.set_trace()
                failed += 1

        except Exception as e:
            if isinstance(expected, type) and isinstance(e, expected):
                print(f"✅ Passed (Expected exception): {expr_display} -> {e}")
                passed += 1
            else:
                print(f"❌ Failed: {expr_display} (Unexpected Exception: {e})")
                import pdb; pdb.set_trace()
                failed += 1

    print(f"\n=== Test Summary ===")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    run_test_cases()