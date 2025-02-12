# 파일럿 프로젝트 사전 준비

### About
1. 스택 기반 계산기 (`stack_calculator.py`)
    - **Input**: 후위 표기법 기반의 수식
    - **Output**: 계산된 결과 값

2. LL(1) 파싱 계산기 (`ll1_calculator.py`)
    - **Input**: 중위 표기법 기반의 수식
    - **Output**: LL(1) 파싱 기반으로 계산된 결과 값
        - `str` 출력: 수식의 중위 표기
        - `repr` 출력: 수식의 파싱된 트리 표현

### Background: LL(1) 문법

- Chomsky 문법 계층 구조
    - 0-type: unrestricted grammar -> 제약이 없고, 가장 표현력이 강함.
    - 1-type: context-sensitive grammar -> 좌우 문맥이 중요한 문법
        - 좌항이 두개 이상의 문자열로 이루어진 문법
    - 2-type: context-free grammar -> 대부분의 프로그래밍 언어에서 사용됨 (LL(1) 포함)
        - 좌항이 하나의 문자열로 이루어진 문법. 우항이 expansion 형태로 표현됨.
    - 3-type: regular grammar -> 정규표현식과 같이 가장 제약이 많고 단순한 문법.
        - 우항에 terminal 항만 포함되어 있음.
    일반적으로 컴파일러 단계에서는 

- LL(1) 문법은?
    - 조건
        1. 파싱 시 분기가 다음 항 1개만으로 결정되어야 함.
            - LL(K) 문법은 다음 항 K 개를 보고 분기가 결정됨.
            - 1개만 보고 분기가 결정되기 때문에 복잡한 파싱 테이블을 사용할 필요 x.
        2. 좌측 재귀가 없어야 함.
            - first 함수와 follow 함수가 교차되지 않아야 함.
    - 정의
        1. **식 (Expression, E)** : 다항식. 내부적으로는 더하기(+), 빼기(-) 연산을 처리.
            - EBNF: E -> T ((+ T) | (- T))*
        2. **항 (Term, T)** : 합의 요소. 내부적으로는 곱하기(*), 나누기(/) 연산을 처리.
            - EBNF: T -> F ((* F) | (/ F))*
        3. **인자 (Factor, F)** : 곱의 요소. 숫자, 괄호, 단항 연산자 (+, -)
            - EBNF: F -> num | (E) | + F | - F
    - 파싱 과정
        1. 토큰 인덱스 초기화 (0에서 시작)
        2. E → T → F 순으로 재귀적으로 파싱
            - 단말 기호가 나오면 적절한 노드를 생성하고, 인덱스를 증가
            - 문법 규칙에 맞지 않는 토큰이 나오면 오류 처리 (e.g. 닫히지 않은 괄호)
        3. 모든 토큰이 정상적으로 파싱됐는지 확인 (EOF 체크)
        4. AST 반환 (디버깅 모드에서는 파싱 트리 출력 가능)
        
### Implementation

1. 스택 기반 계산기
    - **`evaluate(expression)`**
        - 후위 표기법을 기반으로 스택을 사용하여 연산 수행
    - **`main()`**
        - 예제 수식 실행 및 사용자 입력 처리

2. LL(1) 기반 계산기
    - `LL1Calculator` 클래스
        - **`tokenize(expression)`** -> 정규 표현식(regex)을 사용해 입력된 수식을 토큰 리스트로 변환
        - **`parse()`** -> AST 생성
        - **`calculate(expression, debug=False)`** -> 수식 계산 및 결과 반환
        - **`parse_expression()`, `parse_term()`, `parse_factor()`**
            - LL(1) 문법 규칙(E, T, F)에 따라 재귀적으로 노드를 생성
    - **`main()`**
        - 예시 수식 실행 및 사용자 입력 처리



### Test cases
`test_ll1.py`
- 기본 사칙연산
- 연산자 우선순위 (e.g. 곱셈, 나눗셈, 괄호)
- 단항 연산자 (e.g. +, -)
- 괄호 연산
- 소수점 처리 (e.g. 1.23, 0.001 등에서 부동소수점 오류 발생 여부)
- 예외 처리 (e.g. 0으로 나누기, 잘못된 문법, 닫히지 않은 괄호, 잘못된 문자)
- 극단적인 입력


### 추가 기능 (반영 완료)
1. 타입 힌트
    - 서로 다른 종류의 Node 클래스가 존재하고 각각이 서로 다른 evaluate 함수를 가지기 때문에, 이를 모두 포함하는 상위 클래스를 만들어 타입 힌트를 주고자 함.
    - 새로운 클래스를 추가할 때 타입 체크 가능 (type safety - 오류를 구현 과정에서 발견할 수 있음) 하고, 가독성 up
    - 자주/빈번하게/다양한 곳에서 호출되는 함수일수록 hinting 하는 게 좋다
    - 거시적으로는, 객체 지향 프로그래밍(encapsulating, polymorphism, inheritance)의 개념이 적용됨
        - e.g. evaluate 함수는 전체 수식(node의 중첩)에 대해 한 번에 적용되는게 아니라 각 객체에서 개별적으로 동작. 하지만 output 형식이 정해져있으므로, 이를 명시화하기 위함.

2. 컨벤션 상 존재하는 함수
    - `is_{type}` : next token의 type을 검사. terminal 대상 -> 구현 x
    - `consume` : is_ 검사 후 next token을 소비. 아니면 syntax error 발생. terminal 대상
    - `parse` : non-terminal 대상.

3. 재귀 호출의 한계 (구현 x, out of scope)
    - sol : tail recursion, stack 자료구조와 반복 loop 구현