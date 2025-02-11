from decimal import Decimal
from utils import apply_operator

def evaluate(expression):
    stack = []
    for token in expression.split():
        if token not in ['+', '-', '*', '/']:
            stack.append(Decimal(token))
        else:
            b = stack.pop()
            a = stack.pop()
            result = apply_operator(token, a, b)
            stack.append(result)
    
    return stack[0]


def main():
    print("Stack Calculator")
    expression = "-3.2 2 + 0.6 / -1 *"
    print(f"Example: {expression}")

    try:
        result = evaluate(expression)
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
            result = evaluate(expression)
            print(f"Result: {result}")
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()