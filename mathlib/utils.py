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


class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return f"Token({self.value}, type:{self.value})"
    
    def __repr__(self):
        return f"Token({self.value}, type:{self.value})"
    
