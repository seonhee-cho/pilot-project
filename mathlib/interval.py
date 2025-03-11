import math
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Optional
from fractions import Fraction

class Interval:
    """
    expression의 정의역을 나타내기 위한 클래스.
    변수가 정의되는 범위: start, end, (closed_start, closed_end) 로 표현.
    제외할 점들: excluded_points 변수 (set[Decimal])
    그 외의 조건들: conditions 변수 (list[tuple[Expression, str, Decimal]])
        - Expression: 조건을 표현하는 수학식
        - str: 조건의 타입 (all, neq, leq, geq, lt, gt)
        - Decimal: 조건의 값
        따라서 conditions 변수가 존재한다면, 수식에 실제 값을 넣어 evaluate한 결과를 확인하여 조건을 만족하는지 확인.
    """
    def __init__(
            self,
            name: str,
            start: Decimal,
            end: Decimal,
            closed_start: bool = True,
            closed_end: bool = True,
            excluded_points: Optional[set[Decimal]] = None,
            conditions: Optional[list[tuple]] = None # list[tuple['Expression', str, NumExpression]]
        ):

        if start > end:
            raise ValueError("Start must be less than end")
        self.name = name
        self.start = start
        self.end = end
        self.closed_start = closed_start
        self.closed_end = closed_end
        self.excluded_points = excluded_points if excluded_points else set()
        self.conditions = conditions if conditions else set()
        
    @classmethod
    def all_real(cls, name: str):
        return cls(name, -math.inf, math.inf, closed_start=False, closed_end=False)
    
    @classmethod
    def parse(cls, s: str, name: str) -> 'Interval':
        s = s.strip().replace(" ", "")
        assert s.startswith("[") or s.startswith("(")
        assert s.endswith("]") or s.endswith(")")
        
        sbracket = s[0]
        ebracket = s[-1]
        
        closed_start = True if sbracket == "[" else False
        closed_end = True if ebracket == "]" else False
        
        start, end = s[1:-1].split(',')
        start = Decimal(start)
        end = Decimal(end)

        return cls(name, start, end, closed_start, closed_end)
    
    def all_points(self):
        # Decimal로 변환 (정밀도 유지를 위해)
        start = Decimal(self.start)
        end = Decimal(self.end)
        # 무한대인 경우 처리
        if start.is_infinite() and start < 0:
            start = Decimal(-100)
        if end.is_infinite() and end > 0:
            end = Decimal(100)

        closed_start = self.closed_start
        closed_end = self.closed_end
        
        # 시작점 처리
        if closed_start:
            lower = int(start.to_integral_value(rounding=ROUND_CEILING))
        else:
            # 시작값이 정수이면 +1, 아니면 ceil
            if start == start.to_integral_value():
                lower = int(start) + 1
            else:
                lower = int(start.to_integral_value(rounding=ROUND_CEILING))
        
        # 끝점 처리
        if closed_end:
            upper = int(end.to_integral_value(rounding=ROUND_FLOOR))
        else:
            if end == end.to_integral_value():
                upper = int(end) - 1
            else:
                upper = int(end.to_integral_value(rounding=ROUND_FLOOR))
        
        if lower > upper:
            return []
        return list(range(lower, upper + 1))

    def __eq__(self, other: 'Interval') -> bool:
        if not isinstance(other, Interval):
            return False
        if self.start != other.start:
            return False
        if self.end != other.end:
            return False
        if self.closed_start != other.closed_start:
            return False
        if self.closed_end != other.closed_end:
            return False
        if self.excluded_points != other.excluded_points:
            return False
        if self.conditions != other.conditions:
            return False
        return True
    
    def __hash__(self):
        excluded_points = tuple(sorted(self.excluded_points))
        conditions = tuple(sorted(((id(expr), cond_type, val) for (expr, cond_type, val) in self.conditions)))
        return hash((self.start, self.end, self.closed_start, self.closed_end, 
                     excluded_points, conditions))
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        sbracket = "[" if self.closed_start else "("
        ebracket = "]" if self.closed_end else ")"
        main_str = f"{sbracket}{self.start}, {self.end}{ebracket}"

        if self.excluded_points:
            ex_pts_str = f"excluded_points={self.excluded_points}"
            main_str += f", {ex_pts_str}"
        
        if self.conditions:
            cond_dict = {'eq': '=', 'neq': '≠', 'leq': '≤', 'geq': '≥', 'lt': '<', 'gt': '>'}
            cond_str = f"\n - condition:{'\n'.join([(str(expr.canonicalize()) + ' ' + cond_dict[cond_type] + ' ' + str(val.value)) for expr,cond_type,val in self.conditions])}"
            main_str += f", {cond_str}"
        
        return f"Interval({main_str})"
    
    def contains(self, point) -> bool:
        # 1) 제외할 점이 있는 경우
        if self.excluded_points and point in self.excluded_points:
            return False
        
        # 2) 정의된 범위 내에 들어가는지
        if point < self.start or point > self.end:
            return False
        if (point == self.start and not self.closed_start) or (point == self.end and not self.closed_end):
            return False
        
        # 3) 조건 확인
        for expr, cond_type, val in self.conditions:
            if cond_type != "all":
                res = expr.evaluate(env={self.name: point})
                if cond_type == "neq" and res == val:
                    return False
                elif cond_type == "leq" and res > val:
                    return False
                elif cond_type == "geq" and res < val:
                    return False
                elif cond_type == "lt" and res >= val:
                    return False
                elif cond_type == "gt" and res <= val:
                    return False
                elif cond_type == "integer|odd_nominator":
                    flag = False
                    if res.is_integer():
                        flag = True
                    else:
                        frac = Fraction(res.value).limit_denominator()
                        if frac.denominator % 2 != 0:
                            flag = True
                    if not flag:
                        return False
        return True

    def intersects(self, other: 'Interval') -> 'Interval':
        assert self.name == other.name
        # 구간 간의 교차범위 구하기
        new_start = max(self.start, other.start)
        new_end = min(self.end, other.end)
        
        if new_start > new_end:
            return None
            
        # 시작점이 같을 때는 둘 다 닫힌 구간이어야 포함
        new_closed_start = (self.closed_start and other.closed_start) if new_start == self.start == other.start \
            else (self.closed_start if new_start == self.start else other.closed_start)
            
        # 끝점이 같을 때는 둘 다 닫힌 구간이어야 포함  
        new_closed_end = (self.closed_end and other.closed_end) if new_end == self.end == other.end \
            else (self.closed_end if new_end == self.end else other.closed_end)
        
        new_excluded_points = self.excluded_points | other.excluded_points
        new_conditions = self.conditions | other.conditions
            
        return Interval(self.name, new_start, new_end, new_closed_start, new_closed_end, new_excluded_points, new_conditions)
    
    def issubset(self, other: 'Interval') -> bool:
        # 구간 간의 포함 관계 확인. self가 other 에 포함되는지 확인
        # other : -inf, inf # self : -1, 1
        if self.start >= self.end:
            raise ValueError("Interval is not valid: start must be less than end")
        
        if (other.start < self.start < other.end) or (other.start == self.start and other.closed_start):
            if (other.start < self.end < other.end) or (other.end == self.end and other.closed_end):
                for ex_pt in self.excluded_points:
                    if other.contains(ex_pt):
                        return False
                return True
        
        return False
    

def merge_domains(domainA: dict[str, Interval], domainB: dict[str, Interval]) -> dict[str, Interval]:
    merged = {}
    all_vars = set(domainA.keys()) | set(domainB.keys())

    for var in all_vars:
        intervalA = domainA.get(var, Interval.all_real(var))
        intervalB = domainB.get(var, Interval.all_real(var))

        intersected = intervalA.intersects(intervalB)
        if intersected is None:
            return {}
        else:
            merged[var] = intersected
            
    return merged