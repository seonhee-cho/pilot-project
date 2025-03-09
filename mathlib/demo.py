import gradio as gr
from constants import *
from expressions import *
from parser import *
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
from decimal import Decimal, getcontext, ROUND_HALF_DOWN
from mpl_toolkits.mplot3d import Axes3D  # 3D plot

getcontext().prec = 10

def decimal_to_float(d, places=10):
    return float(format(d, f".{places}f"))

def create_constant_figure(const_value=3):
    x_vals = np.linspace(0, 10, 100)
    y_vals = np.full_like(x_vals, const_value)
    
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=f"y={const_value}")
    ax.set_ylim(const_value - 1, const_value + 1)  # y 범위 설정
    ax.set_title("Constant Function")
    ax.legend()
    return fig

def create_figure_from_expr(expr, domain_dict: dict[str, Interval] = None, fig: plt.Figure = None, idx: int = 0):
    """
    expr 내의 변수를 확인해 1D/2D/3D 이상 등의 경우를 나눠 
    Matplotlib figure를 만들어 반환.
    """

    variables = list(expr.vars_)
    n = len(variables)
    domain_dict = domain_dict if domain_dict else expr.domain
    # 샘플링 설정
    samples = {}
    for var in variables:
        interval = domain_dict[var]
        start, end = interval.start, interval.end
        if start == -np.inf:
            start = -100
        if end == np.inf:
            end = 100
        samples[var] = [float(str(x)) if interval.contains(x) else np.nan for x in np.linspace(start, end, 100)]
    
    if n == 0:
        # 상수 함수 처리
        fig = create_constant_figure(expr.evaluate().value)
        return fig

    if n == 1:
        # 1차원 그래프
        var = variables[0]
        x_vals = samples[var]
        y_vals = []
        for xi in x_vals:
            if not np.isnan(xi):
                try:
                    y_val = expr.evaluate({var: xi})
                    y_vals.append(decimal_to_float(y_val.value.quantize(Decimal("1.0000000000"), rounding=ROUND_HALF_DOWN).normalize()))
                except:
                    y_vals.append(np.nan)
            else:
                y_vals.append(np.nan)
        y_vals = np.array(y_vals, dtype=np.float32)

        if fig is None:
            fig, ax = plt.subplots(figsize=(6,4))
        else:
            ax = fig.gca()
        ax.plot(x_vals, y_vals, marker='o')
        ax.set_xlabel(var)
        ax.set_ylabel(f"f({var})")
        ax.set_title(str(expr.canonicalize()))
        ax.grid(True)

    elif n == 2:
        # 2차원 (3D surface)
        X, Y = np.meshgrid(samples[variables[0]], samples[variables[1]])
        points_list = []
        for idx in range(X.size):
            point = {variables[0]: X.flat[idx], variables[1]: Y.flat[idx]}
            points_list.append(point)
        values = []
        for pt in points_list:
            if not np.isnan(pt[variables[0]]) and not np.isnan(pt[variables[1]]):
                try:
                    y_val = expr.evaluate(pt)
                    values.append(decimal_to_float(y_val.value.quantize(Decimal("1.0000000000"), rounding=ROUND_HALF_DOWN).normalize()))
                except:
                    values.append(np.nan)
            else:
                values.append(np.nan)
        values = np.array(values, dtype=np.float32)
        Z = values.reshape(X.shape)
        if fig is None:
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_xlabel(variables[0])
            ax.set_ylabel(variables[1])
            ax.set_zlabel(f"f({variables[0]}, {variables[1]})")
            ax.set_title(str(expr.canonicalize()))

    else:
        # 3개 이상의 변수 => 무작위 샘플링
        points_list = []
        values = []
        for _ in range(100):
            point = {}
            for var in variables:
                interval = expr.domain[var]
                start, end = interval.start, interval.end
                if start == -np.inf:
                    start = -10
                if end == np.inf:
                    end = 10
                point[var] = np.random.uniform(start, end)
            points_list.append(point)
            values.append(
                decimal_to_float(expr.evaluate(point).value.quantize(Decimal("1.0000000000"), rounding=ROUND_HALF_DOWN))
                if domain_dict[variables[0]].contains(point[variables[0]]) and domain_dict[variables[1]].contains(point[variables[1]])
                else np.nan
            )
        values = np.array(values, dtype=np.float32)
        
        if fig is None:
            fig, ax = plt.subplots(figsize=(6,4))
        else:
            ax = fig.gca()
        ax.plot(range(len(values)), values, marker='o')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Function Value")
        ax.set_title("Plot of the Expression on Random Points")
        ax.grid(True)

    return fig

def plot_graph_gradio(state_ast, state_vars):
    """
    state_ast 가 변할 때마다 새로운 그래프를 그리도록 호출될 함수.
    state_ast: 현재 수식(혹은 파싱된 AST)
    state_vars: 현재 수식에서 추출된 변수 정보 (딕셔너리 or 리스트)

    1. 변수가 없으면 메시지 반환
    2. 변수가 있으면, 
       - expr 객체를 얻은 뒤(사용자 코드에 맞춰 변환 필요),
       - 도메인을 설정하고(기본 [-10, 10] 혹은 expr.domain에 이미 값이 있다면 활용),
       - Matplotlib으로 그래프 생성 후 figure를 반환.
    """
    if not state_ast:
        return plt.figure(figsize=(6,4))
    
    if isinstance(state_ast, UnionExpression) or len(state_vars) > 2:
        return plt.figure(figsize=(6,4))
    
    # 우선, 변수가 전혀 없다면 (상수 함수라면) 그래프를 그릴 수 없으므로 안내 메시지
    if not state_vars or len(state_vars) == 0:
        return create_constant_figure(const_value=state_ast.evaluate().value)

    expr = state_ast.canonicalize()

    if not hasattr(expr, "domain"):
        expr.domain = {}
    for var in expr.vars_:
        if var not in expr.domain:
            expr.domain[var] = Interval(-100, 100)

    fig = create_figure_from_expr(expr, expr.domain) 
    return fig


def add_expression_to_figure(fig, expr, history):
    """
    deprecated.. 구현 실패..
    """
    color_idx = len(history.split("\n"))
    fig = create_figure_from_expr(expr, expr.domain, fig, color_idx)
    return fig

def process_expression(expression):
    expression = expression.strip()
    if not expression:
        return "Error: Empty expression", None, None
    try:
        result, ast = calculate_expression(expression, verbose=False)
        # canonicalized 결과를 LaTeX 형식으로 감싸서 출력
        output = f"$$ {result} $$"
        if ast._domain_str():
            output += f"\n\n**도메인**<br>{ast._domain_str().replace('\n', '<br>')}"
        variables = collect_var_names(result)
        return output, ast, variables
    except Exception as e:
        return f"Error: {str(e)}", None, None
    
def evaluate_value_gradio(ast: Expression, value_table: gr.DataFrame):
    if not ast:
        return "No valid AST provided.", ast
    
    new_value_dict = {}
    # import pdb; pdb.set_trace()
    for _, row in value_table.iterrows():

        if row.iloc[1]:
            assert ast.domain[row['Variable']].contains(Decimal(row.iloc[1])), f"Value {row.iloc[1]} is not in the domain of {row['Variable']}"
            new_value_dict[row['Variable']] = Decimal(row.iloc[1])
        
    result = ast.evaluate(new_value_dict)
    output = f"$$ {result} $$"

    return output

def evaluate_range_gradio(ast, range_table):
    """
    range_table에서 변수 범위 업데이트
    """
    new_domain = {}
    for _, row in range_table.iterrows():
        if row['Range']:
            new_domain[row['Variable']] = Interval.parse(row['Range'], row['Variable'])
    
    for var in ast.domain.keys():
        ast.domain[var] = ast.domain[var].intersects(new_domain[var])
    
    result = ast.evaluate()
    output = f"$$ {result} $$"
    if ast._domain_str():
        output += f"\n\n**도메인**<br>{ast._domain_str().replace('\n', '<br>')}"
    return output, ast

def evaluate_plot_range_gradio(ast, plot_output_range):
    """
    plot_output_range에서 변수 범위 업데이트
    """
    new_domain = {}
    for _, row in plot_output_range.iterrows():
        new_domain[row['Variable']] = Interval.parse(row['Range'], row['Variable'])
    fig = create_figure_from_expr(ast, domain_dict=new_domain)
    return fig


def check_continuity_gradio(ast, variables, check_type, value_table, range_table):
    point_dict = {}
    if check_type == "값":
        for _, row in value_table.iterrows():
            if row.iloc[1]:
                point_dict[row['Variable']] = Decimal(row.iloc[1])
        continuity = check_continuity_at(ast, point_dict)
    
    elif check_type == "구간":
        variable_list = sorted(variables)
        for _, row in range_table.iterrows():
            if row['Range']:
                point_dict[row['Variable']] = Interval.parse(row['Range'], row['Variable'])

        flag = True
        in_domain_points = {}
        for var, interval in point_dict.items():
            in_domain_points[var] = interval.all_points()

        all_combinations = list(itertools.product(*(in_domain_points[var] for var in variable_list)))
        for point in all_combinations:
            test_point = {var: Decimal(val) for var, val in zip(variable_list, point)}
            continuity = check_continuity_at(ast, test_point)
            if not continuity:
                flag = False
                break
    else:
        raise ValueError(f"Invalid check type: {check_type}")
    
    output = f"Continuity at {', '.join(f'{k}={v}' for k, v in point_dict.items())}:\n\n$$ {continuity} $$"
    return output, ast

def check_differentiability_gradio(ast, variables, check_type, value_table, range_table):
    point_dict = {}
    variable_list = sorted(variables)

    if check_type == "값":
        for _, row in value_table.iterrows():
            if row.iloc[1]:
                point_dict[row['Variable']] = Decimal(row.iloc[1])
        differentiability = check_differentiability_at(ast, point_dict)
    
    elif check_type == "구간":
        for _, row in range_table.iterrows():
            if row['Range']:
                point_dict[row['Variable']] = Interval.parse(row['Range'], row['Variable'])
        flag = True
        in_domain_points = {}
        for var, interval in point_dict.items():
            in_domain_points[var] = interval.all_points()

        all_combinations = list(itertools.product(*(in_domain_points[var] for var in variable_list)))
        for point in all_combinations:
            test_point = {var: Decimal(val) for var, val in zip(variable_list, point)}
            differentiability = check_differentiability_at(ast, test_point)
            if not differentiability:
                flag = False
                break
        if flag:
            differentiability = True
        else:
            differentiability = False
    else:
        raise ValueError(f"Invalid check type: {check_type}")
    
    output = f"Differentiability at {', '.join(f'{k}={v}' for k, v in point_dict.items())}:\n\n$$ {differentiability} $$\n"
    return output, ast

def differentiate_var_gradio(ast, var_choice):
    ast = ast.derivative(var_choice)
    output = f"Result (derivative with respect to {var_choice}): $$ {ast.canonicalize()} $$"
    return output, ast, collect_var_names(ast)

def differentiate_direction_gradio(ast, direction_table):
    direction = {}
    for _, row in direction_table.iterrows():
        if row.iloc[1]:
            direction[row['Variable']] = Decimal(row.iloc[1])
    ast = ast.directional_derivative(direction)
    output = f"Result (directional derivative):\n\n$$ {ast.canonicalize()} $$"
    return output, ast, collect_var_names(ast)

def differentiate_gradient_gradio(ast):
    gradient_ast = ast.gradient()
    output = f"Result (gradient):\n\n$$ {gradient_ast.canonicalize()} $$"
    return output, ast, collect_var_names(gradient_ast)


def update_variable_table(vars_list, cont_input_type):
    """
    vars_list가 비어있지 않으면 각 변수에 대해 [Variable, Domain, Value] 형태의 행을 생성하여 Dataframe을 업데이트합니다.
    """
    if vars_list:
        vars_list = sorted(vars_list)
        table = [[var, ""] for var in vars_list]
        return gr.update(value=table, visible=True), gr.update(value=table, visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)
    
def update_cont_variable_table(vars_list, cont_input_type):
    """
    vars_list가 비어있지 않으면 각 변수에 대해 [Variable, Domain, Value] 형태의 행을 생성하여 Dataframe을 업데이트합니다.
    """
    if vars_list:
        vars_list = sorted(vars_list)
        table = [[var, ""] for var in vars_list]

        return gr.update(value=table, visible=(cont_input_type=='값')), \
                gr.update(value=table, visible=(cont_input_type=='구간'))
    else:
        return gr.update(visible=False), gr.update(visible=False)
    
def update_diff_var_choice(vars_list):
    """
    vars_list가 비어있지 않으면 각 변수에 대해 [Variable, Value] 형태의 행을 생성하여 Dataframe을 업데이트합니다.
    """
    if vars_list:
        vars_list = sorted(vars_list)
        return gr.update(choices=vars_list)
    else:
        return gr.update(choices=[])

def update_diff_variable_table(vars_list, diff_input_type):
    """
    vars_list가 비어있지 않으면 각 변수에 대해 [Variable, Domain, Value] 형태의 행을 생성하여 Dataframe을 업데이트합니다.
    """
    if vars_list:
        vars_list = sorted(vars_list)
        table = [[var, ""] for var in vars_list]

        return gr.update(value=table, visible=(diff_input_type=='값')), \
                gr.update(value=table, visible=(diff_input_type=='구간'))
    else:
        return gr.update(visible=False), gr.update(visible=False)
    
def update_diff_dir_variable_table(vars_list):
    """
    vars_list가 비어있지 않으면 각 변수에 대해 [Variable, Value] 형태의 행을 생성하여 Dataframe을 업데이트합니다.
    """
    if vars_list:
        vars_list = sorted(vars_list)
        table = [[var, ""] for var in vars_list]
        return gr.update(value=table, visible=True)
    else:
        return gr.update(visible=False)
    
def update_plot_range_table(vars_list):
    """
    vars_list가 비어있지 않으면 각 변수에 대해 [Variable, Range] 형태의 행을 생성하여 Dataframe을 업데이트합니다.
    """
    if vars_list:
        vars_list = sorted(vars_list)
        table = [[var, ""] for var in vars_list]
        return gr.update(value=table, visible=True)
    else:
        return gr.update(visible=False)

# --- 새롭게 추가된 함수들 (체크용 테이블) ---

def init_current_expr():
    return ""

def update_current_expr(history, ast):
    # 입력 수식, 
    history_list = history.split("\n")
    history_list.append(f"$$ {ast.canonicalize()} $$")
    return "\n".join(history_list)

def update_table_visibility(choice):
    if choice == "값":
        return gr.update(visible=True), gr.update(visible=False)
    else:  # choice == "구간"
        return gr.update(visible=False), gr.update(visible=True)



# --- UI 구성 ---

global iface
with gr.Blocks() as demo:
    gr.Markdown("# 간단 계산기 GUI (Gradio)")

    # === 첫번째 섹션: 수식 입력 및 결과 출력 (항상 보임) ===
    gr.Markdown("## 1. 수식 입력 및 결과")
    with gr.Row():
        with gr.Column():
            expr_input = gr.Textbox(label="수식 입력", placeholder="예: 2*x + 3")
            submit_expr = gr.Button("수식 평가")
        with gr.Column():
            gr.Markdown("출력 결과")
            output_text = gr.Markdown()
        with gr.Column():
            gr.Markdown("히스토리")
            history_state = gr.State("")
    # 상태: AST와 변수 목록 저장
    state_ast = gr.State()
    state_vars = gr.State()
    history_state = gr.State("")
    submit_expr.click(fn=process_expression, inputs=expr_input, 
                      outputs=[output_text, state_ast, state_vars])
    submit_expr.click(fn=init_current_expr, outputs=history_state)
    state_ast.change(fn=update_current_expr, inputs=[history_state, state_ast], outputs=history_state)
    
    # === 두번째 섹션: 변수 값 업데이트 (수식에 변수가 있을 때) ===
    gr.Markdown("## 2. 변수 값/범위 업데이트")
    with gr.Row():
        with gr.Column():
            value_table = gr.Dataframe(headers=["Variable", "Value (Int or Float only)"], visible=True)
            update_value_btn = gr.Button("변수 값 업데이트")
        with gr.Column():
            range_table = gr.Dataframe(headers=["Variable", "Range"], visible=True)
            update_range_btn = gr.Button("변수 범위 업데이트")

    # === 세번째 섹션: 연속성/미분가능성 검사 ===
    gr.Markdown("## 3. 연속성 및 미분가능성 검사")
    with gr.Row():
        with gr.Column():
            cont_input_type = gr.Radio(choices=["값", "구간"], label="연속성 검사", value="값")
            cont_value_table = gr.Dataframe(label="연속성 검사 (값 입력)", headers=["Variable", "Value (Int or Float only)"], visible=True)
            cont_range_table = gr.Dataframe(label="연속성 검사 (구간 입력)", headers=["Variable", "Range"], visible=False)
            cont_button = gr.Button("연속성 검사")
        with gr.Column():
            diff_input_type = gr.Radio(choices=["값", "구간"], label="미분가능성 검사", value="값")
            diff_value_table = gr.Dataframe(label="미분가능성 검사 (값 입력)", headers=["Variable", "Value (Int or Float only)"], visible=True)
            diff_range_table = gr.Dataframe(label="미분가능성 검사 (구간 입력)", headers=["Variable", "Range"], visible=False)
            diff_button = gr.Button("미분가능성 검사")

    # === 네번째 섹션: 미분 기능 (편미분, 방향 미분, 기울기) ===
    gr.Markdown("## 4. 미분 기능")
    with gr.Row():
        with gr.Column():
            diff_var_choice = gr.Radio(label="편미분 변수 선택", choices=["x", "y", "z"], value="x")
            diff_var_button = gr.Button("편미분 실행")
        with gr.Column():
            diff_direction_table = gr.Dataframe(label="방향 벡터 (방향 미분 시)", headers=["Variable", "Value (Int or Float only)"], visible=True)
            diff_direction_button = gr.Button("방향 미분 실행")
        with gr.Column():
            diff_gradient_button = gr.Button("기울기 구하기")

    gr.Markdown("## 5. 그래프")
    with gr.Row():
        with gr.Column():
            plot_output = gr.Plot()
        with gr.Column():
            plot_output_range = gr.Dataframe(label="그래프 범위 입력", headers=["Variable", "Range"], visible=False)
            plot_output_range_button = gr.Button("그래프 범위 입력")
        
    state_ast.change(fn=plot_graph_gradio, inputs=[state_ast, state_vars], outputs=plot_output)

    state_vars.change(fn=update_variable_table, inputs=[state_vars], outputs=[value_table, range_table])
    state_vars.change(fn=update_cont_variable_table, inputs=[state_vars, cont_input_type], outputs=[cont_value_table, cont_range_table])
    state_vars.change(fn=update_diff_var_choice, inputs=[state_vars], outputs=[diff_var_choice])
    state_vars.change(fn=update_diff_variable_table, inputs=[state_vars, diff_input_type], outputs=[diff_value_table, diff_range_table])
    state_vars.change(fn=update_diff_dir_variable_table, inputs=[state_vars], outputs=[diff_direction_table])
    state_vars.change(fn=update_plot_range_table, inputs=[state_vars], outputs=[plot_output_range])

    # history_state.change(fn=add_expression_to_figure, inputs=[plot_output, state_ast, history_state], outputs=plot_output)

    update_value_btn.click(fn=evaluate_value_gradio, 
                           inputs=[state_ast, value_table], 
                           outputs=[output_text])
    update_range_btn.click(fn=evaluate_range_gradio, 
                           inputs=[state_ast, range_table], 
                           outputs=[output_text, state_ast])
    plot_output_range_button.click(fn=evaluate_plot_range_gradio, 
                           inputs=[state_ast, plot_output_range], 
                           outputs=[plot_output])
    cont_input_type.change(
        fn=update_table_visibility,
        inputs=cont_input_type,
        outputs=[cont_value_table, cont_range_table]
    )
    diff_input_type.change(
        fn=update_table_visibility,
        inputs=diff_input_type,
        outputs=[diff_value_table, diff_range_table]
    )
    cont_button.click(
        fn=check_continuity_gradio,
        inputs=[state_ast, state_vars, cont_input_type, cont_value_table, cont_range_table],
        outputs=[output_text, state_ast]
    )
    diff_button.click(
        fn=check_differentiability_gradio,
        inputs=[state_ast, state_vars, diff_input_type, diff_value_table, diff_range_table],
        outputs=[output_text, state_ast]
    )
    diff_var_button.click(
        fn=differentiate_var_gradio,
        inputs=[state_ast, diff_var_choice],
        outputs=[output_text, state_ast, state_vars]
    )
    diff_direction_button.click(
        fn=differentiate_direction_gradio,
        inputs=[state_ast, diff_direction_table],
        outputs=[output_text, state_ast, state_vars]
    )
    diff_gradient_button.click(
        fn=differentiate_gradient_gradio,
        inputs=[state_ast],
        outputs=[output_text, state_ast, state_vars]
    )
    
iface = demo.launch(show_error=True)
