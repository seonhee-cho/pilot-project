import gradio as gr
from constants import *
from expressions import *
from parser import *
import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
from decimal import Decimal, getcontext, ROUND_HALF_DOWN

def decimal_to_float(d, places=10):
    return float(format(d, f".{places}f"))

# --- 그래프 그리기 함수 ---
def sample_1d(expr, var, n_samples=1000, tol=1e-12):
    """
    1. 주어진 domain [domain.start, domain.end] 범위에서 linspace로 n_samples 개의 점을 샘플링
    2. 그 중 유효한 도메인의 점에 대해서만 y_vals 를 구함. 유효한 정의역이 아니면 np.nan 으로 처리
    3. 유효 점 비율이 너무 작으면 유효 점들의 min~max 값을 구해서 그래프 범위를 결정
    """
    domain = expr.domain.get(var, Interval(var, -10, 10))
    start = domain.start if domain.start != -np.inf else -10
    end = domain.end if domain.end != np.inf else 10

    # 1) 샘플링
    x_vals, y_vals = _sample_1d(expr, var, domain, start, end, n_samples, tol)
    valid_mask = ~np.isnan(y_vals)

    # import pdb; pdb.set_trace()
    max_resampling_count = 100
    resampling_count = 0
    # 2) 유효 점이 20% 미만이면 그 사이에서 다시 샘플링
    while np.sum(valid_mask) / n_samples < 0.2 and resampling_count < max_resampling_count:
        valid_x = x_vals[valid_mask]
        min_x = np.min(valid_x)
        max_x = np.max(valid_x)
        if min_x < max_x: # 유효 점이 1개거나 없는 경우, 일단 고려하지 않음
            x_vals, y_vals = _sample_1d(expr, var, domain, min_x, max_x, n_samples, tol)
        else:
            min_x -= 1
            max_x += 1
        valid_mask = ~np.isnan(y_vals)
        resampling_count += 1

    return x_vals, y_vals

def _sample_1d(expr, var, domain, start, end, n_samples=10000, tol=1e-10):
    x_vals = np.linspace(int(start), int(end), n_samples)
    y_vals = []
    for xi in x_vals:
        if domain.contains(xi):
            try:
                y_val = expr.evaluate({var: xi})
                # 더 높은 정밀도로 설정
                decimal_val = y_val.value.quantize(Decimal("1.0000000000"), rounding=ROUND_HALF_DOWN).normalize()
                # 더 작은 tolerance 값 사용
                if abs(decimal_val) < Decimal('1e-12'):
                    y_vals.append(0.0)
                else:
                    y_vals.append(decimal_to_float(decimal_val))
            except:
                y_vals.append(np.nan)
        else:
            y_vals.append(np.nan)
    y_vals = np.array(y_vals, dtype=np.float64)
    y_vals = np.where(np.abs(y_vals) <= tol, 0, y_vals)
    return x_vals, y_vals

def sample_2d(expr, var1, var2, n_samples=100, tol=1e-12):
    """
    1. 주어진 domain [domain1.start, domain1.end] x [domain2.start, domain2.end] 범위에서 linspace로 n_samples 개의 점을 샘플링
    2. 그 중 유효한 도메인의 점에 대해서만 z_vals 를 구함. 유효한 정의역이 아니면 np.nan 으로 처리
    3. 유효 점 비율이 너무 작으면 유효 점들의 min~max 값을 구해서 그래프 범위를 결정
    """
    domain1 = expr.domain.get(var1, Interval(var1, -10, 10))
    domain2 = expr.domain.get(var2, Interval(var2, -10, 10))

    start1 = domain1.start if domain1.start != -np.inf else -10
    end1 = domain1.end if domain1.end != np.inf else 10
    start2 = domain2.start if domain2.start != -np.inf else -10
    end2 = domain2.end if domain2.end != np.inf else 10

    X, Y, Z = _sample_2d(expr, var1, var2, domain1, domain2, start1, end1, start2, end2, n_samples, tol)
    valid_mask = ~np.isnan(Z)
    
    max_resampling_count = 100
    resampling_count = 0
    while np.sum(valid_mask) / (n_samples * n_samples) < 0.02 and resampling_count < max_resampling_count:
        min_x = np.min(X[valid_mask])
        max_x = np.max(X[valid_mask])
        min_y = np.min(Y[valid_mask])
        max_y = np.max(Y[valid_mask])
        X, Y, Z = _sample_2d(expr, var1, var2, domain1, domain2, min_x, max_x, min_y, max_y, n_samples, tol)
        valid_mask = ~np.isnan(Z)
        resampling_count += 1

    return X, Y, Z

def _sample_2d(expr, var1, var2, domain1, domain2, start1, end1, start2, end2, n_samples=100, tol=1e-10):
    X, Y = np.meshgrid(np.linspace(start1, end1, n_samples), np.linspace(start2, end2, n_samples))
    Z = []
    for idx in range(X.size):
        px = X.flat[idx]
        py = Y.flat[idx]
        if domain1.contains(px) and domain2.contains(py):
            try:
                z_val = expr.evaluate({var1: px, var2: py})
                # 더 높은 정밀도로 설정
                decimal_val = z_val.value.quantize(Decimal("1.00000000000"), rounding=ROUND_HALF_DOWN).normalize()
                # 더 작은 tolerance 값 사용
                if abs(decimal_val) < Decimal('1e-12'):
                    Z.append(0.0)
                else:
                    Z.append(decimal_to_float(decimal_val))
            except:
                Z.append(np.nan)
        else:
            Z.append(np.nan)
    Z = np.array(Z, dtype=np.float64).reshape(X.shape)
    Z = np.where(np.abs(Z) <= tol, 0, Z)
    return X, Y, Z

def create_figure_from_expr(expr_list: list[Expression]):
    """
    expr 리스트를 받아 하나의 figure에 여러 개의 그래프를 그리도록 함.
    단, 원함수와 도함수의 차원 (변수 개수) 이 달라지는 경우, 원함수의 그래프만 반환
    """
    if not expr_list:
        return plt.figure(figsize=(6,4))
    
    expr_list = flatten_expressions(expr_list)
    
    # 첫번째 수식(원함수)
    org_expr = expr_list[0]
    org_vars = sorted(org_expr.vars_)
    nvars = len(org_vars)

    # 원함수가 상수인 경우, 따로 그리지 않음.
    # 최대 3차원의 그래프를 그릴 수 있음.
    if not 0 < nvars < 3:
        print(f"Warning: {org_expr.canonicalize()} has {nvars} variables, skipping graph...")
        return plt.figure(figsize=(6,4))
    
    tol = 1e-10
    fig = None
    ax = None
    
    for expr in expr_list:
        cur_vars = sorted(expr.vars_)
        if len(cur_vars) != nvars:
            print(f"Warning: {expr.canonicalize()} has different number of variables ({cur_vars}), skipping graph...")
            continue
        
        if len(cur_vars) == 1:
            var = cur_vars[0]
            x_vals, y_vals = sample_1d(expr, var, n_samples=10000, tol=tol) # sample_1d(expr, var, n_samples=100, tol=1e-10)

            if fig is None:
                fig, ax = plt.subplots(figsize=(6,4))
            else:
                ax = fig.gca()
            ax.plot(x_vals, y_vals, label=f"${expr.canonicalize()}$")
            ax.set_xlabel(var)
            ax.set_ylabel(f"f({var})")

        elif len(cur_vars) == 2:
            v1, v2 = cur_vars
            X, Y, Z = sample_2d(expr, v1, v2, n_samples=100, tol=tol)

            if fig is None:
                fig, ax = plt.subplots(figsize=(6,4))
                ax = fig.add_subplot(111, projection='3d')
            else:
                if not fig.axes:
                    ax = fig.add_subplot(111, projection='3d')
                else:
                    ax = fig.gca()
            
            surf = ax.plot_surface(X, Y, Z, alpha=0.5)
            # 3D 플롯의 경우 빈 프록시 아티스트를 사용하여 레전드 생성
            proxy = plt.Rectangle((0, 0), 1, 1, fc=surf._facecolors[0], alpha=0.5)
            if not hasattr(ax, '_proxy_artists'):
                ax._proxy_artists = []
                ax._proxy_labels = []
            ax._proxy_artists.append(proxy)
            ax._proxy_labels.append(f"${expr.canonicalize()}$")
            
            ax.set_xlabel(v1)
            ax.set_ylabel(v2)
            ax.set_zlabel(f"f({v1}, {v2})")

    if ax is not None:
        if hasattr(ax, 'lines') and len(ax.lines) > 0:
            ax.legend()  # 2D 플롯의 경우
        elif hasattr(ax, '_proxy_artists'):  # 3D 플롯의 경우
            ax.legend(ax._proxy_artists, ax._proxy_labels)
    
    return fig

def plot_graph_gradio(history_state):
    """
    state_ast 가 변할 때마다 새로운 그래프를 그리도록 호출될 함수.
    history_state : 원함수와 도함수 ast를 담은 리스트 (원함수 한 개만 있을수도 있음.)
    """
    if len(history_state) <= 1:
        plt.close('all')

    if not history_state:
        return plt.figure(figsize=(6,4))
    fig = create_figure_from_expr(history_state)
    return fig

def flatten_expressions(expr_list: list[Expression]) -> list[Expression]:
    """
    expressions들의 리스트를 받아, 내부에 UnionExpression 객체(gradient 등)가 있는 경우, 각 항을 모아서 리스트로 반환
    e.g. UnionExpression(2y, 2x) -> [2y, 2x]
    """
    flattened = []
    for expr in expr_list:
        if isinstance(expr, UnionExpression):
            flattened.extend(expr.args)
        else:
            flattened.append(expr)
    return flattened


def process_expression(expression):
    expression = expression.strip()
    if not expression:
        return "Error: Empty expression", None, None
    try:
        result, ast = calculate_expression(expression, verbose=False)
        # canonicalized 결과를 LaTeX 형식으로 감싸서 출력
        output = f"$$ {result} $$"
        if ast._domain_str():
            output += f"\n\n도메인<br>{ast._domain_str().replace('\n', '<br>')}"
        output += f"\n\n**표현식**<br>{ast.__repr__().replace('\n', '<br>').replace(' ', '&nbsp;&nbsp;')}"
        variables = collect_var_names(result)
        return output, ast, variables
    except Exception as e:
        return f"Error: {str(e)}", None, None
    
def evaluate_value_gradio(ast: Expression, value_table: gr.DataFrame):
    if not ast:
        return "No valid AST provided.", ast
    
    new_value_dict = {}
    for _, row in value_table.iterrows():
        if row.iloc[1]:
            assert ast.domain[row['Variable']].contains(Decimal(row.iloc[1])), f"Value {row.iloc[1]} is not in the domain of {row['Variable']}"
            new_value_dict[row['Variable']] = Decimal(row.iloc[1])
    result = ast.evaluate(new_value_dict)
    output = f"$$ {result} $$"

    return output

def evaluate_range_gradio(ast, history_state, range_table):
    """
    range_table에서 변수 범위 업데이트
    """
    new_domain = {}
    for _, row in range_table.iterrows():
        if row['Range']:
            new_domain[row['Variable']] = Interval.parse(row['Range'], row['Variable'])
    
    for expr in history_state:
        for var in new_domain.keys():
            if isinstance(expr, UnionExpression):
                for arg_idx in range(len(expr.args)):
                    if var in expr.args[arg_idx].domain:
                        expr.args[arg_idx].domain[var] = expr.args[arg_idx]._intrinsic_domain[var].intersects(new_domain[var])
            else:
                expr.domain[var] = expr._intrinsic_domain[var].intersects(new_domain[var])

    ast = history_state[-1]
    
    result = ast.evaluate()
    output = f"$$ {result} $$"
    if ast._domain_str():
        output += f"\n\n**도메인**<br>{ast._domain_str().replace('\n', '<br>')}"
    return output, ast, history_state

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

def differentiate_var_gradio(ast, var_table):
    # history
    selected_vars = [row['Variable'] for _, row in var_table.iterrows() if row['Selected'].lower() == 'o']
    if not selected_vars:
        raise ValueError("Error: 미분할 변수를 선택해주세요.")
    # 두 개 이상의 변수가 선택된 경우
    if len(selected_vars) > 1:
        raise ValueError("Error: 한 번에 하나의 변수만 선택해주세요.")
    
    var_choice = selected_vars[0]

    ast = ast.derivative(var_choice)
    output = f"(derivative with respect to {var_choice}): $$ {ast.canonicalize()} $$"
    return output, ast, collect_var_names(ast)

def differentiate_direction_gradio(ast, direction_table):
    direction = {}
    for _, row in direction_table.iterrows():
        if row.iloc[1]:
            direction[row['Variable']] = Decimal(row.iloc[1])
    ast = ast.directional_derivative(direction)
    output = f"(directional derivative):\n\n$$ {ast.canonicalize()} $$"
    return output, ast, collect_var_names(ast)

def differentiate_gradient_gradio(ast):
    ast = ast.gradient()
    output = f"(gradient):\n\n$$ {ast.canonicalize()} $$"
    return output, ast, collect_var_names(ast)


# --- 데모에 출력되는 테이블/마크다운 업데이트 함수들 ---

def init_current_expr():
    return [], ""

def get_history_markdown(history):
    markdown = "<br/>".join([f"$$ {expr} $$" for expr in history])
    return markdown

def update_history(history, ast):
    # state_ast 가 업데이트 됨 - history에 추가, history_markdown에 반영
    if ast and (not history or history[-1] != ast.canonicalize()):
        new_history = history + [ast.canonicalize()]
        new_markdown = get_history_markdown(new_history)
        return new_history, new_markdown
    else:
        return [], ""

def update_table_visibility(choice):
    if choice == "값":
        return gr.update(visible=True), gr.update(visible=False)
    else:  # choice == "구간"
        return gr.update(visible=False), gr.update(visible=True)


def update_variable_table(vars_list):
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
        table = [[var, ""] for var in vars_list]
        return gr.update(value=table, visible=True)
    else:
        return gr.update(visible=False)

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


# --- UI 구성 ---

global iface
with gr.Blocks() as demo:
    gr.Markdown("# 간단 계산기 GUI (Gradio)")

    # === 첫번째 섹션: 수식 입력 및 결과 출력 (항상 보임) ===
    with gr.Row():
        with gr.Column():
            expr_input = gr.Textbox(label="수식 입력", placeholder="예: 2*x + 3")
            submit_expr = gr.Button("수식 평가")
        with gr.Column():
            gr.Markdown("출력 결과")
            output_text = gr.Markdown()
        with gr.Column():
            gr.Markdown("히스토리")
            history_state = gr.State([])
            history_markdown = gr.Markdown()
        with gr.Column():
            plot_output = gr.Plot()
            # plot_output_range = gr.Dataframe(label="그래프 범위 입력", headers=["Variable", "Range"], visible=False)
            # plot_output_range_button = gr.Button("그래프 범위 입력")
    # 상태: AST와 변수 목록 저장
    state_ast = gr.State()
    state_vars = gr.State()
    submit_expr.click(
        fn=process_expression, inputs=expr_input, 
        outputs=[output_text, state_ast, state_vars]
    ).then(
        fn=init_current_expr, outputs=[history_state, history_markdown]
    ).then(
        fn=update_history, inputs=[history_state, state_ast], outputs=[history_state, history_markdown]
    )
    
    with gr.Tabs():
        with gr.TabItem("변수 값/범위 업데이트"):
            with gr.Row():
                with gr.Column():
                    value_table = gr.Dataframe(headers=["Variable", "Value (Int or Float only)"], visible=True)
                with gr.Column():
                    update_value_btn = gr.Button("Assign Value")

                with gr.Column():
                    range_table = gr.Dataframe(headers=["Variable", "Range"], visible=True)
                with gr.Column():
                    update_range_btn = gr.Button("Change Range")
        
        with gr.TabItem("연속성/미분가능성 검사"):
            with gr.Row():
                with gr.Column():
                    check_cont_input_type = gr.Radio(choices=["값", "구간"], label="연속성 검사", value="값")
                    check_cont_value_table = gr.Dataframe(label="연속성 검사 (값 입력)", headers=["Variable", "Value (Int or Float only)"], visible=True)
                    check_cont_range_table = gr.Dataframe(label="연속성 검사 (구간 입력)", headers=["Variable", "Range"], visible=False)
                    check_cont_button = gr.Button("연속성 검사")
                with gr.Column():
                    check_diff_input_type = gr.Radio(choices=["값", "구간"], label="미분가능성 검사", value="값")
                    check_diff_value_table = gr.Dataframe(label="미분가능성 검사 (값 입력)", headers=["Variable", "Value (Int or Float only)"], visible=True)
                    check_diff_range_table = gr.Dataframe(label="미분가능성 검사 (구간 입력)", headers=["Variable", "Range"], visible=False)
                    check_diff_button = gr.Button("미분가능성 검사")
        
        with gr.TabItem("미분"):
            with gr.Row():
                with gr.Column():
                    diff_var_table = gr.Dataframe(
                        label="편미분 변수 선택 (o 로 표시)", 
                        headers=["Variable", "Selected"],
                        visible=True
                    )
                    diff_var_button = gr.Button("편미분 실행")
                with gr.Column():
                    diff_direction_table = gr.Dataframe(
                        label="방향 벡터 입력", 
                        headers=["Variable", "Value"],
                        visible=True
                    )
                    diff_direction_button = gr.Button("방향 미분 실행")
                with gr.Column():
                    diff_gradient_table = gr.Dataframe(
                        label="기울기 계산", 
                        headers=["Variable", "-"],
                        visible=True
                    )
                    diff_gradient_button = gr.Button("기울기 구하기")

        
    history_state.change(fn=plot_graph_gradio, inputs=[history_state], outputs=plot_output)

    state_vars.change(
        fn=update_variable_table, inputs=[state_vars], outputs=[value_table, range_table]
    ).then(
        fn=update_cont_variable_table, inputs=[state_vars, check_cont_input_type], outputs=[check_cont_value_table, check_cont_range_table]
    ).then(
        fn=update_diff_variable_table, inputs=[state_vars, check_diff_input_type], outputs=[check_diff_value_table, check_diff_range_table]
    ).then(
        fn=update_diff_var_choice, inputs=[state_vars], outputs=[diff_var_table]
    ).then(
        fn=update_diff_dir_variable_table, inputs=[state_vars], outputs=[diff_direction_table]
    )

    update_value_btn.click(fn=evaluate_value_gradio, 
                           inputs=[state_ast, value_table], 
                           outputs=[output_text])
    update_range_btn.click(fn=evaluate_range_gradio, 
                           inputs=[state_ast, history_state, range_table], 
                           outputs=[output_text, state_ast, history_state]
    ).then( fn=create_figure_from_expr, inputs=[history_state], outputs=[plot_output] )
    
    check_cont_input_type.change(
        fn=update_table_visibility,
        inputs=check_cont_input_type,
        outputs=[check_cont_value_table, check_cont_range_table]
    )

    check_diff_input_type.change(
        fn=update_table_visibility,
        inputs=check_diff_input_type,
        outputs=[check_diff_value_table, check_diff_range_table]
    )

    check_cont_button.click(
        fn=check_continuity_gradio,
        inputs=[state_ast, state_vars, check_cont_input_type, check_cont_value_table, check_cont_range_table],
        outputs=[output_text, state_ast]
    )

    check_diff_button.click(
        fn=check_differentiability_gradio,
        inputs=[state_ast, state_vars, check_diff_input_type, check_diff_value_table, check_diff_range_table],
        outputs=[output_text, state_ast]
    )

    diff_var_button.click(
        fn=differentiate_var_gradio,
        inputs=[state_ast, diff_var_table],
        outputs=[output_text, state_ast, state_vars]
    ).then(fn=update_history, inputs=[history_state, state_ast], outputs=[history_state, history_markdown])

    diff_direction_button.click(
        fn=differentiate_direction_gradio,
        inputs=[state_ast, diff_direction_table],
        outputs=[output_text, state_ast, state_vars]
    ).then(fn=update_history, inputs=[history_state, state_ast], outputs=[history_state, history_markdown])

    diff_gradient_button.click(
        fn=differentiate_gradient_gradio,
        inputs=[state_ast],
        outputs=[output_text, state_ast, state_vars]
    ).then(fn=update_history, inputs=[history_state, state_ast], outputs=[history_state, history_markdown])
    
iface = demo.launch(show_error=True)
