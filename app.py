import os
import json
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc

# --- LangChain 核心组件 ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, messages_to_dict, messages_from_dict
from typing import List

# ==========================================
# 0. 配置与初始化
# ==========================================

# 规划器 LLM
planner_llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0.1,
    format="json",
    base_url="http://192.168.0.102:11434"
)

# 执行器 LLM
executor_llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0.7,
    base_url="http://192.168.0.102:11434"
)


# ==========================================
# 1. 数据模型与辅助函数
# ==========================================

class Step(BaseModel):
    id: int = Field(..., description="步骤的ID")
    description: str = Field(..., description="步骤的描述")


class Plan(BaseModel):
    steps: List[Step] = Field(default_factory=list, description="计划中的步骤列表")


def render_chat_ui(history_list):
    """将历史消息渲染为 Markdown 格式"""
    if not history_list:
        return "👋 欢迎！请在左侧输入需求，我会结合上下文为您规划和翻译..."

    md_output = []
    for msg in history_list:
        role = msg.get('type')
        content = msg.get('data', {}).get('content', '')

        if role == 'human':
            md_output.append(f"\n> 👤 **User**: {content}\n")
        elif role == 'ai':
            md_output.append(f"\n🤖 **AI Translation**: \n\n{content}\n\n---\n")

    return "".join(md_output)


# ==========================================
# 2. 核心类定义 (Planner & Executor)
# ==========================================

class Planner:
    """规划器: 智能识别输入类型(需求vs技术)，并生成拆解步骤"""

    def __init__(self, llm_model):
        self.llm = llm_model
        # 【升级点】Prompt 增加了“场景识别”逻辑
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能的职能沟通任务规划师。
            你的核心能力是**自动识别用户输入的内容属于哪种场景**，并制定对应的翻译/分析策略。

            请先在内心分析用户的输入属于以下哪类，然后生成对应的步骤：

            🔴 **场景 A：需求讨论 (Requirement Mode)**
            - **识别特征**：输入包含"我们需要..."、"用户想要..."、"增加一个功能"、"提升转化率"等业务语言。
            - **执行策略**：将业务需求拆解为技术实现。
            - **步骤模板**：[核心技术架构选型] -> [数据存储与流转设计] -> [API性能与实时性要求] -> [开发难点与工时预估]。

            🔵 **场景 B：技术方案 (Technical Solution Mode)**
            - **识别特征**：输入包含"Redis"、"微服务"、"QPS"、"重构"、"数据库"、"算法模型"等技术术语。
            - **执行策略**：将技术细节翻译为商业价值。
            - **步骤模板**：[用户体验的直接改善] -> [对业务增长/留存的支撑] -> [长期商业价值/竞争力] -> [成本效益(ROI)分析]。

            【通用规则】：
            1. 请忽略用户的登录身份，优先依据**输入内容**来决定策略。
            2. 如果有【历史对话】，且用户指令是"继续"、"详细点"，请延续上一轮的策略。

            请严格只输出 JSON 格式，不要包含 Markdown 标记：
            {{
                "steps": [
                    {{"id": 1, "description": "步骤具体内容..."}},
                    {{"id": 2, "description": "步骤具体内容..."}}
                ]
            }}
            """),
            ("human", "当前登录身份：{role}\n\n【历史对话记录】:\n{history}\n\n【用户当前输入】：{input}")
        ])
        self.chain = self.prompt | self.llm

    def plan(self, input_str: str, role: str, chat_history: InMemoryChatMessageHistory) -> Plan:
        # 保持原有逻辑不变
        history_messages = chat_history.messages[-6:]
        history_str = "\n".join([f"{m.type}: {m.content}" for m in history_messages])

        try:
            response = self.chain.invoke({"input": input_str, "role": role, "history": history_str})
            content = response.content.strip()
            # 清洗可能存在的 Markdown 标记
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            plan_data = json.loads(content)
            return Plan(**plan_data)
        except Exception as e:
            print(f"规划生成失败: {e}")
            return Plan(steps=[Step(id=1, description=f"智能分析输入内容: {input_str}")])

class Executor:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一名执行专家。请根据【当前步骤】的任务要求，结合【原始输入】，撰写该部分的详细分析内容。"),
            ("human", """
            【原始输入】：{original_input}
            【上下文】：{context}
            【当前步骤】：{step_description}
            请直接输出该步骤的分析结果(Markdown格式)：
            """)
        ])
        self.chain = self.prompt | self.llm

    def execute_step(self, original_input: str, step_description: str, context: str = '') -> str:
        response = self.chain.invoke({
            "original_input": original_input,
            "step_description": step_description,
            "context": context
        })
        return response.content


class PlanAndExecuteAgent:
    def __init__(self, planner: Planner, executor: Executor):
        self.planner = planner
        self.executor = executor

    def run(self, input_str: str, role: str, chat_history: InMemoryChatMessageHistory) -> str:
        plan = self.planner.plan(input_str, role, chat_history)

        context = ""
        final_output_buffer = [f"### 📋 本次翻译策略规划\n"]
        for step in plan.steps:
            final_output_buffer.append(f"- **Step {step.id}**: {step.description}")
        final_output_buffer.append("\n\n")

        for i, step in enumerate(plan.steps):
            step_result = self.executor.execute_step(input_str, step.description, context)
            context += f"\n【步骤 {step.id} 结果】:\n{step_result}\n"
            final_output_buffer.append(f"#### {step.description}\n{step_result}\n")

        return "\n".join(final_output_buffer)


# 实例化
planner_instance = Planner(planner_llm)
executor_instance = Executor(executor_llm)
agent_runner = PlanAndExecuteAgent(planner_instance, executor_instance)

# ==========================================
# 3. Dash 前端界面层
# ==========================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "职能沟通翻译助手"

CARD_STYLE = {"boxShadow": "0 4px 8px 0 rgba(0,0,0,0.2)", "borderRadius": "10px"}

# 登录布局
login_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("🚀 职能沟通翻译助手", className="text-center mb-5"),
            html.H4("请选择您的角色登录", className="text-center mb-4 text-muted"),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("我是产品经理", className="card-title text-center"),
                            html.P("Product Manager", className="text-center text-muted"),
                            html.Hr(),
                            dbc.Button("以 PM 身份登录", id={'type': 'auth-btn', 'action': 'login-pm'}, color="primary",
                                       className="w-100 mt-3")
                        ])
                    ], style=CARD_STYLE), width=6
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("我是开发工程师", className="card-title text-center"),
                            html.P("Software Engineer", className="text-center text-muted"),
                            html.Hr(),
                            dbc.Button("以 Dev 身份登录", id={'type': 'auth-btn', 'action': 'login-dev'},
                                       color="success", className="w-100 mt-3")
                        ])
                    ], style=CARD_STYLE), width=6
                )
            ])
        ], width=8)
    ], justify="center", className="mt-5")
], fluid=True)


# 工作台布局
def build_workspace(role):
    theme_color = "primary" if role == "PM" else "success"
    role_name = "产品经理 (PM)" if role == "PM" else "开发工程师 (Dev)"
    target_role = "开发视角" if role == "PM" else "产品视角"
    placeholder = "请输入需求... (我会记住之前的对话)"

    return dbc.Container([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink(f"当前身份: {role_name}", href="#", active=True)),
                dbc.Button("退出登录", id={'type': 'auth-btn', 'action': 'logout'}, color="light", size="sm",
                           className="ms-3")
            ],
            brand="职能沟通翻译助手",
            color=theme_color,
            dark=True,
            className="mb-4 rounded-bottom"
        ),
        dbc.Row([
            # 左侧：输入
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(f"📝 您的输入"),
                    dbc.CardBody([
                        dbc.Textarea(id="input-text", placeholder=placeholder, style={"height": "150px"}),
                        dbc.Button("✨ 发送消息 (Append)", id="btn-translate", color=theme_color, className="w-100 mt-3")
                    ])
                ], style=CARD_STYLE)
            ], width=4),

            # 右侧：聊天历史
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(f"💬 翻译对话流 -> {target_role}"),
                    dbc.CardBody([
                        dcc.Loading(
                            type="cube",
                            color="#119DFF",
                            children=[
                                # 默认显示欢迎语，无需通过 callback 清空
                                dcc.Markdown(
                                    id="output-text",
                                    children="👋 欢迎！请在左侧输入...",
                                    style={"height": "600px", "overflowY": "scroll"},
                                    dangerously_allow_html=True
                                )
                            ]
                        )
                    ])
                ], style=CARD_STYLE)
            ], width=8)
        ])
    ], fluid=True)


app.layout = html.Div([
    dcc.Store(id='user-role-store', storage_type='session'),
    dcc.Store(id='chat-history-store', storage_type='memory', data=[]),
    html.Div(id='page-content', children=login_layout)
])


# ==========================================
# 4. 回调函数
# ==========================================

# 修复后的登录回调：移除了 Output('output-text', ...)
@app.callback(
    Output('user-role-store', 'data'),
    Output('page-content', 'children'),
    Output('chat-history-store', 'data', allow_duplicate=True),
    Input({'type': 'auth-btn', 'action': ALL}, 'n_clicks'),
    State('user-role-store', 'data'),
    prevent_initial_call=True
)
def manage_login(n_clicks_list, current_data):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = ctx.triggered_id
    if not trigger_id or 'action' not in trigger_id:
        return dash.no_update, dash.no_update, dash.no_update

    action = trigger_id['action']

    if action == 'login-pm':
        # 登录时清空历史记录
        return {"role": "PM"}, build_workspace("PM"), []
    elif action == 'login-dev':
        return {"role": "Dev"}, build_workspace("Dev"), []
    elif action == 'logout':
        return None, login_layout, []

    return dash.no_update, dash.no_update, dash.no_update


# 翻译回调：保持不变
@app.callback(
    Output('output-text', 'children'),  # 更新页面显示
    Output('chat-history-store', 'data'),  # 更新后台存储
    Output('input-text', 'value'),  # 清空输入框
    Input('btn-translate', 'n_clicks'),
    State('input-text', 'value'),
    State('user-role-store', 'data'),
    State('chat-history-store', 'data'),
    prevent_initial_call=True
)
def process_translation(n_clicks, text, user_data, history_data):
    if not user_data or not text:
        return dash.no_update, dash.no_update, dash.no_update

    role = user_data.get("role")

    if history_data:
        try:
            loaded_msgs = messages_from_dict(history_data)
            chat_history = InMemoryChatMessageHistory(messages=loaded_msgs)
        except Exception:
            chat_history = InMemoryChatMessageHistory()
    else:
        chat_history = InMemoryChatMessageHistory()

    try:
        # 1. 记录用户输入
        chat_history.add_user_message(text)

        # 2. 运行 Agent
        final_report = agent_runner.run(text, role, chat_history)

        # 3. 记录 Agent 回复
        chat_history.add_ai_message(final_report)

        # 4. 序列化并渲染
        full_serialized_history = messages_to_dict(chat_history.messages)
        full_chat_markdown = render_chat_ui(full_serialized_history)

        return full_chat_markdown, full_serialized_history, ""

    except Exception as e:
        import traceback
        error_msg = f"执行出错: {str(e)}"
        return error_msg, dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run(debug=True, port=8050)