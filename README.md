# 🚀 职能沟通翻译助手 (Functional Communication Translation Assistant)

这是一个基于 AI 的智能沟通辅助工具，旨在解决企业协作中**产品经理 (PM)** 与 **开发工程师 (Dev)** 之间的“语言隔阂”。

系统采用 **Plan-and-Execute (规划与执行)** 架构，能够识别用户身份和输入内容的场景（需求讨论 vs 技术方案），自动进行任务拆解、缺失信息检查（Gap Analysis），并生成对方能听懂的高质量翻译。

## ✨ 核心功能

* **👥 角色化登录**：支持 PM 和 Dev 两种视角，界面与提示词策略自动适配。
* **🧠 智能规划 (Planner)**：
    * **场景识别**：自动判断输入是“业务需求”还是“技术细节”。
    * **缺漏分析 (Gap Analysis)**：主动发现逻辑漏洞（如 PM 没提并发量、Dev 没提 ROI），并生成追问。
    * **策略拆解**：将复杂问题拆解为 3-4 个逻辑步骤。
* **✍️ 深度执行 (Executor)**：基于规划蓝图，生成详细的 Markdown 格式分析报告。
* **💬 上下文记忆**：支持多轮对话，AI 会记住之前的沟通背景，支持“修改上一条”、“再详细点”等指令。
* **📜 增量式聊天**：对话记录自动追加，提供流畅的聊天体验。

## 🛠️ 技术栈

* **前端框架**: [Dash]
* **大模型编排**: [LangChain](https://www.langchain.com/) (Core / Community)
* **本地大模型**: [Ollama](https://ollama.com/) (模型: `qwen2.5:14b`)
* **架构模式**: Plan-and-Execute Agent (Planner -> Executor Pipeline)

---

## 🏃‍♂️ 如何运行

### 1. 环境准备

确保你已经安装了 Python 3.9+ 以及 Ollama。

**启动 Ollama 服务：**
确保你的本地或远程 Ollama 服务已启动，并且已经下载了 `qwen2.5:14b` 模型。
```bash
ollama run qwen2.5:14b
