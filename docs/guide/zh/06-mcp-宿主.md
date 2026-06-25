# MCP 宿主

`guanlan mcp` 是可选叠加层:一个**只读 MCP 服务端**(stdio),把 wiki 的检索/读页/图谱/体检/问答暴露为工具,给任意 MCP 客户端(Claude Code / Codex / Cursor …)使用。它是 P4 宿主层的**第二种传输**(stdio,与 Web 并列)。

> 前置:`pip install 'guanlan-wiki[mcp]'`(带入官方 `mcp` SDK + anyio)。未装时 `guanlan mcp` 优雅降级、提示安装(退出码 `1`)。

## 启动

通常**不直接手跑**——由调用方 Agent 作子进程拉起。手动验证:

```bash
guanlan -C my-wiki mcp                 # 在 stdio 上起只读 MCP 服务端
guanlan -C my-wiki mcp --model <id>    # 覆盖 ask 工具的模型(仅 ask 用)
```

| 参数 | 说明 |
|---|---|
| `--model` | 覆盖 `ask` 工具的 Agentao 模型(**仅 `ask` 用**;其余六个工具零 LLM) |

## 在客户端注册

在 MCP 客户端配置里把它登记为一个 stdio server,例如:

```jsonc
{
  "mcpServers": {
    "guanlan": { "command": "guanlan", "args": ["-C", "my-wiki", "mcp"] }
  }
}
```

## 八个工具

| 工具 | LLM? | 说明 |
|---|---|---|
| `search` | 否 | 整页召回(复用 `guanlan search` 内核) |
| `read_page` | 否 | 读一篇 wiki 页(带路径穿越防护) |
| `list_pages` | 否 | 列出内容页 |
| `graph` | 否 | 图谱(节点/边/社区/拓扑统计) |
| `health` | 否 | 体检报告 |
| `lint` | 否 | lint 报告 |
| `ask` | **是** | 对知识库提问(复用 CLI-query 只读子进程路径) |
| `deposit` | 否 | 把 agent 上下文暂存到 `workspace/staging/`（不碰知识库本体） |

七个零-LLM 工具复用与 Web 读端点相同的内核;`ask` 走只读子进程(故 P4-8 嵌入坑不适用)。
`deposit` 是唯一写工具——但它写的是暂存区（scratch），不是知识库本体（`raw/`/`wiki/`），不违反 P4.10 的只读契约。

### 运行时选择

`ask` 工具需要 Agent 运行时。通过环境变量 `GUANLAN_RUNTIME` 选择（默认 `agentao`；设 `openai` 走 OpenAI SDK，不依赖 agentao）：

```bash
# Agentao（默认）
guanlan -C my-wiki mcp --model <id>

# OpenAI SDK（需 openai 包 + OPENAI_API_KEY）
GUANLAN_RUNTIME=openai guanlan -C my-wiki mcp --model gpt-4o-mini
```

两种运行时共享同一套硬约束注入和 raw 快照保护。

## 设计要点

- **零写契约**:镜像 `--reader` 的零字节 KB 写入姿态——MCP **不做 convert**(写 `raw/` 与只读姿态冲突)。
- **不碰服务端会话状态**:MCP 客户端自持会话,故宿主无需服务端 session/conversation 状态(相比 Web 的最大减法)。
- **方向区别于 DESIGN 的「Tool 注入」**:这里是**观澜作 MCP 服务端**(把 wiki 暴露出去);DESIGN 的 Tool 注入是反方向——**Agentao 作 MCP 客户端**消费外部工具。
- 是 **E2「远程 / scoped MCP」的本地只读前哨**。

参见:仓库 [`docs/P4.10-MCP宿主.md`](../../P4.10-MCP宿主.md)。
