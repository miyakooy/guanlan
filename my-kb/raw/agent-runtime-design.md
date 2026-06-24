# AgentRuntime 协议设计

## 概述

AgentRuntime 是观澜与具体 Agent 运行时之间的唯一接缝。它把对 Agentao 的硬耦合
收敛到一个 Protocol 后面，使引擎可适配任意 Agent 运行时。

## 核心接口

```python
@runtime_checkable
class AgentRuntime(Protocol):
    def run(self, request: AgentTaskRequest) -> AgentRunResult: ...
```

## 关键设计决策

1. AgentRuntime 只认"接口契约"，不认具体领域、不认具体 Agent
2. 行为硬约束由 wrapper 强制注入 prompt 头部，不依赖 Agent 自动读项目级指令文件
3. 只读路径的 raw 安全由快照兜底，不依赖运行时的 read-only 姿态
