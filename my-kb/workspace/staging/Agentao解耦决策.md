# Agentao 解耦决策

## 背景
观澜需要解耦对 Agentao 运行时的硬耦合。

## 方案
抽 AgentRuntime 协议，补齐只读路径 raw 快照 + 硬约束注入。