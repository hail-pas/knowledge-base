"""
Chain 模块

提供类似 LangChain 的 LLM 链式调用、Agent 和工具管理能力

核心特性：
- 统一的 Runnable 接口
- LLM 封装
- Prompt Template
- Tool 定义和执行
- Agent（Function Calling, ReAct）
- Memory 管理（内存和数据库持久化）
- Chain 组合（pipe 操作符）
- 输出解析器
"""

# 核心基础
from ext.llm.chain.base import Runnable, RunnablePassthrough

# LLM
from ext.llm.chain.llm import LLM

# Prompt
from ext.llm.chain.prompt import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)

# Tool
from ext.llm.chain.tool import Tool, tool

# Agent
from ext.llm.chain.agent import (
    Agent,
    AgentStream,
    FunctionCallingAgent,
    ReActAgent,
)

# Memory
from ext.llm.chain.memory import (
    BaseMemory,
    InMemoryMemory,
    DatabaseMemory,
    ConversationBufferMemory,
)

# Chain
from ext.llm.chain.chain import (
    RunnableSequence,
    RunnableBranch,
    RunnableMap,
)

# Output Parser
from ext.llm.chain.output_parser import (
    BaseOutputParser,
    StrOutputParser,
    JsonOutputParser,
)

# Exceptions
from ext.llm.chain.exceptions import (
    ChainError,
    AgentError,
    MaxIterationsError,
    ToolExecutionError,
    ToolError,
    ToolNotFoundError,
    MemoryError,
    MemoryLoadError,
    MemorySaveError,
)

__all__ = [
    # 核心
    "Runnable",
    "RunnablePassthrough",
    # LLM
    "LLM",
    # Prompt
    "PromptTemplate",
    "ChatPromptTemplate",
    "MessagesPlaceholder",
    # Tool
    "Tool",
    "tool",
    # Agent
    "Agent",
    "AgentStream",
    "FunctionCallingAgent",
    "ReActAgent",
    # Memory
    "BaseMemory",
    "InMemoryMemory",
    "DatabaseMemory",
    "ConversationBufferMemory",
    # Chain
    "RunnableSequence",
    "RunnableBranch",
    "RunnableMap",
    # Output Parser
    "BaseOutputParser",
    "StrOutputParser",
    "JsonOutputParser",
    # Exceptions
    "ChainError",
    "AgentError",
    "MaxIterationsError",
    "ToolExecutionError",
    "ToolError",
    "ToolNotFoundError",
    "MemoryError",
    "MemoryLoadError",
    "MemorySaveError",
]
