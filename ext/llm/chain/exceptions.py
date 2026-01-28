"""
Chain 模块异常定义

提供 Chain、Agent、Tool、Memory 相关的自定义异常
"""

from typing import Optional


class ChainError(Exception):
    """Chain 基础异常"""



class AgentError(ChainError):
    """Agent 异常"""



class MaxIterationsError(AgentError):
    """达到最大迭代次数异常"""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        super().__init__(f"Agent reached maximum iterations ({max_iterations}) without completing")


class ToolExecutionError(AgentError):
    """工具执行异常"""

    def __init__(self, tool_name: str, error: Exception):
        self.tool_name = tool_name
        self.original_error = error
        super().__init__(f"Tool '{tool_name}' execution failed: {error}")


class ToolError(ChainError):
    """Tool 异常"""



class ToolNotFoundError(ToolError):
    """工具未找到异常"""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found")


class MemoryError(ChainError):
    """Memory 异常"""



class MemoryLoadError(MemoryError):
    """Memory 加载异常"""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


class MemorySaveError(MemoryError):
    """Memory 保存异常"""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


__all__ = [
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
