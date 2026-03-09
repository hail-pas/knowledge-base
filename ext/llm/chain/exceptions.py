"""
Chain 模块异常定义

提供 Chain、Agent、Tool 相关的自定义异常
"""


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


__all__ = [
    "ChainError",
    "AgentError",
    "MaxIterationsError",
    "ToolExecutionError",
    "ToolError",
    "ToolNotFoundError",
]
