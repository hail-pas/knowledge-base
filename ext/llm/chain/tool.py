"""
Tool 实现和装饰器

提供将 Python 函数转换为 Tool 的能力
"""

import inspect
import json
from typing import Any, Dict, List, Optional, Union, get_origin, get_args
from collections.abc import Callable
from loguru import logger

from ext.llm.types import ToolDefinition, FunctionDefinition
from util.general import truncate_content


class Tool:
    """工具抽象

    将 Python 函数封装为 LLM 可调用的工具
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ):
        """初始化 Tool

        Args:
            func: 工具函数（可以是同步或异步）
            name: 工具名称
            description: 工具描述
            parameters: 参数定义（JSON Schema）
        """
        self.func = func
        self.name = name
        self.description = description
        self.parameters = parameters
        self.is_async = inspect.iscoroutinefunction(func)

    def to_definition(self) -> ToolDefinition:
        """转换为 ToolDefinition

        Returns:
            ToolDefinition 实例
        """
        return ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
            ),
        )

    async def ainvoke(self, **kwargs: Any) -> Any:
        """调用工具（支持同步和异步）

        Args:
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        logger.debug(f"Tool '{self.name}' invoke - params: {list(kwargs.keys())}")

        # 验证必需参数
        required = self.parameters.get("required", [])
        missing_params = [p for p in required if p not in kwargs]
        if missing_params:
            logger.error(f"Tool '{self.name}' missing required parameters: {missing_params}")
            raise ValueError(f"Missing required parameters: {missing_params}")

        try:
            if self.is_async:
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)

            logger.debug(f"Tool '{self.name}' result: {truncate_content(str(result))}")
            return result
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            raise RuntimeError(f"Tool '{self.name}' execution failed: {e}")

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', description='{self.description}')"


def tool(func: Callable | None = None, *, name: str | None = None, description: str | None = None):
    """装饰器：将函数转换为 Tool

    使用方式：
        @tool
        def my_function(param: str) -> str:
            return param

    或者：
        @tool(name="custom_name", description="Custom description")
        def my_function(param: str) -> str:
            return param

    Args:
        func: 被装饰的函数
        name: 自定义工具名称（可选，默认使用函数名）
        description: 自定义工具描述（可选，默认使用函数文档字符串）

    Returns:
        Tool 实例或装饰器函数
    """

    def decorator(f: Callable) -> Tool:
        tool_name = name or f.__name__
        tool_description = description or (f.__doc__ or "").strip()
        parameters = extract_parameters_from_signature(f)

        return Tool(
            func=f,
            name=tool_name,
            description=tool_description,
            parameters=parameters,
        )

    if func is not None:
        return decorator(func)
    return decorator


def extract_parameters_from_signature(func: Callable) -> dict[str, Any]:
    """从函数签名提取参数定义

    Args:
        func: 函数对象

    Returns:
        JSON Schema 格式的参数定义
    """
    sig = inspect.signature(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # 跳过 self 参数
        if param_name == "self":
            continue

        # 获取参数类型
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else "string"
        param_type_str = _get_type_string(param_type)

        # 获取默认值
        has_default = param.default != inspect.Parameter.empty

        # 构建属性定义
        prop_def: dict[str, Any] = {
            "type": param_type_str,
        }

        # 添加描述（从类型提示或默认值推断）
        if param.annotation != inspect.Parameter.empty:
            if hasattr(param.annotation, "__doc__"):
                prop_def["description"] = param.annotation.__doc__

        properties[param_name] = prop_def

        # 记录必需参数
        if not has_default and param_name != "self":
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _get_type_string(type_annotation: Any) -> str:
    """将类型注解转换为 JSON Schema 类型字符串

    Args:
        type_annotation: 类型注解

    Returns:
        JSON Schema 类型字符串
    """
    # 处理 Optional/Union
    origin = get_origin(type_annotation)

    if origin is Union:
        args = get_args(type_annotation)
        return _get_type_string(args[0])

    # 基本类型映射
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    if type_annotation in type_map:
        return type_map[type_annotation]

    # 处理 List
    if origin is list:
        return "array"

    # 处理 Dict
    if origin is dict:
        return "object"

    # 默认返回 string
    return "string"


def validate_json_schema(parameters: dict[str, Any]) -> bool:
    """验证 JSON Schema 是否有效

    Args:
        parameters: 参数定义

    Returns:
        是否有效
    """
    required_fields = ["type", "properties"]
    for field in required_fields:
        if field not in parameters:
            return False

    if parameters["type"] != "object":
        return False

    if not isinstance(parameters["properties"], dict):
        return False

    return True


__all__ = [
    "Tool",
    "tool",
    "extract_parameters_from_signature",
    "validate_json_schema",
]
