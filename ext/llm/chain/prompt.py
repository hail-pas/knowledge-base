"""
Prompt Template 实现

提供灵活的提示词模板支持
"""

from typing import Any, Dict, List, Optional, Union
from loguru import logger

from ext.llm.types import ChatMessage
from ext.llm.chain.base import Runnable


class PromptTemplate(Runnable[dict[str, Any], str]):
    """提示词模板

    使用 Python f-string 风格的模板语法
    """

    def __init__(self, template: str, input_variables: list[str] | None = None):
        """初始化提示词模板

        Args:
            template: 模板字符串（f-string 风格）
            input_variables: 输入变量列表（可选，会自动提取）
        """
        self.template = template
        self.input_variables = input_variables or self._extract_variables(template)

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        """从字符串创建提示词模板

        Args:
            template: 模板字符串

        Returns:
            PromptTemplate 实例
        """
        return cls(template)

    def _extract_variables(self, template: str) -> list[str]:
        """从模板中提取变量名

        Args:
            template: 模板字符串

        Returns:
            变量名列表
        """
        import re

        # 匹配 {variable} 格式
        pattern = r"\{([^{}]+)\}"
        variables = re.findall(pattern, template)
        return list(set(variables))

    def format(self, **kwargs: Any) -> str:
        """格式化模板

        Args:
            **kwargs: 模板变量

        Returns:
            格式化后的字符串
        """
        # 验证输入变量
        missing_vars = set(self.input_variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing input variables: {missing_vars}")

        # 格式化模板
        return self.template.format(**kwargs)

    async def ainvoke(self, input: dict[str, Any]) -> str:
        """异步调用（格式化模板）

        Args:
            input: 输入变量字典

        Returns:
            格式化后的字符串
        """
        logger.debug(f"PromptTemplate ainvoke - variables: {list(input.keys())}")
        result = self.format(**input)
        logger.debug(f"PromptTemplate result length: {len(result)}")
        return result


class ChatPromptTemplate(Runnable[dict[str, Any], list[ChatMessage]]):
    """聊天提示词模板

    支持构建多消息的聊天提示词
    """

    def __init__(self, messages: list[Union[str, PromptTemplate, "MessagesPlaceholder"]]):
        """初始化聊天提示词模板

        Args:
            messages: 消息列表，可以是字符串、PromptTemplate 或 MessagesPlaceholder
        """
        self.messages = messages

    @classmethod
    def from_messages(cls, *messages: Union[str, PromptTemplate, "MessagesPlaceholder"]) -> "ChatPromptTemplate":
        """从消息列表创建聊天提示词模板

        Args:
            *messages: 消息列表

        Returns:
            ChatPromptTemplate 实例
        """
        return cls(list(messages))

    async def ainvoke(self, input: dict[str, Any]) -> list[ChatMessage]:
        """异步调用（构建聊天消息）

        Args:
            input: 输入变量字典

        Returns:
            聊天消息列表
        """
        logger.debug(
            f"ChatPromptTemplate ainvoke - template messages: {len(self.messages)}, variables: {list(input.keys())}",
        )

        result = []

        for msg in self.messages:
            if isinstance(msg, str):
                # 简单字符串，默认为 user 消息
                result.append(ChatMessage(role="user", content=msg))
            elif isinstance(msg, PromptTemplate):
                # PromptTemplate，默认为 user 消息
                result.append(ChatMessage(role="user", content=msg.format(**input)))
            elif isinstance(msg, MessagesPlaceholder):
                # MessagesPlaceholder，从输入中获取消息
                placeholder_messages = input.get(msg.variable_name, [])
                result.extend(placeholder_messages)
            else:
                # (role, content) 元组
                if isinstance(msg, tuple) and len(msg) == 2:
                    role, content = msg # type: ignore
                    if isinstance(content, str):
                        result.append(ChatMessage(role=role, content=content))
                    elif isinstance(content, PromptTemplate):
                        result.append(ChatMessage(role=role, content=content.format(**input)))
                    elif isinstance(content, MessagesPlaceholder):
                        placeholder_messages = input.get(content.variable_name, [])
                        result.extend(placeholder_messages)

        logger.debug(f"ChatPromptTemplate result - messages: {len(result)}")
        return result


class MessagesPlaceholder:
    """消息占位符

    用于在聊天提示词模板中插入动态消息列表
    """

    def __init__(self, variable_name: str):
        """初始化消息占位符

        Args:
            variable_name: 变量名
        """
        self.variable_name = variable_name

    def __repr__(self) -> str:
        return f"MessagesPlaceholder('{self.variable_name}')"


__all__ = [
    "PromptTemplate",
    "ChatPromptTemplate",
    "MessagesPlaceholder",
]
