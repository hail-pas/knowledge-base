"""
Memory 实现

提供对话历史的存储和管理能力，支持内存和数据库持久化
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from loguru import logger

from ext.llm.types import ChatMessage


class BaseMemory(ABC):
    """Memory 抽象基类

    定义对话历史的存储和检索接口
    """

    @abstractmethod
    async def load_memory_variables(self) -> Dict[str, Any]:
        """加载记忆变量

        Returns:
            包含记忆的变量字典
        """
        pass

    @abstractmethod
    async def save_context(self, input: str, output: str) -> None:
        """保存上下文到记忆

        Args:
            input: 用户输入
            output: Assistant 输出
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """清空记忆"""
        pass


class InMemoryMemory(BaseMemory):
    """内存记忆

    使用 Python 列表存储对话历史
    适用于短期会话
    """

    def __init__(self, max_messages: int = 100):
        """初始化内存记忆

        Args:
            max_messages: 最大消息数量
        """
        self.max_messages = max_messages
        self.messages: List[ChatMessage] = []

    async def load_memory_variables(self) -> Dict[str, Any]:
        """加载消息历史

        Returns:
            包含消息列表的字典
        """
        logger.debug(f"InMemoryMemory load - messages: {len(self.messages)}")
        return {"messages": list(self.messages)}

    async def save_context(self, input: str, output: str) -> None:
        """保存对话上下文

        Args:
            input: 用户输入
            output: Assistant 输出
        """
        logger.debug(f"InMemoryMemory save - input length: {len(input)}, output length: {len(output)}")
        self.messages.append(ChatMessage(role="user", content=input))
        self.messages.append(ChatMessage(role="assistant", content=output))

        # 限制消息数量
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]
            logger.debug(f"InMemoryMemory trimmed messages to {len(self.messages)}")

    async def clear(self) -> None:
        """清空所有消息"""
        count = len(self.messages)
        self.messages.clear()
        logger.info(f"InMemoryMemory cleared {count} messages")

    def add_message(self, message: ChatMessage) -> None:
        """添加单条消息

        Args:
            message: 聊天消息
        """
        self.messages.append(message)

        # 限制消息数量
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]


class DatabaseMemory(BaseMemory):
    """数据库持久化记忆

    将对话历史保存到数据库
    支持跨会话恢复
    """

    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        agent_type: str = "default",
        uid: Optional[str] = None,
    ):
        """初始化数据库记忆

        Args:
            session_id: 会话 ID
            user_id: 用户 ID（可选）
            agent_type: Agent 类型
            uid: 会话唯一标识（可选，自动生成）
        """
        self.session_id = session_id
        self.user_id = user_id
        self.agent_type = agent_type
        self.uid = uid or str(uuid4())

    async def _get_or_create_record(self):
        """获取或创建数据库记录"""
        from ext.ext_tortoise.models.knowledge_base import ConversationMemory

        record = await ConversationMemory.filter(uid=self.uid).first()

        if not record:
            record = await ConversationMemory.create(
                uid=self.uid,
                session_id=self.session_id,
                user_id=self.user_id,
                agent_type=self.agent_type,
                messages=[],
                metadata={},
            )

        return record

    async def load_memory_variables(self) -> Dict[str, Any]:
        """从数据库加载消息历史

        Returns:
            包含消息列表的字典
        """
        from ext.ext_tortoise.models.knowledge_base import ConversationMemory
        from .exceptions import MemoryLoadError

        logger.debug(f"DatabaseMemory load - uid: {self.uid}, session_id: {self.session_id}")

        try:
            record = await ConversationMemory.filter(uid=self.uid).first()

            if not record:
                logger.debug(f"DatabaseMemory no record found for uid: {self.uid}")
                return {"messages": []}

            # 将 JSON 数据转换为 ChatMessage 对象
            messages = []
            for msg_data in record.messages or []:
                messages.append(ChatMessage(**msg_data))

            logger.debug(f"DatabaseMemory loaded {len(messages)} messages")
            return {"messages": messages}
        except Exception as e:
            logger.error(f"DatabaseMemory load error: {e}")
            raise MemoryLoadError("Failed to load memory from database", e)

    async def save_context(self, input: str, output: str) -> None:
        """保存对话上下文到数据库

        Args:
            input: 用户输入
            output: Assistant 输出
        """
        from ext.ext_tortoise.models.knowledge_base import ConversationMemory
        from .exceptions import MemorySaveError

        logger.debug(f"DatabaseMemory save - uid: {self.uid}, input length: {len(input)}, output length: {len(output)}")

        try:
            record = await self._get_or_create_record()

            # 添加新消息
            messages = record.messages or []
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})

            # 更新记录
            record.messages = messages
            record.last_updated = datetime.now()
            await record.save()

            logger.debug(f"DatabaseMemory saved - total messages: {len(messages)}")
        except Exception as e:
            logger.error(f"DatabaseMemory save error: {e}")
            raise MemorySaveError("Failed to save memory to database", e)

    async def clear(self) -> None:
        """清空所有消息"""
        from ext.ext_tortoise.models.knowledge_base import ConversationMemory
        from .exceptions import MemorySaveError

        logger.info(f"DatabaseMemory clear - uid: {self.uid}")

        try:
            record = await ConversationMemory.filter(uid=self.uid).first()

            if record:
                count = len(record.messages or [])
                record.messages = []
                record.last_updated = datetime.now()
                await record.save()
                logger.info(f"DatabaseMemory cleared {count} messages")
        except Exception as e:
            logger.error(f"DatabaseMemory clear error: {e}")
            raise MemorySaveError("Failed to clear memory", e)


class ConversationBufferMemory(BaseMemory):
    """对话缓冲记忆

    保留最近的 N 条消息，基于 Token 数量限制
    """

    def __init__(self, max_token_limit: int = 2000):
        """初始化对话缓冲记忆

        Args:
            max_token_limit: 最大 Token 数量
        """
        self.max_token_limit = max_token_limit
        self.messages: List[ChatMessage] = []

    async def load_memory_variables(self) -> Dict[str, Any]:
        """加载消息历史（自动裁剪超出限制的部分）

        Returns:
            包含消息列表的字典
        """
        logger.debug(f"ConversationBufferMemory load - messages: {len(self.messages)}, tokens: {self._count_tokens()}")
        return {"messages": list(self.messages)}

    async def save_context(self, input: str, output: str) -> None:
        """保存对话上下文（自动裁剪超出限制的部分）

        Args:
            input: 用户输入
            output: Assistant 输出
        """
        self.messages.append(ChatMessage(role="user", content=input))
        self.messages.append(ChatMessage(role="assistant", content=output))

        # 裁剪超出 Token 限制的消息
        while self._count_tokens() > self.max_token_limit:
            self.messages.pop(0)

        logger.debug(f"ConversationBufferMemory save - messages: {len(self.messages)}, tokens: {self._count_tokens()}")

    async def clear(self) -> None:
        """清空所有消息"""
        count = len(self.messages)
        self.messages.clear()
        logger.info(f"ConversationBufferMemory cleared {count} messages")

    def _count_tokens(self) -> int:
        """计算消息总 Token 数量

        Returns:
            Token 数量（近似值，使用字符数 / 4）
        """
        total_chars = sum(len(msg.content) for msg in self.messages)
        return total_chars // 4


__all__ = [
    "BaseMemory",
    "InMemoryMemory",
    "DatabaseMemory",
    "ConversationBufferMemory",
]
