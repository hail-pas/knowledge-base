"""
Chain 模块基础抽象

定义统一的 Runnable 接口和基础类
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Generic, List, TypeVar

from ext.llm.chain.exceptions import ChainError

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Runnable(Generic[InputT, OutputT], ABC):
    """统一的可运行接口

    所有 Chain 组件都需要实现此接口，提供统一的调用方式
    """

    @abstractmethod
    async def ainvoke(self, input: InputT) -> OutputT:
        """异步调用

        Args:
            input: 输入数据

        Returns:
            输出数据

        Raises:
            ChainError: 执行失败时抛出
        """
        pass

    async def abatch(self, inputs: List[InputT]) -> List[OutputT]:
        """批量调用

        Args:
            inputs: 输入数据列表

        Returns:
            输出数据列表
        """
        return [await self.ainvoke(inp) for inp in inputs]

    async def astream(self, input: InputT) -> AsyncIterator[OutputT]:
        """流式输出

        默认实现是同步调用，子类可以覆盖以提供真正的流式输出

        Args:
            input: 输入数据

        Yields:
            输出数据流
        """
        result = await self.ainvoke(input)
        yield result

    def __or__(self, other: "Runnable") -> "Runnable":
        """pipe 操作符支持

        允许使用 | 操作符组合多个 Runnable：

            chain = prompt | llm | parser

        Args:
            other: 下一个 Runnable

        Returns:
            组合后的 Runnable
        """
        from ext.llm.chain.chain import RunnableSequence

        return RunnableSequence([self, other])


class RunnablePassthrough(Runnable[InputT, InputT]):
    """透传 Runnable

    不做任何转换，直接返回输入数据
    用于在 Chain 中传递数据到多个地方
    """

    async def ainvoke(self, input: InputT) -> InputT:
        """直接返回输入"""
        return input


__all__ = [
    "Runnable",
    "RunnablePassthrough",
]
