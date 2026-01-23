"""
Chain 组合实现

提供 Runnable 的组合能力，支持顺序执行和条件分支
"""

from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, TypeVar
from loguru import logger

from ext.llm.chain.base import Runnable
from ext.llm.chain.exceptions import ChainError

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class RunnableSequence(Runnable[InputT, OutputT]):
    """顺序执行 Chain

    将多个 Runnable 串联起来，按顺序执行
    """

    def __init__(self, steps: List[Runnable]):
        """初始化顺序执行 Chain

        Args:
            steps: Runnable 列表，按执行顺序排列
        """
        self.steps = steps

    async def ainvoke(self, input: InputT) -> OutputT:
        """顺序执行所有步骤

        Args:
            input: 输入数据

        Returns:
            最终输出数据
        """
        logger.debug(f"RunnableSequence ainvoke - steps: {len(self.steps)}")
        result: Any = input

        for idx, step in enumerate(self.steps):
            logger.debug(f"RunnableSequence step {idx + 1}/{len(self.steps)}: {step.__class__.__name__}")
            result = await step.ainvoke(result)

        logger.debug(f"RunnableSequence completed")
        return result

    async def astream(self, input: InputT) -> AsyncIterator[OutputT]:
        """流式执行

        如果最后一个步骤支持流式输出，则使用流式输出

        Args:
            input: 输入数据

        Yields:
            流式输出
        """
        logger.debug(f"RunnableSequence astream - steps: {len(self.steps)}")

        # 执行前面的步骤
        result: Any = input

        for idx, step in enumerate(self.steps[:-1]):
            logger.debug(
                f"RunnableSequence step {idx + 1}/{len(self.steps) - 1} (non-stream): {step.__class__.__name__}"
            )
            result = await step.ainvoke(result)

        # 最后一个步骤使用流式输出
        last_step = self.steps[-1]
        logger.debug(f"RunnableSequence step {len(self.steps)} (stream): {last_step.__class__.__name__}")

        chunk_count = 0
        async for chunk in last_step.astream(result):
            chunk_count += 1
            yield chunk

        logger.debug(f"RunnableSequence stream completed - total chunks: {chunk_count}")

    def __or__(self, other: Runnable) -> "RunnableSequence":
        """pipe 操作符

        允许继续追加步骤

        Args:
            other: 下一个 Runnable

        Returns:
            新的 RunnableSequence
        """
        return RunnableSequence(self.steps + [other])


class RunnableBranch(Runnable[InputT, OutputT]):
    """条件分支 Chain

    根据条件选择不同的执行路径
    """

    def __init__(self, branches: List[Tuple[Callable[[InputT], bool], Runnable[InputT, OutputT]]]):
        """初始化条件分支 Chain

        Args:
            branches: 分支列表，每个分支是一个元组 (条件函数, Runnable)
        """
        self.branches = branches

    async def ainvoke(self, input: InputT) -> OutputT:
        """根据条件选择分支执行

        Args:
            input: 输入数据

        Returns:
            输出数据
        """
        for condition, runnable in self.branches:
            if condition(input):
                return await runnable.ainvoke(input)

        # 如果没有匹配的条件，抛出异常
        raise ChainError("No matching branch found for input")

    @classmethod
    def from_conditions(
        cls,
        condition_runnable_pairs: List[Tuple[Callable[[InputT], bool], Runnable[InputT, OutputT]]],
        default: Optional[Runnable[InputT, OutputT]] = None,
    ) -> "RunnableBranch":
        """从条件对创建分支

        Args:
            condition_runnable_pairs: 条件和 Runnable 对列表
            default: 默认分支（可选）

        Returns:
            RunnableBranch 实例
        """
        branches = list(condition_runnable_pairs)

        if default is not None:
            branches.append((lambda x: True, default))

        return cls(branches)


class RunnableMap(Runnable[InputT, Dict[str, Any]]):
    """并行执行 Chain

    将输入传递到多个 Runnable 并行执行，返回字典
    """

    def __init__(self, mappings: Dict[str, Runnable]):
        """初始化并行执行 Chain

        Args:
            mappings: 键值对，键是输出键，值是对应的 Runnable
        """
        self.mappings = mappings

    async def ainvoke(self, input: InputT) -> Dict[str, Any]:
        """并行执行所有映射

        Args:
            input: 输入数据

        Returns:
            输出字典
        """
        import asyncio

        logger.debug(f"RunnableMap ainvoke - mappings: {list(self.mappings.keys())}")

        tasks = [runnable.ainvoke(input) for runnable in self.mappings.values()]
        results = await asyncio.gather(*tasks)

        logger.debug(f"RunnableMap completed - results: {list(results)}")
        return dict(zip(self.mappings.keys(), results))


__all__ = [
    "RunnableSequence",
    "RunnableBranch",
    "RunnableMap",
]
