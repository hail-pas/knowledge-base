"""
Demo tasks for quick start

Simple task implementations that log their execution.
Used for testing the workflow system.
"""

import asyncio
import os
from typing import Any

from loguru import logger

from ext.workflow import ActivityTaskTemplate, activity_task


@activity_task
class FetchFileTask(ActivityTaskTemplate):
    """Fetch file task - reads file from disk"""

    async def execute(self) -> dict[str, Any]:
        file_path = self.input.get("file_path")
        logger.info(f"[FetchFileTask] Fetching file: {file_path}")

        if not file_path or not os.path.exists(file_path):
            logger.error(f"[FetchFileTask] File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.success(f"[FetchFileTask] Successfully read {len(content)} characters")

        return {
            "file_path": file_path,
            "content": content,
            "size": len(content),
        }


@activity_task
class LoadFileTask(ActivityTaskTemplate):
    """Load file task - loads content from upstream"""

    async def execute(self) -> dict[str, Any]:
        logger.info("[LoadFileTask] Loading file content")

        await asyncio.sleep(5)
        # Get content from upstream (FetchFileTask)
        upstream_outputs = await self.get_upstream_outputs()
        fetch_output = upstream_outputs.get("fetch_file", {})

        content = fetch_output.get("content", "")
        file_path = fetch_output.get("file_path", "")

        logger.success(f"[LoadFileTask] Loaded {len(content)} characters from {file_path}")

        return {
            "content": content,
            "file_path": file_path,
            "lines": content.count("\n") + 1,
        }


@activity_task
class ReplaceContentTask(ActivityTaskTemplate):
    """Replace content task - applies replace rules"""

    async def execute(self) -> dict[str, Any]:
        logger.info("[ReplaceContentTask] Processing content")
        await asyncio.sleep(5)

        # Get content from upstream (LoadFileTask)
        upstream_outputs = await self.get_upstream_outputs()
        load_output = upstream_outputs.get("load_file", {})

        content = load_output.get("content", "")
        replace_rules = self.input.get("replace_rules", [])

        logger.info(f"[ReplaceContentTask] Applying {len(replace_rules)} replace rules")

        # Apply replace rules (simple implementation)
        processed_content = content
        for rule in replace_rules:
            if isinstance(rule, dict):
                old = rule.get("old", "")
                new = rule.get("new", "")
                if old:
                    processed_content = processed_content.replace(old, new)
                    logger.info(f"[ReplaceContentTask] Replaced: '{old}' -> '{new}'")

        logger.success(f"[ReplaceContentTask] Processed {len(processed_content)} characters")

        return {
            "content": processed_content,
            "original_size": len(content),
            "processed_size": len(processed_content),
        }


@activity_task
class SummaryTask(ActivityTaskTemplate):
    """Summary task - generates a summary"""

    async def execute(self) -> dict[str, Any]:
        logger.info("[SummaryTask] Generating summary")
        await asyncio.sleep(5)

        # Get content from upstream (ReplaceContentTask)
        upstream_outputs = await self.get_upstream_outputs()
        replace_output = upstream_outputs.get("replace_content", {})

        content = replace_output.get("content", "")
        max_length = self.input.get("max_length", 100)

        # Generate simple summary
        word_count = len(content.split())
        char_count = len(content)
        line_count = content.count("\n") + 1

        # Truncate for summary preview
        preview = content[:max_length]
        if len(content) > max_length:
            preview += "..."

        logger.success(f"[SummaryTask] Generated summary: {word_count} words, {char_count} chars, {line_count} lines")

        return {
            "word_count": word_count,
            "char_count": char_count,
            "line_count": line_count,
            "preview": preview,
            "full_content": content,
        }
