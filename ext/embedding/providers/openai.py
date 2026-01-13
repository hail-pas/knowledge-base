"""
OpenAI Embedding 模型实现
"""

from typing import List
import httpx
from core.logger import logger

from ext.embedding.base import EmbeddingModel
from ext.embedding.exceptions import (
    EmbeddingAPIError,
    EmbeddingTimeoutError,
)


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI Embedding 模型实现

    使用 OpenAI API 生成文本 embedding。
    支持自定义 API endpoint（如使用代理或其他 OpenAI 兼容服务）。
    """

    # OpenAI 默认 API endpoint
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    # 默认超时时间（秒）
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        model_name_or_path: str,
        dimension: int,
        max_batch_size: int = 32,
        max_token_per_request: int = 8191,
        max_token_per_text: int = 512,
        config: dict | None = None,
    ):
        """
        初始化 OpenAI Embedding 模型

        Args:
            model_name_or_path: 模型名称，如 "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
            dimension: 向量维度
            max_batch_size: 最大批处理大小（OpenAI API 支持的最大批处理大小受限于 token 数量）
            max_token_per_request: 单次请求最大 token 数
            config: 配置字典，可包含以下字段：
                - api_key: OpenAI API key（必需）
                - base_url: API base URL（可选，默认为官方地址）
                - timeout: 请求超时时间（秒，可选）
                - organization: OpenAI organization ID（可选）
                - max_retries: 最大重试次数（可选，默认为 2）
        """
        super().__init__(
            model_name_or_path=model_name_or_path,
            dimension=dimension,
            max_batch_size=max_batch_size,
            max_token_per_request=max_token_per_request,
            max_token_per_text=max_token_per_text,
            config=config,
        )

        # 验证必需的配置参数
        self.validate_config(required_keys=["api_key"])

        # 提取配置参数
        self.api_key = self.config["api_key"]
        self.base_url = self.config.get("base_url", self.DEFAULT_BASE_URL)
        self.timeout = self.config.get("timeout", self.DEFAULT_TIMEOUT)
        self.organization = self.config.get("organization", None)
        self.max_retries = self.config.get("max_retries", 2)

        # 初始化 HTTP 客户端
        self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def _embed_batch_impl(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        实际执行批量 embedding 的实现

        Args:
            texts: 文本列表

        Returns:
            向量列表

        Raises:
            EmbeddingAPIError: API 调用失败
            EmbeddingTimeoutError: 请求超时
        """
        if not texts:
            return []

        client = self._get_client()
        url = "/embeddings"

        payload = {
            "input": texts,
            "model": self.model_name_or_path,
        }

        # 重试逻辑
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"OpenAI embedding request: model={self.model_name_or_path}, "
                    f"texts_count={len(texts)}, attempt={attempt + 1}"
                )

                response = await client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()

                # 提取 embedding 结果
                embeddings = []
                for item in data["data"]:
                    embeddings.append(item["embedding"])

                logger.debug(
                    f"OpenAI embedding success: model={self.model_name_or_path}, "
                    f"texts_count={len(texts)}, dimension={len(embeddings[0])}"
                )

                return embeddings

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"OpenAI embedding timeout (attempt {attempt + 1}): {str(e)}"
                )
                if attempt < self.max_retries:
                    continue
                raise EmbeddingTimeoutError(
                    f"OpenAI API request timeout after {self.max_retries + 1} attempts"
                ) from e

            except httpx.HTTPStatusError as e:
                last_error = e
                error_text = e.response.text
                logger.error(
                    f"OpenAI embedding API error: status={e.response.status_code}, "
                    f"response={error_text}"
                )

                # 某些错误不应该重试（如认证错误）
                if e.response.status_code in (401, 403, 429):
                    raise EmbeddingAPIError(
                        f"OpenAI API error: {error_text}",
                        status_code=e.response.status_code,
                        response_text=error_text,
                    ) from e

                # 其他错误尝试重试
                if attempt < self.max_retries:
                    logger.warning(f"Retrying after error (attempt {attempt + 1})")
                    continue

                raise EmbeddingAPIError(
                    f"OpenAI API error after {self.max_retries + 1} attempts: {error_text}",
                    status_code=e.response.status_code,
                    response_text=error_text,
                ) from e

            except Exception as e:
                last_error = e
                logger.error(f"OpenAI embedding unexpected error: {str(e)}")
                if attempt < self.max_retries:
                    logger.warning(f"Retrying after error (attempt {attempt + 1})")
                    continue
                raise EmbeddingAPIError(f"Unexpected error: {str(e)}") from e

        # 理论上不应该到这里，但为了类型检查
        raise EmbeddingAPIError(f"Failed after {self.max_retries + 1} attempts: {last_error}")

    async def close(self) -> None:
        """关闭 HTTP 客户端连接"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()

    def __repr__(self) -> str:
        return (
            f"OpenAIEmbedding("
            f"model_name={self.model_name_or_path}, "
            f"dimension={self.dimension}, "
            f"base_url={self.base_url})"
        )
