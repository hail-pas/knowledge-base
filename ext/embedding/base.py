"""
Embedding 模型泛型基类

提供embedding模型的抽象接口和默认实现
"""

from abc import ABC
from typing import TypeVar, Generic, List, Dict, Any, Optional, Type
import asyncio
import httpx
from loguru import logger

from ext.embedding.types import BaseExtraConfig

# 定义泛型类型变量
ExtraConfigT = TypeVar("ExtraConfigT", bound=BaseExtraConfig)


class BaseEmbeddingModel(Generic[ExtraConfigT], ABC):
    """
    Embedding 模型抽象基类（泛型）

    类型参数:
        ExtraConfigT: extra_config 的具体类型

    设计原则:
        1. 提供大量默认实现，子类只需在必要时覆盖
        2. 通过配置化处理80%的provider差异
        3. 使用全局httpx client
        4. 支持批处理和自动分批
    """

    extra_config: ExtraConfigT

    # ========== __init__：所有字段都标注类型 ==========

    def __init__(
        self,
        model_name: str,
        model_type: str,
        dimension: int,
        api_key: str | None,
        base_url: str | None,
        max_chunk_length: int,
        batch_size: int,
        max_retries: int,
        timeout: int,
        rate_limit: int,
        extra_config: dict[str, Any],
    ):
        """
        初始化embedding模型

        Args:
            model_name: 模型名称
            model_type: 模型类型
            dimension: 向量维度
            api_key: API密钥
            base_url: API基础URL
            max_chunk_length: 单条chunk最大长度
            batch_size: 批处理大小
            max_retries: 最大重试次数
            timeout: 请求超时时间(秒)
            rate_limit: 每分钟最大请求数(0=无限制)
            extra_config: provider特定配置（dict），内部会转换成具体的 dataclass
        """
        # 必要配置
        self.model_name = model_name
        self.model_type = model_type
        self.dimension = dimension
        self.api_key = api_key
        self.base_url = base_url

        # 批处理和性能配置
        self.max_chunk_length = max_chunk_length
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit = rate_limit

        # 转换 extra_config（从 dict 转成 pydantic model）
        self.extra_config: ExtraConfigT = self._convert_extra_config(extra_config)

        # 验证配置
        self._validate_config()

    def _validate_config(self) -> None:
        """验证配置"""
        if not self.base_url:
            raise ValueError(f"{self.model_type} requires base_url")

        if not self.api_key and self.requires_auth():
            raise ValueError(f"{self.model_type} requires api_key")

    def _get_extra_config_cls(self) -> type[BaseExtraConfig]:
        """
        从泛型参数自动提取 extra_config 类型

        子类通过 `class OpenAIEmbeddingModel(BaseEmbeddingModel[OpenAIExtraConfig])` 声明泛型参数
        此方法会自动从 `__orig_bases__` 中提取泛型类型

        Returns:
            extra_config 的 pydantic model 类型

        Raises:
            ValueError: 如果无法提取泛型类型
        """
        # 从 __orig_bases__ 提取泛型类型
        if hasattr(self, "__orig_bases__"):
            for base in self.__orig_bases__:  # type: ignore
                if hasattr(base, "__args__") and base.__args__:
                    extra_config_type = base.__args__[0]
                    # 确保是 BaseExtraConfig 的子类
                    if isinstance(extra_config_type, type) and issubclass(extra_config_type, BaseExtraConfig):
                        return extra_config_type

        # 如果无法提取，返回默认类型
        logger.warning(
            "无法从泛型参数提取 extra_config 类型，使用默认类型 BaseExtraConfig。"
            "请确保子类正确声明泛型参数，如：class OpenAIEmbeddingModel(BaseEmbeddingModel[OpenAIExtraConfig])",
        )
        return BaseExtraConfig

    def _convert_extra_config(self, extra_config_dict: dict[str, Any]) -> ExtraConfigT:
        """
        将 dict 转换成具体的 pydantic model 类型

        自动从泛型参数提取类型并转换

        Args:
            extra_config_dict: extra_config 的字典形式

        Returns:
            类型化的 extra_config 实例
        """
        extra_config_cls = self._get_extra_config_cls()
        return extra_config_cls.from_dict(extra_config_dict)  # type: ignore

    # ========== 可选的hook方法（子类可覆盖） ==========

    def requires_auth(self) -> bool:
        """是否需要认证（默认True）"""
        return True

    def supports_batch(self) -> bool:
        """是否支持原生批处理（默认True）"""
        return True

    def should_retry(self, status_code: int, attempt: int) -> bool:
        """判断是否应该重试（默认重试429和5xx）"""
        if attempt >= self.max_retries:
            return False
        return status_code in [429, 500, 502, 503, 504]

    def get_retry_delay(self, attempt: int) -> float:
        """获取重试延迟（默认指数退避）"""
        return 2**attempt

    # ========== 默认实现的方法 ==========

    def get_httpx_client(self) -> httpx.AsyncClient:
        """获取全局httpx client"""
        from config.main import local_configs

        return local_configs.extensions.httpx.instance

    # ==================== 端点构建（默认实现） ====================

    def build_endpoint_url(self) -> str:
        """
        构建API端点URL（可被子类覆盖）

        默认实现: {base_url}{endpoint}
        """
        base_url = self.base_url or ""
        endpoint = self._get_endpoint()

        # 标准化
        base_url = base_url.rstrip("/")
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        # 添加查询参数（如api_version）
        query_params = self._get_query_params()
        if query_params:
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
            endpoint = f"{endpoint}?{query_string}"

        return f"{base_url}{endpoint}"

    def _get_endpoint(self) -> str:
        """获取endpoint路径（从extra_config或默认值）"""
        return self.extra_config.endpoint or self.extra_config.DEFAULT_EMBEDDING_ENDPOINT

    def _get_query_params(self) -> dict[str, str]:
        """获取查询参数（子类可覆盖）"""
        return self.extra_config.query_params or {}

    # ==================== 认证头构建（默认实现） ====================

    def build_auth_headers(self) -> dict[str, str]:
        """
        构建认证头（默认Bearer Token实现）

        子类可通过覆盖 BaseExtraConfig 类属性来调整默认值
        """
        if not self.requires_auth():
            return {}

        api_key = self.api_key
        if not api_key:
            raise ValueError("API key is required")

        auth_header = self.extra_config.auth_header
        auth_type = self.extra_config.auth_type

        value = f"{auth_type} {api_key}" if auth_type else api_key
        return {auth_header: value}

    def build_request_headers(self) -> dict[str, str]:
        """构建完整的请求头"""
        headers = {"Content-Type": "application/json"}
        headers.update(self.build_auth_headers())

        # 添加额外headers
        extra_headers = self.extra_config.headers
        headers.update(extra_headers)

        return headers

    # ==================== 请求体构建（默认实现） ====================

    def build_request_body(self, texts: list[str]) -> dict[str, Any]:
        """
        构建请求体（默认OpenAI格式实现）

        子类可通过设置类属性或extra_config来调整：
        - DEFAULT_INPUT_FIELD
        - DEFAULT_MODEL_FIELD
        - model_in_body
        """
        # 标准化输入
        if not isinstance(texts, list):
            texts = [texts]

        input_field = self.extra_config.input_field
        model_field = self.extra_config.model_field
        model_in_body = self.extra_config.model_in_body

        body = {input_field: texts}

        # 添加模型字段（如果配置了）
        if model_in_body and model_field:
            body[model_field] = self.model_name # type: ignore

        # 添加额外参数
        body.update(self._get_extra_request_params())

        return body

    def _get_extra_request_params(self) -> dict[str, Any]:
        """
        获取额外请求参数（从extra_config中提取）

        支持的参数：
        - encoding_format: OpenAI的float/base64
        - truncate: Cohere的NONE/END/START
        - dimensions: OpenAI的自定义维度
        - user: 用户标识
        """
        supported_params = [
            "encoding_format",
            "truncate",
            "user",
            "dimensions",
        ]

        params = {k: v for k, v in self.extra_config.model_dump().items() if k in supported_params and v is not None}

        params["dimensions"] = self.dimension

        return params

    # ==================== 响应解析（默认实现） ====================

    def parse_response(self, response: dict[str, Any]) -> list[list[float]]:
        """
        解析响应，返回embeddings列表（默认OpenAI格式实现）

        子类可通过extra_config来调整：
        - embedding_field_path
        - embedding_value_field
        - index_field
        """
        field_path = self.extra_config.embedding_field_path
        value_field = self.extra_config.embedding_value_field
        index_field = self.extra_config.index_field

        # 提取嵌入数据
        embeddings_data = self._extract_by_path(response, field_path)

        if embeddings_data is None:
            raise ValueError(f"Cannot find embeddings at path: {field_path}")

        # 提取向量
        embeddings = []
        for item in embeddings_data:
            if isinstance(item, list):
                # 直接是向量
                embeddings.append(item)
            elif isinstance(item, dict):
                # 对象包含embedding字段
                emb = item.get(value_field)
                if emb is not None:
                    embeddings.append(emb)

        # 排序（如果有index字段）
        if index_field and all(isinstance(x, dict) for x in embeddings_data):
            embeddings = self._sort_by_index(embeddings_data, embeddings, index_field)

        return embeddings

    def _extract_by_path(self, data: dict[str, Any], path: str) -> Any:
        """通过路径提取数据（简单的点分隔路径）"""
        keys = path.split(".")
        result = data
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
            else:
                return None
            if result is None:
                return None
        return result

    def _sort_by_index(
        self, items: list[dict[str, Any]], embeddings: list[list[float]], index_field: str,
    ) -> list[list[float]]:
        """根据index字段排序"""
        indexed = list(zip(items, embeddings, strict=False))
        indexed.sort(key=lambda x: x[0].get(index_field, 0))
        return [emb for _, emb in indexed]

    # ==================== 错误处理（默认实现） ====================

    def extract_error_message(self, response: dict[str, Any]) -> str | None:
        """
        从响应中提取错误信息（尝试多种常见格式）

        支持的格式：
        - OpenAI: {"error": {"message": "..."}}
        - Cohere: {"message": "..."}
        - 直接: {"error": "..."}
        """
        error_paths = [
            ["error", "message"],  # OpenAI, Azure
            ["error"],  # 直接是错误
            ["message"],  # Cohere
            ["detail"],  # DRF
        ]

        for path in error_paths:
            value = self._extract_by_path(response, ".".join(path))
            if value:
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    msg = value.get("message") or value.get("msg")
                    if msg:
                        return msg

        return None

    # ==================== 批处理逻辑（默认实现） ====================

    def check_chunk_length(self, texts: list[str]) -> None:
        """检查chunk长度并输出警告"""
        for i, text in enumerate(texts):
            if len(text) > self.max_chunk_length:
                logger.warning(
                    f"Chunk #{i} 长度({len(text)})超过配置限制({self.max_chunk_length})，可能会被模型截断或导致错误",
                )

    def split_into_batches(self, texts: list[str]) -> list[list[str]]:
        """将文本列表分批"""
        return [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

    # ==================== 公开API ====================

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成embeddings

        Args:
            texts: 文本列表

        Returns:
            embeddings列表
        """
        if not texts:
            return []

        # 检查超长chunk
        self.check_chunk_length(texts)

        # 获取httpx client
        client = self.get_httpx_client()

        # 如果不支持批处理，循环调用单条接口
        if not self.supports_batch():
            logger.debug(f"Provider {self.model_type} 不支持批处理，将循环调用单条接口")
            all_embeddings = []
            for text in texts:
                embedding = await self._do_single_request(client, text)
                all_embeddings.append(embedding)
            return all_embeddings

        # 支持批处理，并发处理
        batches = self.split_into_batches(texts)
        tasks = [self._do_batch_request(client, batch) for batch in batches]
        all_embeddings_list = await asyncio.gather(*tasks)

        # 合并结果
        all_embeddings = []
        for embeddings in all_embeddings_list:
            all_embeddings.extend(embeddings)

        return all_embeddings

    # ==================== 内部请求方法 ====================

    async def _do_single_request(self, client: httpx.AsyncClient, text: str) -> list[float]:
        """发送单条请求（用于不支持批处理的provider）"""
        return (await self._do_batch_request(client, [text]))[0]

    async def _do_batch_request(self, client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
        """发送批量请求（带重试）"""
        url = self.build_endpoint_url()
        headers = self.build_request_headers()
        body = self.build_request_body(texts)

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=body, headers=headers, timeout=self.timeout)

                # 成功
                response_data = response.json()
                try:
                    return self.parse_response(response_data)
                except:
                    ...

                # 错误
                error_data = response.json()
                error_message = self.extract_error_message(error_data) or "Unknown error"

                # 判断是否重试
                if self.should_retry(response.status_code, attempt):
                    delay = self.get_retry_delay(attempt)
                    logger.warning(
                        f"请求失败（{response.status_code}: {error_message}），"
                        f"{delay}秒后重试（{attempt + 1}/{self.max_retries}）",
                    )
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(
                    f"{self.model_type} embedding请求失败: {error_message} (状态码: {response.status_code})",
                )

            except httpx.TimeoutException as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"请求超时: {str(e)}")
                delay = self.get_retry_delay(attempt)
                logger.warning(f"请求超时: {str(e)}，{delay}秒后重试")
                await asyncio.sleep(delay)

            except httpx.ConnectError as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"连接错误: {str(e)}")
                delay = self.get_retry_delay(attempt)
                logger.warning(f"连接错误: {str(e)}，{delay}秒后重试")
                await asyncio.sleep(delay)

        raise RuntimeError(f"{self.model_type} embedding请求失败，已达最大重试次数")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name}, "
            f"model_type={self.model_type}, "
            f"dimension={self.dimension}, "
            f"batch_size={self.batch_size})"
        )
