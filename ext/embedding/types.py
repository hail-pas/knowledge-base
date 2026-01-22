"""
Embedding 类型定义

定义 extra_config 的基础类型和转换方法
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class BaseExtraConfig(BaseModel):
    """
    extra_config 基础类型

    所有provider的extra_config都应继承此类
    提供通用的字段转换方法

    类属性：默认值配置
    """

    # 默认值配置
    DEFAULT_EMBEDDING_ENDPOINT: str = "/v1/embeddings"
    DEFAULT_AUTH_HEADER: str = "Authorization"
    DEFAULT_AUTH_TYPE: str = "Bearer"
    DEFAULT_INPUT_FIELD: str = "input"
    DEFAULT_MODEL_FIELD: str = "model"
    DEFAULT_EMBEDDING_FIELD_PATH: str = "data"
    DEFAULT_EMBEDDING_VALUE_FIELD: str = "embedding"
    DEFAULT_INDEX_FIELD: str = "index"

    # 通用字段（使用类属性作为默认值）
    encoding_format: Optional[str] = Field(default=None, description="编码格式")
    user: Optional[str] = Field(default=None, description="用户标识")

    # API 配置（使用类属性作为默认值）
    endpoint: str = Field(default=DEFAULT_EMBEDDING_ENDPOINT, description="API端点路径")
    auth_header: str = Field(default=DEFAULT_AUTH_HEADER, description="认证头名称")
    auth_type: str = Field(default=DEFAULT_AUTH_TYPE, description="认证类型")

    # 请求体配置（使用类属性作为默认值）
    input_field: str = Field(default=DEFAULT_INPUT_FIELD, description="输入字段名")
    model_field: str = Field(default=DEFAULT_MODEL_FIELD, description="模型字段名")
    model_in_body: bool = Field(default=True, description="模型是否在请求体中")

    # 响应解析配置（使用类属性作为默认值）
    embedding_field_path: str = Field(default=DEFAULT_EMBEDDING_FIELD_PATH, description="嵌入数据路径")
    embedding_value_field: str = Field(default=DEFAULT_EMBEDDING_VALUE_FIELD, description="嵌入值字段名")
    index_field: str = Field(default=DEFAULT_INDEX_FIELD, description="索引字段名")

    # 额外配置
    headers: Dict[str, str] = Field(default_factory=dict, description="额外的HTTP头")
    query_params: Dict[str, str] = Field(default_factory=dict, description="查询参数")

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（用于存储到数据库）

        Returns:
            不包含 None 值的字典
        """
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseExtraConfig":
        """
        从字典创建实例

        Args:
            data: 配置字典

        Returns:
            类型化的实例
        """
        valid_data = {k: v for k, v in data.items() if k in cls.model_fields}
        return cls.model_validate(valid_data)
