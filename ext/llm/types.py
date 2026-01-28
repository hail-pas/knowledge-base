"""
LLM 类型定义

定义 LLM 请求/响应的统一 Pydantic 模型
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """聊天消息

    支持文本和多模态内容（图像、音频等）
    """

    role: Literal["system", "user", "assistant", "tool"] = Field(description="消息角色")
    content: str | list[dict[str, Any]] = Field(description="消息内容")
    name: str | None = Field(default=None, description="消息名称（可选）")
    tool_call_id: str | None = Field(default=None, description="工具调用ID（tool消息专用）")


class FunctionDefinition(BaseModel):
    """函数定义

    用于 function calling 功能
    """

    name: str = Field(description="函数名称")
    description: str | None = Field(default=None, description="函数描述")
    parameters: dict[str, Any] | None = Field(default=None, description="函数参数（JSON Schema）")


class ToolDefinition(BaseModel):
    """工具定义

    支持多种工具类型（目前主要是 function）
    """

    type: Literal["function"] = Field(default="function", description="工具类型")
    function: FunctionDefinition = Field(description="函数定义")


class ToolCall(BaseModel):
    """工具调用

    LLM 返回的工具调用信息
    """

    id: str = Field(description="工具调用ID")
    type: Literal["function"] = Field(default="function", description="工具类型")
    function: dict[str, Any] = Field(description="函数调用信息")


class TokenUsage(BaseModel):
    """Token 使用统计"""

    prompt_tokens: int = Field(description="输入token数")
    completion_tokens: int = Field(description="输出token数")
    total_tokens: int = Field(description="总token数")


class LLMRequest(BaseModel):
    """LLM 请求（对话模式）

    支持的参数尽量兼容主流 provider
    """

    messages: list[ChatMessage] = Field(description="对话消息列表")
    model: str | None = Field(default=None, description="模型名称（可选，默认使用配置的模型）")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int | None = Field(default=None, ge=1, description="最大输出token数")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="nucleus sampling参数")
    top_k: int | None = Field(default=None, ge=1, description="top-k sampling参数")
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0, description="频率惩罚")
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0, description="存在惩罚")
    stop: str | list[str] | None = Field(default=None, description="停止序列")
    stream: bool = Field(default=False, description="是否流式输出")
    tools: list[ToolDefinition] | None = Field(default=None, description="工具列表")
    tool_choice: str | dict[str, Any] | None = Field(default=None, description="工具选择策略")
    response_format: dict[str, str] | None = Field(default=None, description="响应格式（如JSON mode）")


class LLMResponse(BaseModel):
    """LLM 响应（对话模式）"""

    content: str = Field(description="响应内容")
    role: str = Field(default="assistant", description="响应角色")
    usage: TokenUsage = Field(description="Token使用统计")
    finish_reason: str = Field(description="结束原因（stop/length/tool_calls/content_filter等）")
    tool_calls: list[ToolCall] | None = Field(default=None, description="工具调用列表")
    model: str | None = Field(default=None, description="使用的模型")


class StreamChunk(BaseModel):
    """流式响应块"""

    delta: dict[str, Any] = Field(default_factory=dict, description="增量内容")
    usage: TokenUsage | None = Field(default=None, description="Token使用统计（仅最后一块）")
    finish_reason: str | None = Field(default=None, description="结束原因（仅最后一块）")
    index: int = Field(default=0, description="选择索引（多输出时使用）")


class CompletionRequest(BaseModel):
    """补全模式请求

    简单的单次文本补全（兼容传统模式）
    """

    prompt: str = Field(description="提示词")
    model: str | None = Field(default=None, description="模型名称（可选，默认使用配置的模型）")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int | None = Field(default=None, ge=1, description="最大输出token数")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="nucleus sampling参数")
    stop: str | list[str] | None = Field(default=None, description="停止序列")
    suffix: str | None = Field(default=None, description="后缀（插入在完成文本后）")


class CompletionResponse(BaseModel):
    """补全模式响应"""

    text: str = Field(description="生成的文本")
    usage: TokenUsage = Field(description="Token使用统计")
    finish_reason: str = Field(description="结束原因")
    model: str | None = Field(default=None, description="使用的模型")


class BaseExtraConfig(BaseModel):
    """extra_config 基础类型

    所有 provider 的 extra_config 都应继承此类
    通过配置驱动行为，减少 hook 方法
    """

    # ========== API配置（配置驱动） ==========
    endpoint: str = Field(default="/v1/chat/completions", description="API端点路径")
    requires_auth: bool = Field(default=True, description="是否需要认证")
    auth_header: str = Field(default="Authorization", description="认证头名称")
    auth_type: str = Field(default="Bearer", description="认证类型")

    # ========== 请求配置 ==========
    model_in_body: bool = Field(default=True, description="模型字段是否在请求体中")
    model_field: str = Field(default="model", description="模型字段名称")

    # ========== 响应解析配置 ==========
    response_content_path: str = Field(default="choices.0.message.content", description="响应内容路径")
    response_usage_path: str = Field(default="usage", description="响应usage路径")
    response_finish_reason_path: str = Field(default="choices.0.finish_reason", description="响应finish_reason路径")
    response_model_path: str = Field(default="model", description="响应model路径")
    response_tool_calls_path: str | None = Field(
        default="choices.0.message.tool_calls", description="响应tool_calls路径",
    )

    # ========== 重试配置（配置驱动） ==========
    retry_on_status_codes: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504], description="重试的状态码列表",
    )
    retry_strategy: str = Field(default="exponential", description="重试策略：exponential/linear/constant")

    # ========== 额外HTTP头和查询参数 ==========
    headers: dict[str, str] = Field(default_factory=dict, description="额外的HTTP头")
    query_params: dict[str, str] = Field(default_factory=dict, description="查询参数")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于存储到数据库）

        Returns:
            不包含 None 值的字典
        """
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseExtraConfig":
        """从字典创建实例

        Args:
            data: 配置字典

        Returns:
            类型化的实例
        """
        valid_data = {k: v for k, v in data.items() if k in cls.model_fields}
        return cls.model_validate(valid_data)


class OpenAIExtraConfig(BaseExtraConfig):
    """OpenAI 特定配置

    使用官方SDK，大部分配置通过SDK处理
    """



class AzureOpenAIExtraConfig(BaseExtraConfig):
    """Azure OpenAI 特定配置

    额外的部署和API版本信息
    """

    deployment_name: str = Field(default="", description="部署名称")
    api_version: str = Field(default="2024-02-15-preview", description="API版本")


class DeepSeekExtraConfig(BaseExtraConfig):
    """DeepSeek 特定配置

    完全OpenAI兼容，使用默认配置即可
    """



class AnthropicExtraConfig(BaseExtraConfig):
    """Anthropic 特定配置

    Anthropic API 独特的消息格式
    """

    endpoint: str = Field(default="/v1/messages", description="API端点路径")
    response_content_path: str = Field(default="content.0.text", description="响应内容路径")
    response_stop_reason_path: str = Field(default="stop_reason", description="响应stop_reason路径")
    system_prompt_in_messages: bool = Field(
        default=False, description="系统提示词是否在messages中（Anthropic使用单独system参数）",
    )
