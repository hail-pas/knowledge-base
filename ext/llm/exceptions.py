"""
LLM 模块异常定义

定义 LLM 相关的所有异常类，用于处理模型创建、配置、API 调用等过程中的错误。
"""


class LLMError(Exception):
    """LLM 模块基础异常类

    所有 LLM 相关异常的基类，用于统一捕获和处理 LLM 模块的错误。
    """
    pass


class LLMConfigError(LLMError):
    """LLM 配置错误

    当 LLM 模型配置无效、缺少必需参数或配置格式错误时抛出。

    典型场景：
    - 缺少必需的配置项（如 api_key、base_url 等）
    - 配置参数格式错误
    - 配置值超出有效范围
    """
    pass


class LLMModelNotFoundError(LLMError):
    """LLM 模型未找到错误

    当尝试创建不支持的模型类型或模型名称无效时抛出。

    典型场景：
    - 模型类型未注册（如尝试创建不存在的 provider）
    - 模型名称在对应的 API 中不存在
    - 数据库中的配置已删除但仍在尝试使用
    """
    pass


class LLMAPIError(LLMError):
    """LLM API 调用错误

    当调用 LLM 服务提供商的 API 失败时抛出。

    典型场景：
    - API 请求失败（网络错误、服务不可用）
    - API 返回错误响应（认证失败、配额超限等）
    - API 响应格式异常
    - 请求被限流
    """
    pass


class LLMTimeoutError(LLMAPIError):
    """LLM 请求超时错误

    当 LLM API 请求超过配置的超时时间时抛出。

    典型场景：
    - 网络延迟导致请求超时
    - 模型推理时间过长
    - 服务端处理缓慢
    """
    pass


class LLMRateLimitError(LLMAPIError):
    """LLM 速率限制错误

    当超过 LLM 服务提供商的 API 调用速率限制时抛出。

    典型场景：
    - 每分钟请求数超过限制
    - 每日 token 使用量超过配额
    - 并发请求数超过限制
    """
    pass


class LLMTokenLimitError(LLMError):
    """LLM Token 限制错误

    当请求或响应的 token 数量超过模型限制时抛出。

    典型场景：
    - 输入文本超过模型的最大上下文长度
    - 配置的 max_tokens 超过模型支持的最大值
    - 需要进行文本截断或分块处理
    """
    pass


class LLMCapabilityError(LLMError):
    """LLM 能力不支持错误

    当尝试使用模型不支持的功能时抛出。

    典型场景：
    - 模型不支持 function calling 但尝试使用工具调用
    - 模型不支持 json output 但请求结构化输出
    - 模型不支持多模态但尝试处理图片
    - 模型不支持流式输出但请求 stream
    """
    pass


class LLMStreamingError(LLMError):
    """LLM 流式输出错误

    当处理流式输出时发生错误时抛出。

    典型场景：
    - 流式连接中断
    - 流式响应格式错误
    - 流式解析失败
    """
    pass
