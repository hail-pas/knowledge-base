from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import BaseModel, ConfigDict, Field


class LangChainModelCapabilities(BaseModel):
    supports_tools: bool = False
    supports_image_output: bool = False
    supports_json_schema_output: bool = False
    supports_json_object_output: bool = False
    default_structured_output_mode: str = "tool"
    native_output_requires_schema_in_instructions: bool = False


class CapabilityAwareChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    wrapped: BaseChatModel
    capabilities: LangChainModelCapabilities = Field(default_factory=LangChainModelCapabilities)

    @property
    def _llm_type(self) -> str:
        return self.wrapped._llm_type

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return dict(self.wrapped._identifying_params)

    def get_capabilities(self) -> LangChainModelCapabilities:
        return self.capabilities.model_copy()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self.wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await self.wrapped._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools: Any, *, tool_choice: str | None = None, **kwargs: Any) -> Any:
        return self.wrapped.bind_tools(tools, tool_choice=tool_choice, **kwargs)

    def with_structured_output(self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any) -> Any:
        return self.wrapped.with_structured_output(schema, include_raw=include_raw, **kwargs)
