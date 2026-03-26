from __future__ import annotations

from loguru import logger
from pydantic import Field
from pydantic_ai import Agent

from service.chat.domain.schema import (
    ActionCapabilityMetadata,
    ActionMetadata,
    CapabilityCategoryEnum,
    CapabilityKindEnum,
    CapabilityPlannerModeEnum,
    CapabilityRuntimeKindEnum,
    ResourceAction,
    StrictModel,
    merge_action_metadata,
)
from service.llm_model.factory import LLMModelFactory


class RuntimeCapabilityDescriptor(StrictModel):
    capability_id: int | None = Field(default=None, ge=1)
    capability_key: str = Field(min_length=1, max_length=128)
    capability_kind: CapabilityKindEnum
    category: CapabilityCategoryEnum
    runtime_kind: CapabilityRuntimeKindEnum
    name: str = Field(min_length=1, max_length=128)
    capability_version: int | None = Field(default=None, ge=1)
    description: str = Field(default="", max_length=4000)
    tags: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    preferred_capability_keys: list[str] = Field(default_factory=list)
    actions: list[ResourceAction] = Field(default_factory=list)
    explicit: bool = False
    required: bool = False
    always_on: bool = False


class RuntimeCapabilityCandidate(StrictModel):
    descriptor: RuntimeCapabilityDescriptor
    reasons: list[str] = Field(default_factory=list)
    selected: bool = False


class CapabilityCatalogSnapshot(StrictModel):
    descriptors: list[RuntimeCapabilityDescriptor] = Field(default_factory=list)
    inline_actions: list[ResourceAction] = Field(default_factory=list)

    def descriptor_map(self) -> dict[str, RuntimeCapabilityDescriptor]:
        return {item.capability_key: item for item in self.descriptors}


class RuntimeExecutionPlan(StrictModel):
    planner_mode: CapabilityPlannerModeEnum
    summary: str = Field(default="", max_length=1000)
    candidates: list[RuntimeCapabilityCandidate] = Field(default_factory=list)
    selected_capabilities: list[RuntimeCapabilityDescriptor] = Field(default_factory=list)
    actions: list[ResourceAction] = Field(default_factory=list)

    @property
    def selected_capability_keys(self) -> list[str]:
        return [item.capability_key for item in self.selected_capabilities]


class _LLMCapabilityPlan(StrictModel):
    selected_capability_keys: list[str] = Field(default_factory=list)
    summary: str = ""


class CapabilityCandidateConverger:
    def __init__(self, *, max_candidates: int = 12) -> None:
        self.max_candidates = max_candidates

    def converge(
        self,
        *,
        descriptors: list[RuntimeCapabilityDescriptor],
        allow_system_defaults: bool,
        include_optional: bool,
    ) -> list[RuntimeCapabilityCandidate]:
        candidates: list[RuntimeCapabilityCandidate] = []
        for descriptor in descriptors:
            selected = descriptor.explicit or descriptor.required or descriptor.always_on
            if not allow_system_defaults and not selected:
                continue
            if not selected and not include_optional:
                continue
            reasons = self._build_reasons(descriptor=descriptor, selected=selected)
            candidates.append(
                RuntimeCapabilityCandidate(
                    descriptor=descriptor,
                    reasons=reasons,
                    selected=selected,
                ),
            )

        candidates.sort(
            key=lambda item: (
                -int(item.selected),
                item.descriptor.capability_kind.value,
                item.descriptor.capability_key,
            ),
        )
        return candidates[: self.max_candidates]

    def _build_reasons(
        self,
        *,
        descriptor: RuntimeCapabilityDescriptor,
        selected: bool,
    ) -> list[str]:
        if descriptor.required:
            return ["required by selection"]
        if descriptor.explicit:
            return ["explicitly selected"]
        if descriptor.always_on:
            return ["always_on"]
        if selected:
            return ["selected"]
        return ["eligible for llm planning"]


class CapabilityPlanner:
    def __init__(
        self,
        *,
        converger: CapabilityCandidateConverger | None = None,
    ) -> None:
        self.converger = converger or CapabilityCandidateConverger()

    async def build_plan(
        self,
        *,
        query: str,
        catalog: CapabilityCatalogSnapshot,
        allow_system_defaults: bool,
        planner_mode: CapabilityPlannerModeEnum,
        planner_model_config_id: int | None = None,
    ) -> RuntimeExecutionPlan:
        llm_planning_enabled = (
            allow_system_defaults
            and planner_model_config_id is not None
            and planner_mode == CapabilityPlannerModeEnum.llm
        )
        candidates = self.converger.converge(
            descriptors=catalog.descriptors,
            allow_system_defaults=allow_system_defaults,
            include_optional=llm_planning_enabled,
        )
        selected_keys = {
            item.descriptor.capability_key
            for item in candidates
            if item.selected
        }

        if llm_planning_enabled and candidates:
            llm_keys = await self._select_with_llm(
                query=query,
                candidates=candidates,
                planner_model_config_id=planner_model_config_id,
            )
            selected_keys.update(llm_keys)

        ordered_descriptors = self._order_selected_descriptors(
            selected_keys=selected_keys,
            candidates=candidates,
            descriptor_map=catalog.descriptor_map(),
        )
        actions = self._compile_actions(
            selected_descriptors=ordered_descriptors,
            inline_actions=catalog.inline_actions,
        )

        for candidate in candidates:
            candidate.selected = candidate.descriptor.capability_key in {
                item.capability_key for item in ordered_descriptors
            }

        return RuntimeExecutionPlan(
            planner_mode=planner_mode,
            summary=self._build_summary(
                ordered_descriptors,
                allow_system_defaults=allow_system_defaults,
                llm_planning_enabled=llm_planning_enabled,
            ),
            candidates=candidates,
            selected_capabilities=ordered_descriptors,
            actions=actions,
        )

    async def _select_with_llm(
        self,
        *,
        query: str,
        candidates: list[RuntimeCapabilityCandidate],
        planner_model_config_id: int | None,
    ) -> set[str]:
        if planner_model_config_id is None or len(candidates) <= 1:
            return set()
        try:
            model = await LLMModelFactory.create_by_id(planner_model_config_id)
            agent = Agent(
                model=model,
                output_type=_LLMCapabilityPlan,
                system_prompt=(
                    "你是 capability planner。"
                    "从候选能力中选择真正需要的一个或多个能力。"
                    "避免冗余，不要为了看起来全面而多选。"
                ),
            )
            prompt = "\n".join(
                [
                    f"用户问题: {query}",
                    "候选能力:",
                    *[
                        (
                            f"- key={item.descriptor.capability_key}; "
                            f"name={item.descriptor.name}; "
                            f"kind={item.descriptor.capability_kind.value}; "
                            f"runtime={item.descriptor.runtime_kind.value}; "
                            f"description={item.descriptor.description or 'n/a'}; "
                            f"instructions={'; '.join(item.descriptor.instructions[:3]) or 'n/a'}; "
                            f"constraints={'; '.join(item.descriptor.constraints[:3]) or 'n/a'}; "
                            f"tags={', '.join(item.descriptor.tags[:5]) or 'n/a'}; "
                            f"reasons={'; '.join(item.reasons)}"
                        )
                        for item in candidates
                    ],
                ],
            )
            result = await agent.run(
                prompt,
                model_settings={"temperature": 0.0, "max_tokens": 400},
            )
            return {item for item in result.output.selected_capability_keys if item}
        except Exception:
            logger.exception("Capability planner LLM failed")
            return set()

    def _order_selected_descriptors(
        self,
        *,
        selected_keys: set[str],
        candidates: list[RuntimeCapabilityCandidate],
        descriptor_map: dict[str, RuntimeCapabilityDescriptor],
    ) -> list[RuntimeCapabilityDescriptor]:
        ordered: list[RuntimeCapabilityDescriptor] = []
        visited: set[str] = set()

        def append_descriptor(descriptor: RuntimeCapabilityDescriptor) -> None:
            if descriptor.capability_key in visited:
                return
            visited.add(descriptor.capability_key)
            ordered.append(descriptor)
            for dependency_key in descriptor.preferred_capability_keys:
                dependency = descriptor_map.get(dependency_key)
                if dependency is not None:
                    append_descriptor(dependency)

        for candidate in candidates:
            if candidate.descriptor.capability_key in selected_keys:
                append_descriptor(candidate.descriptor)

        return ordered

    def _compile_actions(
        self,
        *,
        selected_descriptors: list[RuntimeCapabilityDescriptor],
        inline_actions: list[ResourceAction],
    ) -> list[ResourceAction]:
        actions: list[ResourceAction] = []
        for order, descriptor in enumerate(selected_descriptors, start=1):
            capability_metadata = ActionMetadata(
                capability=ActionCapabilityMetadata(
                    capability_id=descriptor.capability_id,
                    capability_key=descriptor.capability_key,
                    capability_kind=descriptor.capability_kind,
                    capability_category=descriptor.category,
                    capability_name=descriptor.name,
                    capability_version=descriptor.capability_version,
                    capability_order=order,
                    capability_required=descriptor.required,
                    capability_runtime_kind=descriptor.runtime_kind,
                ),
            )
            for action in descriptor.actions:
                actions.append(
                    action.model_copy(
                        update={
                            "metadata": merge_action_metadata(action.metadata, capability_metadata),
                        },
                    ),
                )
        actions.extend(inline_actions)
        return actions

    def _build_summary(
        self,
        selected_descriptors: list[RuntimeCapabilityDescriptor],
        *,
        allow_system_defaults: bool,
        llm_planning_enabled: bool,
    ) -> str:
        if not selected_descriptors:
            if not allow_system_defaults:
                return "系统默认能力已关闭，仅保留基础聊天回答"
            if llm_planning_enabled:
                return "本轮未选择额外能力，回退到基础聊天回答"
            return "未启用自动 capability planner，仅保留显式选择能力和基础聊天回答"
        capability_names = "、".join(item.name for item in selected_descriptors[:4])
        if len(selected_descriptors) > 4:
            capability_names += " 等"
        return f"已为当前问题选择能力：{capability_names}"
