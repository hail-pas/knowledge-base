from __future__ import annotations

import re
from typing import Iterable

from loguru import logger
from pydantic_ai import Agent

from service.llm_model.factory import LLMModelFactory
from service.chat.domain.schema import StrictModel
from service.chat.capability.schema import (
    CapabilityKindEnum,
    CapabilityRoutingRule,
    CapabilityPackageSummary,
    CapabilityRoutingDecision,
    CapabilityRoutingModeEnum,
    CapabilityRoutingCandidate,
)

_TOKEN_SPLIT_RE = re.compile(r"[\s,.;:!?/\\|()\[\]{}<>\"']+")


class _LLMCapabilityPlan(StrictModel):
    selected_capability_keys: list[str]
    summary: str = ""


class ChatCapabilityRouter:
    def __init__(
        self,
        *,
        mode: CapabilityRoutingModeEnum = CapabilityRoutingModeEnum.hybrid,
        planner_model_config_id: int | None = None,
    ) -> None:
        self.mode = mode
        self.planner_model_config_id = planner_model_config_id

    async def route(
        self,
        *,
        query: str,
        packages: Iterable[CapabilityPackageSummary],
        forced_capability_keys: set[str] | None = None,
        required_capability_keys: set[str] | None = None,
        mode_override: CapabilityRoutingModeEnum | None = None,
        planner_model_config_id: int | None = None,
    ) -> CapabilityRoutingDecision:
        effective_mode = mode_override or self.mode
        package_list = list(packages)
        ranked = self.rank(
            query=query,
            packages=package_list,
            forced_capability_keys=forced_capability_keys or set(),
            required_capability_keys=required_capability_keys or set(),
        )
        if not ranked:
            return CapabilityRoutingDecision(
                mode=effective_mode,
                summary="当前未安装可用 capability package，已回退到基础聊天",
                candidates=[],
            )

        selected_ids = [item.capability_id for item in ranked if item.selected]
        summary = "基于 capability manifest 路由规则完成启发式选择"

        if effective_mode in (CapabilityRoutingModeEnum.llm, CapabilityRoutingModeEnum.hybrid) and len(ranked) > 1:
            llm_decision = await self._plan_with_llm(
                query=query,
                candidates=ranked,
                planner_model_config_id=planner_model_config_id or self.planner_model_config_id,
            )
            if llm_decision is not None:
                selected_keys = set(llm_decision.selected_capability_keys)
                if selected_keys:
                    selected_ids = []
                    for candidate in ranked:
                        candidate.selected = candidate.capability_key in selected_keys or candidate.selected
                        if candidate.selected:
                            selected_ids.append(candidate.capability_id)
                    summary = llm_decision.summary or "基于模型规划器完成 capability 选择"

        if not selected_ids:
            summary = "未命中可用 capability package，已回退到基础聊天"

        return CapabilityRoutingDecision(
            mode=effective_mode,
            summary=summary,
            selected_capability_ids=selected_ids,
            candidates=ranked,
        )

    def rank(
        self,
        *,
        query: str,
        packages: Iterable[CapabilityPackageSummary],
        forced_capability_keys: set[str],
        required_capability_keys: set[str],
    ) -> list[CapabilityRoutingCandidate]:
        ranked: list[CapabilityRoutingCandidate] = []
        per_kind_selected: dict[CapabilityKindEnum, int] = {}
        for package in packages:
            if not package.is_enabled:
                continue
            score, reasons = self._score(query=query, package=package)
            rule = package.manifest.routing
            forced = package.capability_key in forced_capability_keys
            required = package.capability_key in required_capability_keys
            selected = forced or required or rule.always_on or score >= rule.min_score
            if selected:
                selected_count = per_kind_selected.get(package.kind, 0)
                if not forced and not required and selected_count >= rule.max_selected:
                    selected = False
                    reasons.append("exceeded max_selected")
                elif selected:
                    per_kind_selected[package.kind] = selected_count + 1
            if required:
                reasons = ["required by selection", *reasons]
            if forced:
                reasons = ["forced by selection", *reasons]
            ranked.append(
                CapabilityRoutingCandidate(
                    capability_id=package.id,
                    capability_key=package.capability_key,
                    capability_kind=package.kind,
                    name=package.name,
                    score=1.0 if (forced or required) else score,
                    selected=selected,
                    reasons=reasons,
                    source="capability_router",
                ),
            )
        return sorted(
            ranked,
            key=lambda item: (-item.selected, -item.score, item.capability_kind.value, item.capability_key),
        )

    def _score(
        self,
        *,
        query: str,
        package: CapabilityPackageSummary,
    ) -> tuple[float, list[str]]:
        normalized_query = query.strip().casefold()
        tokens = self._tokenize(normalized_query)
        rule: CapabilityRoutingRule = package.manifest.routing
        score = 0.0
        reasons: list[str] = []

        if rule.always_on:
            reasons.append("always_on")
            return 1.0, reasons

        if package.capability_key.casefold() in normalized_query or package.name.casefold() in normalized_query:
            score += 0.35
            reasons.append("matched name/key")

        keyword_hits = [item for item in rule.keywords if item and item.casefold() in normalized_query]
        if keyword_hits:
            score += min(0.75, 0.3 * len(keyword_hits))
            reasons.append(f"keywords: {', '.join(keyword_hits[:3])}")

        if rule.all_of:
            hit_count = sum(1 for item in rule.all_of if item and item.casefold() in normalized_query)
            if hit_count == len(rule.all_of):
                score += 0.25
                reasons.append("matched all_of")
            elif hit_count:
                score += min(0.12, 0.04 * hit_count)
                reasons.append("partial all_of")

        if rule.any_of:
            any_hits = [item for item in rule.any_of if item and item.casefold() in normalized_query]
            if any_hits:
                score += 0.15
                reasons.append(f"any_of: {', '.join(any_hits[:2])}")

        tag_hits = [tag for tag in package.manifest.tags if tag and tag.casefold() in normalized_query]
        if tag_hits:
            score += min(0.16, 0.08 * len(tag_hits))
            reasons.append(f"tags: {', '.join(tag_hits[:2])}")

        semantic_hints = self._tokenize(
            " ".join(
                [
                    package.name,
                    package.description,
                    *package.manifest.tags,
                    *package.manifest.triggers,
                    *package.manifest.constraints,
                ],
            ),
        )
        overlap = len(tokens & semantic_hints)
        if overlap:
            score += min(0.2, 0.05 * overlap)
            reasons.append("semantic overlap")

        example_hits = sum(self._example_overlap(normalized_query, item) for item in rule.examples)
        if example_hits:
            score += min(0.18, 0.09 * example_hits)
            reasons.append("example similarity")

        excluded_hits = [item for item in rule.excluded_keywords if item and item.casefold() in normalized_query]
        if excluded_hits:
            score -= min(0.6, 0.3 * len(excluded_hits))
            reasons.append(f"excluded: {', '.join(excluded_hits[:2])}")

        return max(0.0, min(1.0, score)), reasons or ["no explicit match"]

    async def _plan_with_llm(
        self,
        *,
        query: str,
        candidates: list[CapabilityRoutingCandidate],
        planner_model_config_id: int | None,
    ) -> _LLMCapabilityPlan | None:
        if planner_model_config_id is None:
            return None
        try:
            model = await LLMModelFactory.create_by_id(planner_model_config_id)
            agent = Agent(
                model=model,
                output_type=_LLMCapabilityPlan,
                system_prompt=(
                    "你是 capability planner。"
                    "从候选 capability packages 中选择最适合解决用户问题的一个或多个能力包。"
                    "只选择真正需要的 packages，避免冗余。"
                ),
            )
            prompt = "\n".join(
                [
                    f"用户问题: {query}",
                    "候选 capability packages:",
                    *[
                        (
                            f"- key={item.capability_key}; kind={item.capability_kind.value}; "
                            f"name={item.name}; heuristic_score={item.score:.2f}; reasons={'; '.join(item.reasons)}"
                        )
                        for item in candidates[:12]
                    ],
                ],
            )
            result = await agent.run(
                prompt,
                model_settings={"temperature": 0.0, "max_tokens": 500},
            )
            return result.output
        except Exception:
            logger.exception("Capability planner LLM fallback to heuristic mode")
            return None

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in _TOKEN_SPLIT_RE.split(text) if token}

    def _example_overlap(self, query: str, example: str) -> int:
        normalized_example = example.strip().casefold()
        if not normalized_example:
            return 0
        if normalized_example in query or query in normalized_example:
            return 1
        example_tokens = self._tokenize(normalized_example)
        query_tokens = self._tokenize(query)
        return 1 if len(example_tokens & query_tokens) >= 2 else 0
