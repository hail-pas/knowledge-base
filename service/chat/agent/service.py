from __future__ import annotations

from typing import Any

from tortoise.exceptions import IntegrityError

from core.types import ApiException
from service.chat.agent.repository import ChatAgentRepository
from service.chat.agent.schema import (
    AgentMountCreate,
    AgentMountSummary,
    AgentMountUpdate,
    AgentProfileCreate,
    AgentProfileManifest,
    AgentProfileSummary,
    AgentProfileUpdate,
)
from service.chat.domain.schema import (
    AgentRoleEnum,
    AgentMountModeEnum,
    ChatActionKindEnum,
    ResourceSelection,
    SystemPromptConfig,
)
from service.chat.execution.registry import (
    ExecutionActionRegistry,
    create_default_action_registry,
)
from ext.ext_tortoise.models.knowledge_base import ChatAgentMount, ChatAgentProfile


class ChatAgentService:
    DEFAULT_ORCHESTRATOR_KEY = "orchestrator.default"
    DEFAULT_SPECIALIST_KEY = "specialist.math"
    DEFAULT_SPECIALIST_CAPABILITY_KEY = "agent_math_delegate"

    def __init__(
        self,
        *,
        repository: ChatAgentRepository | None = None,
        action_registry: ExecutionActionRegistry | None = None,
    ) -> None:
        self.repository = repository or ChatAgentRepository()
        self.action_registry = action_registry or create_default_action_registry()

    async def create_agent(
        self,
        payload: AgentProfileCreate,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> AgentProfileSummary:
        manifest = self._normalize_manifest(payload.manifest)
        try:
            agent = await self.repository.create_agent(
                owner_account_id=None if is_staff or account_id is None else account_id,
                agent_key=manifest.agent_key,
                role=manifest.role.value,
                name=manifest.name,
                description=manifest.description,
                system_prompt=manifest.system_prompt,
                llm_model_config_id=manifest.llm_model_config_id,
                default_resource_config=manifest.default_resource_selection.model_dump(mode="json"),
                capability_keys=manifest.capability_keys,
                is_enabled=payload.is_enabled,
                metadata={**manifest.metadata, **payload.metadata},
            )
        except IntegrityError as exc:
            raise ApiException("Agent 已存在") from exc
        return self._serialize_agent(agent)

    async def update_agent(
        self,
        agent_id: int,
        payload: AgentProfileUpdate,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> AgentProfileSummary:
        agent = await self._get_agent_for_write(agent_id, account_id=account_id, is_staff=is_staff)
        if agent is None:
            raise ApiException("Agent 不存在")

        update_fields: list[str] = []
        if payload.manifest is not None:
            manifest = self._normalize_manifest(payload.manifest)
            agent.agent_key = manifest.agent_key
            agent.role = manifest.role.value  # type: ignore
            agent.name = manifest.name
            agent.description = manifest.description
            agent.system_prompt = manifest.system_prompt
            agent.llm_model_config_id = manifest.llm_model_config_id  # type: ignore
            agent.default_resource_config = manifest.default_resource_selection.model_dump(mode="json")
            agent.capability_keys = manifest.capability_keys
            agent.version += 1
            update_fields.extend(
                [
                    "agent_key",
                    "role",
                    "name",
                    "description",
                    "system_prompt",
                    "llm_model_config_id",
                    "default_resource_config",
                    "capability_keys",
                    "version",
                ],
            )
        if payload.is_enabled is not None:
            agent.is_enabled = payload.is_enabled
            update_fields.append("is_enabled")
        if payload.metadata is not None:
            agent.metadata = payload.metadata
            update_fields.append("metadata")
        if update_fields:
            try:
                await agent.save(update_fields=update_fields)
            except IntegrityError as exc:
                raise ApiException("Agent 已存在") from exc
        return self._serialize_agent(agent)

    async def get_agent(
        self,
        agent_id: int,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> AgentProfileSummary:
        await self.ensure_builtin_agents()
        agent = await self.repository.get_agent(agent_id)
        if agent is None or (not is_staff and agent.owner_account_id not in {None, account_id}):
            raise ApiException("Agent 不存在")
        return self._serialize_agent(agent)

    async def get_agent_by_key(
        self,
        agent_key: str,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
    ) -> AgentProfileSummary:
        await self.ensure_builtin_agents()
        agent = await self.repository.get_agent_by_key(agent_key=agent_key, account_id=account_id, is_staff=is_staff)
        if agent is None:
            raise ApiException("Agent 不存在")
        return self._serialize_agent(agent)

    async def list_agents(
        self,
        *,
        account_id: int | None = None,
        is_staff: bool = True,
        role: AgentRoleEnum | None = None,
        is_enabled: bool | None = None,
    ) -> list[AgentProfileSummary]:
        await self.ensure_builtin_agents()
        agents = await self.repository.list_agents(
            account_id=account_id,
            is_staff=is_staff,
            role=role.value if isinstance(role, AgentRoleEnum) else None,
            is_enabled=is_enabled,
        )
        return [self._serialize_agent(item) for item in agents]

    async def list_agents_by_ids(
        self,
        agent_ids: list[int],
        *,
        account_id: int | None = None,
        is_staff: bool = True,
        is_enabled: bool | None = None,
    ) -> list[AgentProfileSummary]:
        await self.ensure_builtin_agents()
        agents = await self.repository.list_agents_by_ids(
            agent_ids,
            account_id=account_id,
            is_staff=is_staff,
            is_enabled=is_enabled,
        )
        return [self._serialize_agent(item) for item in agents]

    async def create_mount(self, payload: AgentMountCreate) -> AgentMountSummary:
        await self.ensure_builtin_agents()
        source_agent = await self.repository.get_agent(payload.source_agent_id)
        mounted_agent = await self.repository.get_agent(payload.mounted_agent_id)
        if source_agent is None or mounted_agent is None:
            raise ApiException("Agent 不存在")
        try:
            mount = await self.repository.create_mount(
                source_agent_id=payload.source_agent_id,
                mounted_agent_id=payload.mounted_agent_id,
                mode=payload.mode.value,
                purpose=payload.purpose,
                trigger_tags=payload.trigger_tags,
                pass_message_history=payload.pass_message_history,
                pass_deps_fields=payload.pass_deps_fields,
                output_contract=payload.output_contract,
                mounted_as_capability=payload.mounted_as_capability,
                is_enabled=payload.is_enabled,
                metadata=payload.metadata,
            )
        except IntegrityError as exc:
            raise ApiException("Agent Mount 已存在") from exc
        mount = await self.repository.get_mount(mount.id)
        if mount is None:
            raise ApiException("Agent Mount 查询失败")
        return self._serialize_mount(mount)

    async def update_mount(self, mount_id: int, payload: AgentMountUpdate) -> AgentMountSummary:
        mount = await self.repository.get_mount(mount_id)
        if mount is None:
            raise ApiException("Agent Mount 不存在")
        update_fields: list[str] = []
        if payload.mode is not None:
            mount.mode = payload.mode.value  # type: ignore
            update_fields.append("mode")
        if payload.purpose is not None:
            mount.purpose = payload.purpose
            update_fields.append("purpose")
        if payload.trigger_tags is not None:
            mount.trigger_tags = payload.trigger_tags
            update_fields.append("trigger_tags")
        if payload.pass_message_history is not None:
            mount.pass_message_history = payload.pass_message_history
            update_fields.append("pass_message_history")
        if payload.pass_deps_fields is not None:
            mount.pass_deps_fields = payload.pass_deps_fields
            update_fields.append("pass_deps_fields")
        if payload.output_contract is not None:
            mount.output_contract = payload.output_contract
            update_fields.append("output_contract")
        if payload.mounted_as_capability is not None:
            mount.mounted_as_capability = payload.mounted_as_capability
            update_fields.append("mounted_as_capability")
        if payload.is_enabled is not None:
            mount.is_enabled = payload.is_enabled
            update_fields.append("is_enabled")
        if payload.metadata is not None:
            mount.metadata = payload.metadata
            update_fields.append("metadata")
        if update_fields:
            try:
                await mount.save(update_fields=update_fields)
            except IntegrityError as exc:
                raise ApiException("Agent Mount 已存在") from exc
        mount = await self.repository.get_mount(mount.id)
        if mount is None:
            raise ApiException("Agent Mount 查询失败")
        return self._serialize_mount(mount)

    async def list_mounts(
        self,
        *,
        source_agent_id: int | None = None,
        mounted_agent_id: int | None = None,
        is_enabled: bool | None = None,
    ) -> list[AgentMountSummary]:
        await self.ensure_builtin_agents()
        mounts = await self.repository.list_mounts(
            source_agent_id=source_agent_id,
            mounted_agent_id=mounted_agent_id,
            is_enabled=is_enabled,
        )
        return [self._serialize_mount(item) for item in mounts]

    async def get_mounts_for_agent(
        self,
        *,
        agent_id: int,
        is_enabled: bool = True,
    ) -> list[AgentMountSummary]:
        return await self.list_mounts(source_agent_id=agent_id, is_enabled=is_enabled)

    async def ensure_builtin_agents(self) -> None:
        orchestrator = await self.repository.get_agent_by_key(
            agent_key=self.DEFAULT_ORCHESTRATOR_KEY,
            account_id=None,
            is_staff=True,
        )
        if orchestrator is None:
            try:
                orchestrator = await self.repository.create_agent(
                    owner_account_id=None,
                    agent_key=self.DEFAULT_ORCHESTRATOR_KEY,
                    role=AgentRoleEnum.orchestrator.value,
                    name="默认编排代理",
                    description="通用聊天主代理，负责 capability 选择与最终回答。",
                    system_prompt="你是通用聊天平台的主编排代理。",
                    default_resource_config=ResourceSelection().model_dump(mode="json"),
                    capability_keys=[],
                    metadata={"builtin": True},
                    is_enabled=True,
                )
            except IntegrityError:
                orchestrator = await self.repository.get_agent_by_key(
                    agent_key=self.DEFAULT_ORCHESTRATOR_KEY,
                    account_id=None,
                    is_staff=True,
                )

        specialist = await self.repository.get_agent_by_key(
            agent_key=self.DEFAULT_SPECIALIST_KEY,
            account_id=None,
            is_staff=True,
        )
        if specialist is None:
            selection = self.action_registry.normalize_selection(
                ResourceSelection(
                    actions=[
                        self.action_registry.build_action(
                            ChatActionKindEnum.system_prompt,
                            config=SystemPromptConfig(
                                instructions=[
                                    "你是数学专家能力，只在需要时负责计算并输出准确结果。",
                                ],
                            ),
                            action_id="builtin:agent_math:prompt",
                            priority=10,
                            source="builtin:agent",
                        ),
                        self.action_registry.build_action(
                            ChatActionKindEnum.tool_call,
                            config={
                                "tools": [{"tool_name": "calculate_expression"}],
                                "stop_after_terminal": True,
                            },
                            action_id="builtin:agent_math:tool",
                            priority=20,
                            source="builtin:agent",
                        ),
                    ],
                ),
            )
            try:
                specialist = await self.repository.create_agent(
                    owner_account_id=None,
                    agent_key=self.DEFAULT_SPECIALIST_KEY,
                    role=AgentRoleEnum.specialist.value,
                    name="数学专家代理",
                    description="处理算术表达式和简单数学问题。",
                    system_prompt="你是数学专家代理。",
                    default_resource_config=selection.model_dump(mode="json"),
                    capability_keys=[],
                    metadata={"builtin": True, "tags": ["math", "calculator"]},
                    is_enabled=True,
                )
            except IntegrityError:
                specialist = await self.repository.get_agent_by_key(
                    agent_key=self.DEFAULT_SPECIALIST_KEY,
                    account_id=None,
                    is_staff=True,
                )

        if orchestrator is None or specialist is None:
            raise ApiException("内置 Agent 初始化失败")

        existing_mounts = await self.repository.list_mounts(
            source_agent_id=orchestrator.id,
            mounted_agent_id=specialist.id,
            is_enabled=None,
        )
        if not any(item.mounted_as_capability == self.DEFAULT_SPECIALIST_CAPABILITY_KEY for item in existing_mounts):
            try:
                await self.repository.create_mount(
                    source_agent_id=orchestrator.id,
                    mounted_agent_id=specialist.id,
                    mode=AgentMountModeEnum.delegate.value,
                    purpose="把数学专家代理以 capability 的形式挂载给默认编排代理。",
                    trigger_tags=["计算", "算一下", "math", "expression"],
                    pass_message_history=False,
                    pass_deps_fields=[],
                    output_contract="terminal_text",
                    mounted_as_capability=self.DEFAULT_SPECIALIST_CAPABILITY_KEY,
                    metadata={"builtin": True, "category": "agent"},
                    is_enabled=True,
                )
            except IntegrityError:
                pass

    def _normalize_manifest(self, manifest: AgentProfileManifest) -> AgentProfileManifest:
        return manifest.model_copy(
            update={
                "default_resource_selection": self.action_registry.normalize_inline_selection(
                    manifest.default_resource_selection,
                    source=f"agent_manifest:{manifest.agent_key}",
                    prefix=f"agent_manifest:{manifest.agent_key}",
                ),
            },
        )

    async def _get_agent_for_write(
        self,
        agent_id: int,
        *,
        account_id: int | None,
        is_staff: bool,
    ) -> ChatAgentProfile | None:
        agent = await self.repository.get_agent(agent_id)
        if agent is None:
            return None
        if is_staff:
            return agent
        if account_id is None:
            return agent if agent.owner_account_id is None else None
        return agent if agent.owner_account_id == account_id else None

    def _serialize_agent(self, agent: ChatAgentProfile) -> AgentProfileSummary:
        manifest = AgentProfileManifest(
            agent_key=agent.agent_key,
            name=agent.name,
            role=AgentRoleEnum(str(agent.role)),
            description=agent.description or "",
            system_prompt=agent.system_prompt or "",
            llm_model_config_id=agent.llm_model_config_id,
            default_resource_selection=ResourceSelection.model_validate(agent.default_resource_config or {}),
            capability_keys=[str(item) for item in (agent.capability_keys or [])],
            metadata=agent.metadata or {},
        )
        return AgentProfileSummary(
            id=agent.id,
            owner_account_id=agent.owner_account_id,
            agent_key=agent.agent_key,
            name=agent.name,
            role=AgentRoleEnum(str(agent.role)),
            description=agent.description or "",
            system_prompt=agent.system_prompt or "",
            llm_model_config_id=agent.llm_model_config_id,
            manifest=manifest,
            is_enabled=agent.is_enabled,
            metadata=agent.metadata or {},
            version=agent.version,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        )

    def _serialize_mount(self, mount: ChatAgentMount) -> AgentMountSummary:
        return AgentMountSummary(
            id=mount.id,
            source_agent_id=mount.source_agent_id,  # type: ignore
            source_agent_key=mount.source_agent.agent_key,
            mounted_agent_id=mount.mounted_agent_id,  # type: ignore
            mounted_agent_key=mount.mounted_agent.agent_key,
            mounted_agent_name=mount.mounted_agent.name,
            mode=AgentMountModeEnum(str(mount.mode)),
            purpose=mount.purpose or "",
            trigger_tags=[str(item) for item in (mount.trigger_tags or [])],
            pass_message_history=mount.pass_message_history,
            pass_deps_fields=[str(item) for item in (mount.pass_deps_fields or [])],
            output_contract=mount.output_contract,
            mounted_as_capability=mount.mounted_as_capability,
            is_enabled=mount.is_enabled,
            metadata=mount.metadata or {},
            created_at=mount.created_at,
            updated_at=mount.updated_at,
        )
