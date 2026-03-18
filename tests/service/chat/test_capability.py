from service.chat.capability.registry import create_default_capability_registry
from service.chat.domain.schema import ChatCapabilityKindEnum, KnowledgeRetrievalConfig, ResourceSelection


def test_resource_selection_preserves_multiple_capabilities_of_same_kind() -> None:
    selection = ResourceSelection.model_validate(
        {
            "capabilities": [
                {
                    "capability_id": "retrieval:1",
                    "kind": "knowledge_retrieval",
                    "priority": 10,
                    "config": {"collection_ids": [1], "top_k": 3},
                },
                {
                    "capability_id": "retrieval:2",
                    "kind": "knowledge_retrieval",
                    "priority": 20,
                    "config": {"collection_ids": [2], "top_k": 5},
                },
            ],
        },
    )

    normalized = selection.normalized_capabilities()

    assert [item.capability_id for item in normalized if item.kind == ChatCapabilityKindEnum.knowledge_retrieval] == [
        "retrieval:1",
        "retrieval:2",
    ]
    assert sum(1 for item in normalized if item.kind == ChatCapabilityKindEnum.llm_response) == 1


def test_resource_selection_keeps_single_llm_response_capability() -> None:
    selection = ResourceSelection.model_validate(
        {
            "capabilities": [
                {
                    "capability_id": "llm:2",
                    "kind": "llm_response",
                    "priority": 20,
                    "config": {},
                },
                {
                    "capability_id": "llm:1",
                    "kind": "llm_response",
                    "priority": 10,
                    "config": {},
                },
            ],
        },
    )

    normalized = selection.normalized_capabilities()

    assert [item.capability_id for item in normalized if item.kind == ChatCapabilityKindEnum.llm_response] == ["llm:1"]


def test_capability_registry_builds_capability_from_persisted_config() -> None:
    registry = create_default_capability_registry()

    capability = registry.build_capability(
        "knowledge_retrieval",
        config={"collection_ids": [7], "top_k": 4},
        capability_id="profile:7",
        profile_id=7,
        name="kb-search",
        source="profile",
        priority=30,
        metadata={"channel": "default"},
    )
    descriptor = registry.build(capability)

    assert descriptor is not None
    assert descriptor.capability_id == "profile:7"
    assert descriptor.name == "kb-search"
    assert descriptor.profile_id == 7
    assert isinstance(descriptor.config, KnowledgeRetrievalConfig)
    assert descriptor.config.collection_ids == [7]
