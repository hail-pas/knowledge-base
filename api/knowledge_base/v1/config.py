from typing import Annotated

from fastapi import Depends, Request, APIRouter
from tortoise.queryset import QuerySet
from tortoise.expressions import Q

from core.types import ApiException
from core.schema import CRUDPager
from core.response import Resp, PageData
from service.depend import api_permission_check
from ext.ext_tortoise.curd import (
    list_view,
    create_obj,
    update_obj,
    detail_view,
    pagination_factory,
)
from service.llm_model.schema import (
    LLMModelConfigList,
    LLMModelConfigCreate,
    LLMModelConfigDetail,
    LLMModelConfigUpdate,
    LLMModelConfigFilterSchema,
)
from service.file_source.schema import (
    FileSourceList,
    FileSourceCreate,
    FileSourceDetail,
    FileSourceUpdate,
    FileSourceFilterSchema,
)
from service.embedding_model.schema import (
    EmbeddingModelConfigList,
    EmbeddingModelConfigCreate,
    EmbeddingModelConfigDetail,
    EmbeddingModelConfigFilterSchema,
)
from service.indexing_backend.schema import (
    IndexingBackendConfigList,
    IndexingBackendConfigCreate,
    IndexingBackendConfigDetail,
    IndexingBackendConfigFilterSchema,
)
from ext.ext_tortoise.models.user_center import Account
from ext.ext_tortoise.models.knowledge_base import (
    FileSource,
    LLMModelConfig,
    EmbeddingModelConfig,
    IndexingBackendConfig,
)

router = APIRouter(dependencies=[Depends(api_permission_check)])


# =============================================================================
# FileSource Routes
# =============================================================================


def get_file_source_queryset(request: Request) -> QuerySet[FileSource]:
    user: Account = request.scope["user"]
    queryset = FileSource.filter(deleted_at=0)

    if not user.is_staff:
        queryset = queryset.filter(Q(user_id=None) | Q(user_id=user.id))

    return queryset


@router.post("/file-source", summary="创建文件源")
async def create_file_source(request: Request, schema: FileSourceCreate) -> Resp[FileSourceList]:
    # user: Account = request.state.user

    schema.validate_required_fields_by_type()
    schema.validate_extra_config()

    obj: FileSource = await create_obj(FileSource, schema.model_dump(exclude_unset=True))  # type: ignore

    if schema.is_default:  # type: ignore
        await (
            FileSource.filter(is_default=True, user_id=obj.user_id, deleted_at=0)
            .exclude(id=obj.id)
            .update(is_default=False)
        )

    return Resp(data=FileSourceList.model_validate(obj))


@router.put("/file-source/{pk}", summary="更新文件源")
async def update_file_source(request: Request, pk: int, schema: FileSourceUpdate) -> Resp:
    queryset = get_file_source_queryset(request)

    obj = await queryset.get_or_none(pk=pk)
    if not obj:
        raise ApiException("文件源不存在")

    if schema.extra_config is not None:  # type: ignore
        FileSourceCreate.validate_extra_config_by_type(obj.type, schema.extra_config)  # type: ignore

    await update_obj(obj, queryset, schema.model_dump(exclude_unset=True))

    if schema.is_default:  # type: ignore
        await (
            FileSource.filter(is_default=True, user_id=obj.user_id, deleted_at=0)
            .exclude(pk=pk)
            .update(is_default=False)
        )

    return Resp()


@router.get("/file-source", summary="文件源列表")
async def list_file_sources(
    request: Request,
    filter_: Annotated[FileSourceFilterSchema, Depends(FileSourceFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=FileSource,
        search_fields={"name"},
        order_fields={"id", "created_at"},
        list_schema=FileSourceList,
        max_limit=100,
    ),
) -> Resp[PageData[FileSourceList]]:
    return await list_view(get_file_source_queryset(request), filter_, pager)


@router.get("/file-source/{pk}", summary="文件源详情")
async def get_file_source_detail(request: Request, pk: int) -> Resp[FileSourceDetail]:
    return await detail_view(get_file_source_queryset(request), pk, FileSourceDetail)


# =============================================================================
# LLMModelConfig Routes
# =============================================================================


def get_llm_config_queryset(request: Request) -> QuerySet[LLMModelConfig]:
    return LLMModelConfig.filter(deleted_at=0)


@router.post("/llm-model", summary="创建LLM模型配置")
async def create_llm_model(request: Request, schema: LLMModelConfigCreate) -> Resp[LLMModelConfigList]:
    schema.validate_required_fields_by_type()
    schema.validate_extra_config()

    obj = await create_obj(LLMModelConfig, schema.model_dump(exclude_unset=True))

    if schema.is_default:  # type: ignore
        await LLMModelConfig.filter(is_default=True, deleted_at=0).exclude(pk=obj.id).update(is_default=False)  # type: ignore

    return Resp(data=LLMModelConfigList.model_validate(obj))


@router.put("/llm-model/{pk}", summary="更新LLM模型配置")
async def update_llm_model(request: Request, pk: int, schema: LLMModelConfigUpdate) -> Resp:
    queryset = get_llm_config_queryset(request)

    obj = await queryset.get_or_none(pk=pk)
    if not obj:
        raise ApiException("LLM模型配置不存在")

    if schema.extra_config is not None:  # type: ignore
        LLMModelConfigCreate.validate_extra_config_by_type(obj.type, schema.extra_config)  # type: ignore

    await update_obj(obj, queryset, schema.model_dump(exclude_unset=True))
    if schema.is_default:  # type: ignore
        await LLMModelConfig.filter(is_default=True, deleted_at=0).exclude(pk=pk).update(is_default=False)

    return Resp()


@router.get("/llm-model", summary="LLM模型配置列表")
async def list_llm_models(
    request: Request,
    filter_: Annotated[LLMModelConfigFilterSchema, Depends(LLMModelConfigFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=LLMModelConfig,
        search_fields={"name"},
        order_fields={"id", "created_at"},
        list_schema=LLMModelConfigList,
        max_limit=100,
    ),
) -> Resp[PageData[LLMModelConfigList]]:
    return await list_view(get_llm_config_queryset(request), filter_, pager)


@router.get("/llm-model/{pk}", summary="LLM模型配置详情")
async def get_llm_model_detail(request: Request, pk: int) -> Resp[LLMModelConfigDetail]:
    return await detail_view(get_llm_config_queryset(request), pk, LLMModelConfigDetail)


# =============================================================================
# IndexingBackendConfig Routes
# =============================================================================


def get_indexing_backend_queryset(request: Request) -> QuerySet[IndexingBackendConfig]:
    return IndexingBackendConfig.filter(deleted_at=0)


@router.post("/indexing-backend", summary="创建索引后端配置")
async def create_indexing_backend(
    request: Request,
    schema: IndexingBackendConfigCreate,
) -> Resp[IndexingBackendConfigList]:
    schema.validate_required_fields_by_type()
    schema.validate_extra_config()

    obj = await create_obj(IndexingBackendConfig, schema.model_dump(exclude_unset=True))

    if obj.is_default:  # type: ignore
        await (
            IndexingBackendConfig.filter(is_default=True, type=obj.type, deleted_at=0)  # type: ignore
            .exclude(pk=obj.id)  # type: ignore
            .update(is_default=False)
        )  # type: ignore

    return Resp(data=IndexingBackendConfigList.model_validate(obj))


@router.get("/indexing-backend", summary="索引后端配置列表")
async def list_indexing_backends(
    request: Request,
    filter_: Annotated[IndexingBackendConfigFilterSchema, Depends(IndexingBackendConfigFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=IndexingBackendConfig,
        search_fields={"name"},
        order_fields={"id", "created_at"},
        list_schema=IndexingBackendConfigList,
        max_limit=100,
    ),
) -> Resp[PageData[IndexingBackendConfigList]]:
    return await list_view(get_indexing_backend_queryset(request), filter_, pager)


@router.get("/indexing-backend/{pk}", summary="索引后端配置详情")
async def get_indexing_backend_detail(request: Request, pk: int) -> Resp[IndexingBackendConfigDetail]:
    return await detail_view(get_indexing_backend_queryset(request), pk, IndexingBackendConfigDetail)


# =============================================================================
# EmbeddingModelConfig Routes
# =============================================================================


def get_embedding_model_queryset(request: Request) -> QuerySet[EmbeddingModelConfig]:
    return EmbeddingModelConfig.filter(deleted_at=0)


@router.post("/embedding-model", summary="创建Embedding模型配置")
async def create_embedding_model(
    request: Request,
    schema: EmbeddingModelConfigCreate,
) -> Resp[EmbeddingModelConfigList]:
    schema.validate_required_fields_by_type()
    schema.validate_extra_config()

    obj = await create_obj(EmbeddingModelConfig, schema.model_dump(exclude_unset=True))

    if obj.is_default:  # type: ignore
        await (
            EmbeddingModelConfig.filter(is_default=True, type=obj.type, deleted_at=0)  # type: ignore
            .exclude(pk=obj.id)  # type: ignore
            .update(is_default=False)
        )  # type: ignore

    return Resp(data=EmbeddingModelConfigList.model_validate(obj))


@router.get("/embedding-model", summary="Embedding模型配置列表")
async def list_embedding_models(
    request: Request,
    filter_: Annotated[EmbeddingModelConfigFilterSchema, Depends(EmbeddingModelConfigFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=EmbeddingModelConfig,
        search_fields={"name"},
        order_fields={"id", "created_at"},
        list_schema=EmbeddingModelConfigList,
        max_limit=100,
    ),
) -> Resp[PageData[EmbeddingModelConfigList]]:
    return await list_view(get_embedding_model_queryset(request), filter_, pager)


@router.get("/embedding-model/{pk}", summary="Embedding模型配置详情")
async def get_embedding_model_detail(request: Request, pk: int) -> Resp[EmbeddingModelConfigDetail]:
    return await detail_view(get_embedding_model_queryset(request), pk, EmbeddingModelConfigDetail)
