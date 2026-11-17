from typing import Annotated

from fastapi import Depends, Query, Request, APIRouter
from tortoise.queryset import QuerySet
from tortoise.expressions import Q

from core.types import ApiException
from core.schema import CRUDPager
from core.response import Resp, PageData
from service.depend import CurrentUser, api_permission_check, ensure_super_admin, is_super_admin_user
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
from util.secret import encrypt_secret, transform_extra_config_secrets
from ext.file_source.secrets import encrypt_file_source_extra_config
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


def get_file_source_queryset(request: Request, *, writable: bool = False) -> QuerySet[FileSource]:
    user: CurrentUser = request.scope["user"]
    queryset = FileSource.filter(deleted_at=0)
    if writable:
        return queryset.filter(user_id=user.id)
    return queryset.filter(Q(user_id=None) | Q(user_id=user.id))


@router.post("/file-source", summary="创建文件源")
async def create_file_source(request: Request, schema: FileSourceCreate) -> Resp[FileSourceList]:
    user: CurrentUser = request.scope["user"]

    schema.validate_required_fields_by_type()
    schema.validate_extra_config()

    source_type = getattr(schema.type, "value", schema.type)
    is_super_admin = await is_super_admin_user(user.id)
    if not is_super_admin:
        if source_type in {"local_file", "minio", "aliyun_oss"} or (
            source_type == "s3" and schema.endpoint
        ):
            raise ApiException("本地或自定义存储端点仅允许系统超级管理员配置", code=403)
        if not schema.use_ssl or not schema.verify_ssl:
            raise ApiException("非超级管理员配置的对象存储必须启用 TLS 和证书验证", code=403)

        provider_config = (schema.extra_config or {}).get("config", {})
        if source_type == "s3" and isinstance(provider_config, dict) and provider_config.get("proxies"):
            raise ApiException("自定义对象存储代理仅允许系统超级管理员配置", code=403)

    data = schema.model_dump(exclude_unset=True)
    data["user_id"] = user.id
    data["access_key"] = encrypt_secret(data.get("access_key"), purpose="file_source.access_key")
    data["secret_key"] = encrypt_secret(data.get("secret_key"), purpose="file_source.secret_key")
    data["extra_config"] = encrypt_file_source_extra_config(source_type, data.get("extra_config"))

    obj: FileSource = await create_obj(FileSource, data)  # type: ignore

    if schema.is_default:  # type: ignore
        await (
            FileSource.filter(is_default=True, user_id=obj.user_id, deleted_at=0)
            .exclude(id=obj.id)
            .update(is_default=False)
        )

    return Resp(data=FileSourceList.model_validate(obj))


@router.put("/file-source/{pk}", summary="更新文件源")
async def update_file_source(request: Request, pk: int, schema: FileSourceUpdate) -> Resp:
    queryset = get_file_source_queryset(request, writable=True)

    obj = await queryset.get_or_none(pk=pk)
    if not obj:
        raise ApiException("文件源不存在")

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
    filter_: Annotated[FileSourceFilterSchema, Query()],
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
    user: CurrentUser = request.scope["user"]
    await ensure_super_admin(user.id)

    schema.validate_required_fields_by_type()
    schema.validate_extra_config()

    data = schema.model_dump(exclude_unset=True)
    data["api_key"] = encrypt_secret(data.get("api_key"), purpose="llm_model.api_key")
    data["extra_config"] = transform_extra_config_secrets(
        data.get("extra_config"),
        purpose="llm_model.extra_config",
    )
    obj = await create_obj(LLMModelConfig, data)

    if schema.is_default:  # type: ignore
        await LLMModelConfig.filter(is_default=True, deleted_at=0).exclude(pk=obj.id).update(is_default=False)  # type: ignore

    return Resp(data=LLMModelConfigList.model_validate(obj))


@router.put("/llm-model/{pk}", summary="更新LLM模型配置")
async def update_llm_model(request: Request, pk: int, schema: LLMModelConfigUpdate) -> Resp:
    user: CurrentUser = request.scope["user"]
    await ensure_super_admin(user.id)

    queryset = get_llm_config_queryset(request)

    obj = await queryset.get_or_none(pk=pk)
    if not obj:
        raise ApiException("LLM模型配置不存在")

    await update_obj(obj, queryset, schema.model_dump(exclude_unset=True))
    if schema.is_default:  # type: ignore
        await LLMModelConfig.filter(is_default=True, deleted_at=0).exclude(pk=pk).update(is_default=False)

    return Resp()


@router.get("/llm-model", summary="LLM模型配置列表")
async def list_llm_models(
    request: Request,
    filter_: Annotated[LLMModelConfigFilterSchema, Query()],
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
    user: CurrentUser = request.scope["user"]
    await ensure_super_admin(user.id)

    schema.validate_required_fields_by_type()
    schema.validate_extra_config()

    data = schema.model_dump(exclude_unset=True)
    data["password"] = encrypt_secret(data.get("password"), purpose="indexing_backend.password")
    data["extra_config"] = transform_extra_config_secrets(
        data.get("extra_config"),
        purpose="indexing_backend.extra_config",
    )
    obj = await create_obj(IndexingBackendConfig, data)

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
    filter_: Annotated[IndexingBackendConfigFilterSchema, Query()],
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
    user: CurrentUser = request.scope["user"]
    await ensure_super_admin(user.id)

    schema.validate_required_fields_by_type()
    schema.validate_extra_config()

    data = schema.model_dump(exclude_unset=True)
    data["api_key"] = encrypt_secret(data.get("api_key"), purpose="embedding_model.api_key")
    data["extra_config"] = transform_extra_config_secrets(
        data.get("extra_config"),
        purpose="embedding_model.extra_config",
    )
    obj = await create_obj(EmbeddingModelConfig, data)

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
    filter_: Annotated[EmbeddingModelConfigFilterSchema, Query()],
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
