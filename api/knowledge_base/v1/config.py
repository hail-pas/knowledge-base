from typing import Annotated

from fastapi import Depends, Request, APIRouter
from tortoise.queryset import QuerySet

from core.types import ApiException
from core.schema import CRUDPager
from core.response import Resp, PageData
from ext.ext_tortoise.curd import (
    list_view,
    create_obj,
    delete_view,
    update_view,
    pagination_factory,
    obj_prefetch_fields,
)
from api.service.config.schema import (
    EmbeddingModelConfigList,
    EmbeddingModelConfigCreate,
    EmbeddingModelConfigDetail,
    EmbeddingModelConfigUpdate,
    EmbeddingModelConfigFilterSchema,
)
from ext.ext_tortoise.models.knowledge_base import EmbeddingModelConfig

router = APIRouter()


def get_embeddding_config_queryset(request: Request) -> QuerySet[EmbeddingModelConfig]:
    filter_ = {"deleted_at": 0}
    return EmbeddingModelConfig.filter(**filter_)


@router.get(
    "/embedding",
    description=f"{EmbeddingModelConfig.Meta.table_description}列表",
    summary=f"{EmbeddingModelConfig.Meta.table_description}列表",
)
async def get_embedding_model_config_list(
    request: Request,
    filter_: Annotated[EmbeddingModelConfigFilterSchema, Depends(EmbeddingModelConfigFilterSchema.as_query)],  # type: ignore
    pager: CRUDPager = pagination_factory(
        db_model=EmbeddingModelConfig,
        list_schema=EmbeddingModelConfigList,
        search_fields={"name", "description"},
        order_fields={
            "created_at",
            "name",
        },
        max_limit=1000,
    ),
) -> Resp[PageData[EmbeddingModelConfigList]]:
    return await list_view(get_embeddding_config_queryset(request), filter_, pager)


@router.get(
    "/embedding/{pk}",
    description=f"获取{EmbeddingModelConfig.Meta.table_description}详情",
    summary=f"获取{EmbeddingModelConfig.Meta.table_description}详情",
)
async def get_embedding_model_config_detail(request: Request, pk: int) -> Resp[EmbeddingModelConfigDetail]:
    queryset = get_embeddding_config_queryset(request)
    obj: EmbeddingModelConfig | None = await queryset.get_or_none(
        **{queryset.model._meta.pk_attr: pk},
    )
    if not obj:
        raise ApiException("对象不存在")
    obj = await obj_prefetch_fields(obj, EmbeddingModelConfigDetail)  # type: ignore
    data = EmbeddingModelConfigDetail.model_validate(obj)
    return Resp(data=data)


@router.post(
    "/embedding",
    description=f"创建{EmbeddingModelConfig.Meta.table_description}",
    summary=f"创建{EmbeddingModelConfig.Meta.table_description}",
)
async def create_embedding_model_config(request: Request, schema: EmbeddingModelConfigCreate) -> Resp:
    await create_obj(EmbeddingModelConfig, schema.model_dump(exclude_unset=True))
    return Resp()


@router.put(
    "/embedding/{pk}",
    description=f"更新{EmbeddingModelConfig.Meta.table_description}",
    summary=f"更新{EmbeddingModelConfig.Meta.table_description}",
)
async def update_embedding_model_config(request: Request, pk: int, schema: EmbeddingModelConfigUpdate) -> Resp:
    return await update_view(get_embeddding_config_queryset(request), pk, schema)  # type: ignore


@router.delete(
    "/embedding/{pk}",
    description=f"删除{EmbeddingModelConfig.Meta.table_description}",
    summary=f"删除{EmbeddingModelConfig.Meta.table_description}",
)
async def delete_embedding_model_config(request: Request, pk: int) -> Resp:
    return await delete_view(pk, get_embeddding_config_queryset(request))
