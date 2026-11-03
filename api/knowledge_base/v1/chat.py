"""Knowledge-base chat API placeholder.

TODO: implement RAG chat after retrieval policy, conversation persistence,
streaming response, citation format, and LLM orchestration are specified.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from service.depend import api_permission_check

router = APIRouter(dependencies=[Depends(api_permission_check)])


@router.post("/completions", summary="基于知识库的聊天（规划中）")
async def create_chat_completion() -> None:
    """Reserved endpoint for future retrieval-augmented chat."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Knowledge-base chat is planned but not implemented yet.",
    )
