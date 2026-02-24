from fastapi import Depends, APIRouter

from service.depend import api_permission_check

router = APIRouter(prefix="/v2", dependencies=[Depends(api_permission_check)])
