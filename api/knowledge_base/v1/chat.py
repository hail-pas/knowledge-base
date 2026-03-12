from fastapi import Depends, APIRouter

from service.depend import api_permission_check

router = APIRouter(dependencies=[Depends(api_permission_check)])
