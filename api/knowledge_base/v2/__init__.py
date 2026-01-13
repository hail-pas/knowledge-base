from fastapi import Depends, APIRouter

# from api.depend import api_permission_check

router = APIRouter(prefix="/v2") # dependencies=[Depends(api_permission_check)])
