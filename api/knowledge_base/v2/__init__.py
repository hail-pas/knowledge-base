from fastapi import APIRouter, Depends

from service.depend import api_permission_check

# All v2 endpoints require a verified Pero identity and local RBAC permission.
router = APIRouter(prefix="/v2", dependencies=[Depends(api_permission_check)])
