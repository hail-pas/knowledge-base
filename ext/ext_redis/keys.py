from enum import Enum, unique


@unique
class GeneralCacheKey(str, Enum): ...


@unique
class UserCenterKey(str, Enum):
    AccountApiPermissionSet = "UC:Account:Apis:{uuid}"
    Token2AccountKey = "UC:Token:{token}"  # Authorization 拼接key， 存储的 acount_id:scene
    Account2TokenKey = "UC:Account:{account_id}:{scene}"  # 存储的 token:account_id
    AccountBaseInfo = "UC:Account:BaseInfo:{uuid}"
    CodeUniqueKey = "UC:Code:{scene}:{identifier}"

    ApiSecretKey = "ApiKey:SecretKey:{api_key}"  # ApiKey 密钥
    ApiKeyPermissionSet = "ApiKey:Apis:{api_key}"  # ApiKey接口权限
