from enum import Enum, unique


@unique
class RedisCacheKey(str, Enum):
    """Redis keys unrelated to user-account lifecycle."""

    ApiSecretKey = "ApiKey:SecretKey:{api_key}"
    ApiKeyPermissionSet = "ApiKey:Apis:{api_key}"
