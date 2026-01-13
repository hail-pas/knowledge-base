from typing import AsyncIterator
from datetime import datetime

import httpx
from ext.file_source.base import FileSourceAdapter, FileItem



class SharePointAdapter(FileSourceAdapter):
    """SharePoint 适配器

    通过 Microsoft Graph API 访问 SharePoint 文件。
    授权 token 从 Redis 中获取。

    Config 格式:
    {
        "site_url": "https://contoso.sharepoint.com/sites/site_name",
        "redis_key": "sharepoint_token:user_id",  # Redis 中 token 的 key
        "drive_name": "Documents"  # 可选，默认为 Documents
    }
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.site_url = config.get("site_url")
        self.redis_key = config.get("redis_key")
        self.drive_name = config.get("drive_name", "Documents")
        self._access_token = None
        self._client = None

    async def _get_token_from_redis(self) -> str:
        """从 Redis 获取访问 token

        Returns:
            访问 token

        Raises:
            ValueError: Redis key 未配置或 token 不存在
        """
        from config.main import local_configs
        if not self.redis_key:
            raise ValueError("SharePoint 配置中缺少 redis_key")

        async with local_configs.extensions.redis.instance as redis_client:
            token = await redis_client.get(self.redis_key)
            if not token:
                raise ValueError(f"无法从 Redis 获取 token, key: {self.redis_key}")

        if isinstance(token, bytes):
            token = token.decode('utf-8')

        return token

    async def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端（带认证）"""
        if self._client is None:
            token = await self._get_token_from_redis()
            self._access_token = token
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=30.0
            )
        return self._client

    async def validate(self) -> bool:
        """验证配置是否正确"""
        if not self.site_url:
            return False

        try:
            client = await self._get_client()
            # 尝试获取站点信息验证连接
            response = await client.get(f"{self.site_url}/_api/web")
            return response.status_code == 200
        except Exception:
            return False

    async def get_file(self, uri: str) -> bytes:
        """获取单文件内容

        Args:
            uri: SharePoint 文件的服务器相对路径
                  例如: "/sites/site_name/Shared Documents/folder/file.pdf"

        Returns:
            文件内容（字节数组）

        Raises:
            FileNotFoundError: 文件不存在
            PermissionError: 无访问权限
        """
        client = await self._get_client()

        # 构建文件 API URL
        # 使用服务器相对路径
        api_url = f"{self.site_url}/_api/web/GetFileByServerRelativeUrl('{uri}')/$value"

        response = await client.get(api_url)

        if response.status_code == 404:
            raise FileNotFoundError(f"SharePoint 文件不存在: {uri}")
        elif response.status_code == 401:
            raise PermissionError(f"无权限访问 SharePoint 文件: {uri}")
        elif response.status_code != 200:
            raise RuntimeError(f"获取 SharePoint 文件失败: {response.status_code} - {response.text}")

        return response.content

    async def get_file_stream(self, uri: str) -> AsyncIterator[bytes]: # type: ignore
        """获取文件流（大文件）

        Args:
            uri: SharePoint 文件的服务器相对路径

        Yields:
            文件内容分块（8KB）
        """
        client = await self._get_client()

        api_url = f"{self.site_url}/_api/web/GetFileByServerRelativeUrl('{uri}')/$value"

        async with client.stream("GET", api_url) as response:
            if response.status_code == 404:
                raise FileNotFoundError(f"SharePoint 文件不存在: {uri}")
            elif response.status_code == 401:
                raise PermissionError(f"无权限访问 SharePoint 文件: {uri}")
            elif response.status_code != 200:
                raise RuntimeError(f"获取 SharePoint 文件失败: {response.status_code}")

            async for chunk in response.aiter_bytes(chunk_size=8192):
                yield chunk

    async def get_file_meta(self, uri: str) -> FileItem:
        """获取文件元数据

        Args:
            uri: SharePoint 文件的服务器相对路径

        Returns:
            文件元数据

        Raises:
            FileNotFoundError: 文件不存在
        """
        client = await self._get_client()

        # 获取文件属性
        api_url = f"{self.site_url}/_api/web/GetFileByServerRelativeUrl('{uri}')"

        response = await client.get(api_url)

        if response.status_code == 404:
            raise FileNotFoundError(f"SharePoint 文件不存在: {uri}")
        elif response.status_code != 200:
            raise RuntimeError(f"获取 SharePoint 文件元数据失败: {response.status_code}")

        data = response.json()
        data["d"]  # SharePoint 返回的数据在 d 键中

        # 从 URI 中提取文件名
        file_name = data["d"].get("Name", uri.split("/")[-1])
        length = data["d"].get("Length", 0)
        time_created = data["d"].get("TimeCreated")
        time_last_modified = data["d"].get("TimeLastModified")

        # 解析时间
        last_modified = datetime.fromisoformat(
            time_last_modified.replace("Z", "+00:00")
        ) if time_last_modified else datetime.now()

        return FileItem(
            uri=uri,
            name=file_name,
            size=length,
            content_type=data["d"].get("ListItemAllFields", {}).get("File_x0020_Type", "application/octet-stream"),
            last_modified=last_modified,
            metadata={
                "server_relative_url": data["d"].get("ServerRelativeUrl"),
                "encoding_url": data["d"].get("EncodingUrl"),
            }
        )

    async def list_files(self, prefix: str = "", filter=None) -> AsyncIterator[FileItem]: # type: ignore
        """列出文件夹中的文件

        Args:
            prefix: 文件夹路径（服务器相对路径）
                  例如: "/sites/site_name/Shared Documents/folder"
            filter: 文件过滤条件（暂未实现）

        Yields:
            文件项
        """
        client = await self._get_client()

        # 构建文件夹 API URL
        folder_url = f"{self.site_url}/_api/web/GetFolderByServerRelativeUrl('{prefix}')/Files"

        response = await client.get(folder_url)

        if response.status_code == 404:
            raise FileNotFoundError(f"SharePoint 文件夹不存在: {prefix}")
        elif response.status_code != 200:
            raise RuntimeError(f"列出 SharePoint 文件失败: {response.status_code}")

        data = response.json()

        for file_data in data.get("d", {}).get("results", []):
            file_name = file_data.get("Name")
            server_relative_url = file_data.get("ServerRelativeUrl")
            length = file_data.get("Length", 0)
            time_last_modified = file_data.get("TimeLastModified")

            last_modified = datetime.fromisoformat(
                time_last_modified.replace("Z", "+00:00")
            ) if time_last_modified else datetime.now()

            yield FileItem(
                uri=server_relative_url,
                name=file_name,
                size=length,
                content_type="application/octet-stream",  # SharePoint 文件列表可能不返回 MIME 类型
                last_modified=last_modified,
                metadata={"server_relative_url": server_relative_url}
            )

    async def close(self):
        """关闭客户端连接"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def check_file_exists(self, uri: str) -> bool:
        """检查文件是否存在

        Args:
            uri: SharePoint 文件的服务器相对路径

        Returns:
            文件是否存在
        """
        try:
            client = await self._get_client()
            api_url = f"{self.site_url}/_api/web/GetFileByServerRelativeUrl('{uri}')"
            response = await client.get(api_url)
            return response.status_code == 200
        except Exception:
            return False

    async def upload_file(
        self,
        uri: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict | None = None
    ) -> bool:
        """上传文件到 SharePoint

        SharePoint 通过 Graph API 或 REST API 上传文件。

        Args:
            uri: SharePoint 文件的服务器相对路径
            content: 文件内容
            content_type: MIME 类型
            metadata: 元数据（暂未使用）

        Returns:
            是否上传成功

        Raises:
            NotImplementedError: SharePoint 上传功能暂未实现
        """
        raise NotImplementedError("SharePoint 文件上传功能暂未实现")

    async def delete_file(self, uri: str) -> bool:
        """删除 SharePoint 中的文件

        Args:
            uri: SharePoint 文件的服务器相对路径

        Returns:
            是否删除成功

        Raises:
            NotImplementedError: SharePoint 删除功能暂未实现
        """
        raise NotImplementedError("SharePoint 文件删除功能暂未实现")
