"""
监控和调试 httpx 连接池的工具类
"""

import asyncio
import httpx
from typing import Dict, Any
from loguru import logger


class HttpxConnectionPoolMonitor:
    """httpx 连接池监控器"""

    @staticmethod
    async def get_client_stats(client: httpx.AsyncClient | None) -> dict[str, Any]:
        """获取 httpx 客户端的统计信息

        Args:
            client: httpx.AsyncClient 实例

        Returns:
            包含连接池统计信息的字典
        """

        if client is None:
            return {"status": "not_initialized", "message": "Httpx client not initialized"}

        stats = {
            "status": "active",
            "client_type": str(type(client).__name__),
        }

        # 获取连接池信息
        if hasattr(client, "_transport"):
            transport = client._transport
            if hasattr(transport, "_pool"):
                pool = transport._pool  # type: ignore
                stats["pool"] = {  # type: ignore
                    "max_connections": pool._max_connections if hasattr(pool, "_max_connections") else "unknown",
                    "max_keepalive": pool._max_keepalive_connections
                    if hasattr(pool, "_max_keepalive_connections")
                    else "unknown",
                }

                # 尝试获取当前连接数
                if hasattr(pool, "_connections"):
                    stats["pool"]["active_connections"] = len(pool._connections)
                elif hasattr(pool, "_idle_connections"):
                    stats["pool"]["idle_connections"] = len(pool._idle_connections)

        return stats

    @staticmethod
    async def log_stats(client: httpx.AsyncClient | None):
        """打印连接池统计信息到日志

        Args:
            client: httpx.AsyncClient 实例
        """
        stats = await HttpxConnectionPoolMonitor.get_client_stats(client)
        logger.info("=== HTTPX Connection Pool Stats ===")
        logger.info(f"Status: {stats['status']}")

        if "pool" in stats:
            pool_stats = stats["pool"]
            logger.info(f"Max Connections: {pool_stats.get('max_connections', 'unknown')}")
            logger.info(f"Max Keepalive: {pool_stats.get('max_keepalive', 'unknown')}")

            if "active_connections" in pool_stats:
                logger.info(f"Active Connections: {pool_stats['active_connections']}")
            if "idle_connections" in pool_stats:
                logger.info(f"Idle Connections: {pool_stats['idle_connections']}")

        logger.info("===================================")

    @staticmethod
    async def monitor_pool(client: httpx.AsyncClient | None, interval: float = 60.0, duration: float = 3600.0):
        """持续监控连接池状态

        Args:
            client: httpx.AsyncClient 实例
            interval: 监控间隔（秒）
            duration: 监控总时长（秒）

        示例:
            # 在后台任务中监控连接池
            await monitor_pool(client, interval=30, duration=300)  # 监控 5 分钟，每 30 秒一次
        """
        logger.info(f"Starting connection pool monitoring (interval={interval}s, duration={duration}s)")

        end_time = asyncio.get_event_loop().time() + duration

        while asyncio.get_event_loop().time() < end_time:
            await HttpxConnectionPoolMonitor.log_stats(client)
            await asyncio.sleep(interval)

        logger.info("Connection pool monitoring completed")

    @staticmethod
    def recommend_config(
        current_max_connections: int, current_keepalive: int, active_connections: int, idle_connections: int,
    ) -> dict[str, Any]:
        """
        根据当前连接使用情况推荐配置

        Args:
            current_max_connections: 当前最大连接数
            current_keepalive: 当前 keepalive 连接数
            active_connections: 当前活动连接数
            idle_connections: 当前空闲连接数

        Returns:
            推荐配置和建议
        """
        recommendations = {
            "current_config": {
                "max_connections": current_max_connections,
                "max_keepalive_connections": current_keepalive,
            },
            "current_usage": {
                "active_connections": active_connections,
                "idle_connections": idle_connections,
                "total_connections": active_connections + idle_connections,
            },
            "recommendations": [],
            "issues": [],
        }

        # 检查连接利用率
        utilization = (active_connections / current_max_connections) * 100
        if utilization > 80:
            recommendations["issues"].append(f"⚠️  连接利用率过高 ({utilization:.1f}%)，建议增加 max_connections")
            recommended_max = int(current_max_connections * 1.5)
            recommendations["recommendations"].append(f"建议将 max_connections 调整为 {recommended_max}")

        # 检查 keepalive 配置
        keepalive_ratio = (current_keepalive / current_max_connections) * 100
        if keepalive_ratio < 30:
            recommendations["recommendations"].append(
                f"💡 keepalive 连接数过低 ({keepalive_ratio:.1f}%)，" + "建议调整为 max_connections 的 40-50%",
            )
            recommended_keepalive = int(current_max_connections * 0.4)
            recommendations["recommendations"].append(
                f"建议将 max_keepalive_connections 调整为 {recommended_keepalive}",
            )

        # 检查空闲连接
        idle_ratio = (idle_connections / current_keepalive) * 100 if current_keepalive > 0 else 0
        if idle_ratio > 90:
            recommendations["recommendations"].append(
                f"💡 空闲连接过多 ({idle_ratio:.1f}%)，" + "考虑减少 max_keepalive_connections",
            )

        if not recommendations["issues"] and not recommendations["recommendations"]:
            recommendations["recommendations"].append("✅ 当前配置合理，无需调整")

        return recommendations


# ============================================
# FastAPI 集成示例
# ============================================

"""
在 FastAPI 路由中使用监控器:

from fastapi import APIRouter, Depends
from ext.ext_httpx.monitor import HttpxConnectionPoolMonitor
from ext.ext_httpx.main import HttpxConfig

router = APIRouter()

def get_httpx_client():
    '''获取 httpx 客户端'''
    httpx_config = HttpxConfig()
    return httpx_config.instance

@router.get("/admin/httpx-stats")
async def get_httpx_stats(client: httpx.AsyncClient = Depends(get_httpx_client)):
    '''获取 httpx 连接池统计信息'''
    stats = await HttpxConnectionPoolMonitor.get_client_stats(client)
    return stats

@router.post("/admin/httpx-recommend")
async def get_httpx_recommendations(
    current_max_connections: int,
    current_keepalive: int,
    active_connections: int,
    idle_connections: int
):
    '''获取配置推荐'''
    recommendations = HttpxConnectionPoolMonitor.recommend_config(
        current_max_connections,
        current_keepalive,
        active_connections,
        idle_connections
    )
    return recommendations
"""
