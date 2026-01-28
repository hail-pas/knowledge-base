from typing import override
from celery import Celery
from pydantic import BaseModel, RedisDsn
from config.default import EnvironmentEnum, InstanceExtensionConfig

_CELERY_APP: Celery | None = None

class DefaultConfig(BaseModel):
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list = ["json"]

    # 时区设置
    timezone: str = "Asia/Shanghai"
    enable_utc: bool = True

    # 结果过期时间 (1天)
    result_expires: int = 86400

    # 任务默认配置
    task_default_queue: str = "default"
    task_default_priority: int = 5
    task_default_routing_key: str = "default"

    # 任务执行超时 (单位：秒)
    task_time_limit: int = 3600  # 1小时
    task_soft_time_limit: int = 3000  # 50分钟

    # 工作者配置
    worker_prefetch_multiplier: int = 4
    worker_concurrency: int = 4
    worker_max_tasks_per_child: int = 1000

    # 任务重试配置
    task_autoretry_for: tuple = (Exception,)
    task_retry_backoff: bool = True
    task_retry_backoff_max: int = 600  # 最大退避时间 10 分钟
    task_retry_jitter: bool = True
    task_default_max_retries: int = 3

    # 结果后端配置
    result_backend_transport_options: dict[str, dict | str | int] = {
        "retry_policy": {
            "timeout": 5.0,
            "max_retries": 3,
        },
        "visibility_timeout": 3600,
    }

    # 幂等性支持
    # 使用 task_id 作为幂等键
    task_track_started: bool = True
    task_publish_retry: bool = True
    task_publish_retry_policy: dict[str, float] = {
        "max_retries": 3,
        "interval_start": 0,
        "interval_step": 0.2,
        "interval_max": 0.5,
    }

    # 监控和日志
    task_send_sent_event: bool = True
    task_send_fail_event: bool = True
    task_send_retry_event: bool = True
    task_send_success_event: bool = True

    # 任务结果模式
    result_extended: bool = True

    # 任务压缩
    task_compression: str = "gzip"
    result_compression: str = "gzip"

    # Celery Beat 配置 (用于定时任务)
    beat_schedule_filename: str = "celerybeat-schedule"
    beat_sync_every: int = 0  # 不自动同步

    # 信号处理
    worker_proc_alive_threshold: int = 600  # 10 分钟


class DevCeleryConfig(DefaultConfig):
    """开发环境配置"""
    worker_concurrency: int = 2
    task_default_max_retries: int = 2

# 生产环境配置
class ProdCeleryConfig(DefaultConfig):
    """生产环境配置"""
    worker_concurrency: int = 8
    worker_prefetch_multiplier: int = 8
    task_default_max_retries: int = 2
    task_time_limit: int = 7200  # 2小时
    task_soft_time_limit: int = 6600  # 110分钟


# 根据环境选择配置
def get_celery_config() -> DefaultConfig:
    """根据环境获取 Celery 配置"""
    from config.main import local_configs


    if local_configs.project.environment == EnvironmentEnum.production:
        return ProdCeleryConfig()

    return DevCeleryConfig()


class CeleryConfig(InstanceExtensionConfig[Celery]):
    broker_url: RedisDsn
    result_backend: RedisDsn

    @property
    @override
    def instance(self) -> Celery: # type: ignore
        global _CELERY_APP

        if _CELERY_APP:
            return _CELERY_APP

        _CELERY_APP = Celery(
            "knowledge_base",
            broker=self.broker_url.encoded_string(),
            backend=self.result_backend.encoded_string(),
        )

        celery_config = get_celery_config()

        # 定义任务路由（Redis broker 不需要 routing_key）
        task_routes = {
            # 'workflow.activity_handoff': {
            #     'queue': 'workflow_activity_handoff',
            # },
            # "workflow.schedule_start": {
            #     "queue": "workflow_entry"
            # },
            # "workflow.schedule_resume": {
            #     "queue": "workflow_entry"
            # },
            "workflow.*": {
                "queue": "default",
            },
        }

        _CELERY_APP.conf.update(
            # 序列化配置
            task_serializer=celery_config.task_serializer,
            result_serializer=celery_config.result_serializer,
            accept_content=celery_config.accept_content,

            # 时区配置
            timezone=celery_config.timezone,
            enable_utc=celery_config.enable_utc,

            # 结果配置
            result_expires=celery_config.result_expires,

            # 任务路由
            task_routes=task_routes,
            task_default_queue=celery_config.task_default_queue,
            task_default_priority=celery_config.task_default_priority,
            task_default_routing_key=celery_config.task_default_routing_key,

            # 超时配置
            task_time_limit=celery_config.task_time_limit,
            task_soft_time_limit=celery_config.task_soft_time_limit,

            # 工作者配置
            worker_prefetch_multiplier=celery_config.worker_prefetch_multiplier,
            worker_concurrency=celery_config.worker_concurrency,
            worker_max_tasks_per_child=celery_config.worker_max_tasks_per_child,

            # 重试配置
            task_autoretry_for=celery_config.task_autoretry_for,
            task_retry_backoff=celery_config.task_retry_backoff,
            task_retry_backoff_max=celery_config.task_retry_backoff_max,
            task_retry_jitter=celery_config.task_retry_jitter,
            task_default_max_retries=celery_config.task_default_max_retries,

            # Broker 连接重试配置（Celery 6.0+）
            broker_connection_retry_on_startup=True,

            # SSL 配置（用于 Redis 连接）
            broker_use_ssl={
                # 'ssl_cert_reqs': 0,  # 0 = ssl.CERT_NONE, 不验证证书
                #
                "ssl_cert_reqs": 2,  # 2 = ssl.CERT_REQUIRED
                "ssl_ca_certs": self.broker_url.query,
            },
            # result_backend_use_ssl={
            #     'ssl_cert_reqs': 0,
            # },

            # 结果后端配置
            result_backend_transport_options=celery_config.result_backend_transport_options,

            # 幂等性和追踪
            task_track_started=celery_config.task_track_started,
            task_publish_retry=celery_config.task_publish_retry,
            task_publish_retry_policy=celery_config.task_publish_retry_policy,

            # 事件追踪
            task_send_sent_event=celery_config.task_send_sent_event,
            task_send_fail_event=celery_config.task_send_fail_event,
            task_send_retry_event=celery_config.task_send_retry_event,
            task_send_success_event=celery_config.task_send_success_event,

            # 结果扩展
            result_extended=celery_config.result_extended,

            # 压缩配置
            task_compression=celery_config.task_compression,
            result_compression=celery_config.result_compression,

            # 任务限制
            # task_annotations=celery_config.task_annotations,

            # Celery Beat 配置
            beat_schedule_filename=celery_config.beat_schedule_filename,
            beat_sync_every=celery_config.beat_sync_every,

            # 信号处理
            worker_proc_alive_threshold=celery_config.worker_proc_alive_threshold,
        )

        return _CELERY_APP
