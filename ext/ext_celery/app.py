from config.main import local_configs

celery_app = local_configs.extensions.celery.instance


def get_celery_app():
    """获取 Celery 应用实例

    Returns:
        Celery: Celery 应用实例
    """
    return celery_app
