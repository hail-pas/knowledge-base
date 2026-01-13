from pathlib import Path

from ext.ext_tortoise.main import gen_tortoise_config_dict

TORTOISE_ORM_CONFIG = gen_tortoise_config_dict()


VERSION_FILE_PATH = str(Path(__file__).resolve().parent.absolute())

# 写配置块 [tool.aerich] 到 pyproject.toml
# aerich init -t ext.ext_tortoise.migrate.env.TORTOISE_ORM_CONFIG --location ext/ext_tortoise/migrate
# 初始化迁移目录和记录表
# aerich --app {app} init-db
# aerich --app {app} migrate
# aerich --app {app} upgrade
