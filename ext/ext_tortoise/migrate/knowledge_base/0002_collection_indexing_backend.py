from tortoise import migrations, fields
from tortoise.migrations import operations as ops
from tortoise.fields.base import OnDelete


class Migration(migrations.Migration):
    dependencies = [("knowledge_base", "0001_initial")]
    initial = False

    operations = [
        ops.AddField(
            model_name="Collection",
            name="indexing_backend_config",
            field=fields.ForeignKeyField(
                "knowledge_base.IndexingBackendConfig",
                source_field="indexing_backend_config_id",
                null=True,
                description="关联索引后端配置；为空时使用系统默认后端",
                db_constraint=True,
                to_field="id",
                related_name="collections",
                on_delete=OnDelete.RESTRICT,
            ),
        ),
    ]
