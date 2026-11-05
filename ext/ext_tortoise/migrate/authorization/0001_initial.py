from tortoise import migrations
from tortoise.migrations import operations as ops
from ext.ext_tortoise.base.fields import TimestampField
from ext.ext_tortoise.enums import ClientSceneEnum, PermissionTypeEnum, SystemResourceSubTypeEnum, SystemResourceTypeEnum
from tortoise.fields.base import OnDelete
from tortoise import fields

class Migration(migrations.Migration):
    initial = True

    operations = [
        ops.CreateModel(
            name='Permission',
            fields=[
                ('code', fields.CharField(primary_key=True, unique=True, db_index=True, description='权限编码', max_length=256)),
                ('label', fields.CharField(description='权限名称', max_length=128)),
                ('permission_type', fields.CharEnumField(default=PermissionTypeEnum.api, description='权限类型, api: API', enum_type=PermissionTypeEnum, max_length=16)),
                ('is_deprecated', fields.BooleanField(default=False, description='是否已废弃')),
            ],
            options={'table': 'permission', 'app': 'authorization', 'pk_attr': 'code', 'table_description': '权限'},
            bases=['Model'],
        ),
        ops.CreateModel(
            name='Role',
            fields=[
                ('id', fields.BigIntField(generated=True, primary_key=True, unique=True, db_index=True, description='主键')),
                ('created_at', fields.DatetimeField(db_index=True, description='创建时间', auto_now=False, auto_now_add=True)),
                ('updated_at', fields.DatetimeField(description='更新时间', auto_now=True, auto_now_add=False)),
                ('deleted_at', TimestampField(null=True, db_index=True, description='删除时间', read_only=True, index=True, timezone=zoneinfo.ZoneInfo(key='Asia/Shanghai'))),
                ('label', fields.CharField(description='角色名称', max_length=50)),
                ('remark', fields.CharField(null=True, description='备注', max_length=200)),
                ('is_super_admin', fields.BooleanField(default=False, description='是否为系统超级管理员')),
            ],
            options={'table': 'role', 'app': 'authorization', 'unique_together': (('label', 'deleted_at'),), 'pk_attr': 'id', 'table_description': '角色'},
            bases=['BaseModel'],
        ),
        ops.CreateModel(
            name='Resource',
            fields=[
                ('id', fields.BigIntField(generated=True, primary_key=True, unique=True, db_index=True, description='主键')),
                ('created_at', fields.DatetimeField(db_index=True, description='创建时间', auto_now=False, auto_now_add=True)),
                ('code', fields.CharField(db_index=True, description='资源编码', max_length=32)),
                ('label', fields.CharField(db_index=True, description='资源名称', max_length=64)),
                ('front_route', fields.CharField(null=True, description='前端路由', max_length=128)),
                ('resource_type', fields.CharEnumField(description='资源类型, menu: 菜单, button: 按钮, api: 接口', enum_type=SystemResourceTypeEnum, max_length=16)),
                ('sub_resource_type', fields.CharEnumField(description='资源子类型, add_tab: 选项卡, dialog: 弹窗, ajax: Ajax请求, link: 链接', enum_type=SystemResourceSubTypeEnum, max_length=16)),
                ('order_num', fields.IntField(default=1, description='排列序号')),
                ('enabled', fields.BooleanField(default=True, description='是否启用')),
                ('assignable', fields.BooleanField(default=True, description='是否允许分配')),
                ('parent', fields.ForeignKeyField('authorization.Resource', source_field='parent_id', null=True, description='父级资源', db_constraint=True, to_field='id', related_name='children', on_delete=OnDelete.CASCADE)),
                ('scene', fields.CharEnumField(description='客户端场景', enum_type=ClientSceneEnum, max_length=16)),
                ('permissions', fields.ManyToManyField('authorization.Permission', unique=True, db_constraint=True, through='resource_permission', forward_key='permission_id', backward_key='resource_id', related_name='resources', on_delete=OnDelete.CASCADE)),
                ('roles', fields.ManyToManyField('authorization.Role', unique=True, db_constraint=True, through='resource_role', forward_key='role_id', backward_key='resource_id', related_name='resources', on_delete=OnDelete.CASCADE)),
            ],
            options={'table': 'resource', 'app': 'authorization', 'unique_together': (('code', 'parent', 'scene'),), 'pk_attr': 'id', 'table_description': '系统资源'},
            bases=['BigIntegerIDPrimaryKeyModel', 'CreateOnlyModel'],
        ),
        ops.CreateModel(
            name='UserRole',
            fields=[
                ('id', fields.BigIntField(generated=True, primary_key=True, unique=True, db_index=True, description='主键')),
                ('created_at', fields.DatetimeField(db_index=True, description='创建时间', auto_now=False, auto_now_add=True)),
                ('updated_at', fields.DatetimeField(description='更新时间', auto_now=True, auto_now_add=False)),
                ('deleted_at', TimestampField(null=True, db_index=True, description='删除时间', read_only=True, index=True, timezone=zoneinfo.ZoneInfo(key='Asia/Shanghai'))),
                ('user_id', fields.UUIDField(db_index=True, description='Pero 用户 UUID')),
                ('role', fields.ForeignKeyField('authorization.Role', source_field='role_id', description='本地角色', db_constraint=True, to_field='id', related_name='user_assignments', on_delete=OnDelete.CASCADE)),
            ],
            options={'table': 'user_role', 'app': 'authorization', 'unique_together': (('user_id', 'role', 'deleted_at'),), 'pk_attr': 'id', 'table_description': '外部用户角色关联'},
            bases=['BaseModel'],
        ),
    ]
