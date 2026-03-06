from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `document_chunk` ADD `manual_add` BOOL NOT NULL COMMENT '是否手动添加' DEFAULT 0;
        ALTER TABLE `file_source` MODIFY COLUMN `storage_location` VARCHAR(1000) COMMENT '存储位置  - type=local_file: 本地文件路径（如 /data/documents）  - type=s3/minio/aliyun_oss: 存储桶名称（如 my-bucket）, sharepoint: 站点路径, api: API路径';"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE `document_chunk` DROP COLUMN `manual_add`;
        ALTER TABLE `file_source` MODIFY COLUMN `storage_location` VARCHAR(1000) COMMENT '存储位置  - type=local_file: 本地文件路径（如 /data/documents）  - type=s3/minio/aliyun_oss: 存储桶名称（如 my-bucket）';"""
