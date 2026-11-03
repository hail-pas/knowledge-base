from service.resource.schema import ResourceLevelTreeNode, ResourceLevelTreeBaseNode
from ext.ext_tortoise.models.authorization import Resource


async def resource_list_to_trees(nodes: list[Resource]) -> list[ResourceLevelTreeNode]:
    """Build a resource tree and include the permission codes attached to each node."""

    if not nodes:
        return []

    node_dict: dict[int, ResourceLevelTreeNode] = {}
    for node in nodes:
        permissions = await node.permissions.all()
        base = ResourceLevelTreeBaseNode.model_validate(node)
        base.permission_codes = [permission.code for permission in permissions]
        node_dict[node.id] = ResourceLevelTreeNode.model_validate(base.model_dump())

    tree_list: list[ResourceLevelTreeNode] = []
    for node in node_dict.values():
        if node.parent_id is None:
            tree_list.append(node)
            continue

        parent_node = node_dict.get(node.parent_id)
        if parent_node is None:
            tree_list.append(node)
            continue
        parent_node.children.append(node)
        parent_node.children.sort(key=lambda item: item.order_num)

    tree_list.sort(key=lambda item: item.order_num)
    return tree_list
