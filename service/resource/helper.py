from service.resource.schema import ResourceLevelTreeNode, ResourceLevelTreeBaseNode
from ext.ext_tortoise.models.authorization import Permission, Resource


async def resource_list_to_trees(nodes: list[Resource]) -> list[ResourceLevelTreeNode]:
    """Build a resource tree and include the permission codes attached to each node."""

    if not nodes:
        return []

    resource_ids = [node.id for node in nodes]
    permission_rows = await Permission.filter(resources__id__in=resource_ids).values("code", "resources__id")
    permission_codes_by_resource: dict[int, list[str]] = {}
    for row in permission_rows:
        permission_codes_by_resource.setdefault(row["resources__id"], []).append(row["code"])

    node_dict: dict[int, ResourceLevelTreeNode] = {}
    for node in nodes:
        base = ResourceLevelTreeBaseNode.model_validate(node)
        base.permission_codes = sorted(permission_codes_by_resource.get(node.id, []))
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
