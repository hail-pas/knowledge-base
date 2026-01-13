from api.service.resource.schema import ResourceLevelTreeNode, ResourceLevelTreeBaseNode
from ext.ext_tortoise.models.user_center import Resource


def resource_list_to_trees(nodes: list[Resource]) -> list[ResourceLevelTreeNode]:
    if not nodes:
        return []
    node_dict = {
        node.id: ResourceLevelTreeNode.model_validate(
            ResourceLevelTreeBaseNode.model_validate(node),
        )
        for node in nodes
    }
    tree_list = []
    for node in node_dict.values():
        if node.parent_id is None:  # type: ignore
            tree_list.append(node)
        else:
            parent_node = node_dict.get(node.parent_id)  # type: ignore
            if not parent_node:
                # 缺失父节点直接顶替
                tree_list.append(node)
                continue

            parent_node.children.append(node)
            parent_node.children.sort(key=lambda x: x.order_num)  # type: ignore
    tree_list.sort(key=lambda x: x.order_num)
    return tree_list
