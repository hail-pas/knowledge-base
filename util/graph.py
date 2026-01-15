import json
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import networkx as nx
import pydantic
import yaml

NodeName = str
ConfigFormat = Literal["yaml", "json", "dict"]
OptionsDict = Dict[str, Any]
ExecuteParamsDict = Dict[str, Any]
InputDict = Dict[str, Any]


class NodeConfig(pydantic.BaseModel):
    options: OptionsDict = pydantic.Field(default_factory=dict, description="节点选项")
    execute_params: ExecuteParamsDict = pydantic.Field(default_factory=dict, description="执行参数")
    depends_on: List[NodeName] = pydantic.Field(default_factory=list, description="依赖的父节点列表")
    input: InputDict = pydantic.Field(default_factory=dict, description="输入数据")

    model_config = pydantic.ConfigDict(extra="allow")


class NodeInfo(pydantic.BaseModel):
    node_name: NodeName
    options: OptionsDict = pydantic.Field(default_factory=dict)
    execute_params: ExecuteParamsDict = pydantic.Field(default_factory=dict)
    input: InputDict = pydantic.Field(default_factory=dict)

    parents: List[NodeName] = pydantic.Field(default_factory=list, description="直接父节点")
    children: List[NodeName] = pydantic.Field(default_factory=list, description="直接子节点")
    ancestors: List[NodeName] = pydantic.Field(default_factory=list, description="所有祖先节点")
    descendants: List[NodeName] = pydantic.Field(default_factory=list, description="所有后代节点")


class DagError(Exception):
    """DAG 基础错误"""
    pass


class DagConfigError(DagError):
    """配置错误（字段缺失/类型不对/引用不存在等）"""
    pass


class DagCycleError(DagError):
    """不是 DAG（存在环）"""
    pass


class DagValidationError(DagError):
    """DAG 验证错误"""
    pass


class GraphUtil:
    """DAG 图管理工具（基于NetworkX DiGraph，使用 Pydantic 进行配置验证）"""

    config_format: ConfigFormat
    deterministic: bool
    config: Dict[NodeName, NodeConfig]
    graph: nx.DiGraph
    activities: Dict[NodeName, NodeInfo]
    _topo_order: List[NodeName]

    def __init__(
        self,
        config: Union[str, Dict[NodeName, Union[NodeConfig, Dict[str, Any]]]],
        config_format: ConfigFormat = "dict",
        *,
        deterministic: bool = True,
    ) -> None:
        self.config_format = config_format
        self.deterministic = deterministic

        self.config = self._parse_config(config, config_format)

        self.graph = nx.DiGraph()
        self.activities = {}
        self._topo_order = []

        self._build_graph()
        self._build_node_infos()


    def _parse_config(
        self,
        config: Union[str, Dict[NodeName, Union[NodeConfig, Dict[str, Any]]]],
        config_format: ConfigFormat,
    ) -> Dict[NodeName, NodeConfig]:
        config_dict: Dict[str, Any]

        match config_format:
            case "yaml":
                if not isinstance(config, str):
                    raise DagConfigError("config_format='yaml' but config is not a string")
                try:
                    loaded = yaml.safe_load(config)
                except Exception as e:
                    raise DagConfigError(f"YAML parse error: {e}") from e
                config_dict = loaded or {}

            case "json":
                if not isinstance(config, str):
                    raise DagConfigError("config_format='json' but config is not a string")
                try:
                    loaded = json.loads(config)
                except Exception as e:
                    raise DagConfigError(f"JSON parse error: {e}") from e
                config_dict = loaded or {}

            case "dict":
                if not isinstance(config, dict):
                    raise DagConfigError("config_format='dict' but config is not a dict")
                config_dict = config  # type: ignore

            case _:
                raise DagConfigError(f"Unsupported config_format: {config_format}")

        if not isinstance(config_dict, dict) or not config_dict:
            raise DagConfigError(f"Config parsed error, raw config is: {config!r}")

        normalized: Dict[NodeName, NodeConfig] = {}

        for k, v in config_dict.items():
            if not isinstance(k, str) or not k.strip():
                raise DagConfigError(f"Node name must be a non-empty str, got: {k!r}")

            # v 允许是 NodeConfig 或 dict
            if isinstance(v, NodeConfig):
                node_config = v
            elif isinstance(v, dict):
                try:
                    node_config = NodeConfig(**v)
                except pydantic.ValidationError as e:
                    raise DagConfigError(f"Node '{k}' config validation failed: {e}") from e
            else:
                raise DagConfigError(f"Node '{k}' config must be dict or NodeConfig, got: {type(v)}")

            # depends_on 去重 + 校验
            validated: List[NodeName] = []
            seen: Set[NodeName] = set()
            for dep in node_config.depends_on:
                if not isinstance(dep, str) or not dep.strip():
                    raise DagConfigError(f"Node '{k}' has invalid dependency: {dep!r}")
                if dep not in seen:
                    seen.add(dep)
                    validated.append(dep)

            node_config.depends_on = validated
            normalized[k] = node_config

        return normalized

    def export_config(self) -> Dict[str, Any]:
        """导出为 Python dict（可用于持久化/打印/再构建）"""
        out: Dict[str, Any] = {}
        # 稳定输出顺序：按 topo 或字典序
        nodes = self._topo_order if self._topo_order else sorted(self.config.keys())
        for name in nodes:
            cfg = self.config[name]
            out[name] = cfg.model_dump()
        return out

    def export_config_json(self, *, indent: Optional[int] = None) -> str:
        config_dict = self.export_config()
        return json.dumps(config_dict, indent=indent, ensure_ascii=False)

    def export_config_yaml(self) -> str:
        config_dict = self.export_config()
        return yaml.dump(config_dict, allow_unicode=True, sort_keys=False)

    def _build_graph(self) -> None:
        # 只添加配置中显式存在的节点
        for node_name in self.config.keys():
            self.graph.add_node(node_name)

        # 依赖建边（严格校验 parent 必须存在，避免“幽灵节点”）
        for node_name, node_config in self.config.items():
            for parent in node_config.depends_on:
                if parent not in self.graph:
                    raise DagConfigError(f"Node '{node_name}' depends on unknown node '{parent}'")
                if parent == node_name:
                    raise DagConfigError(f"Node '{node_name}' cannot depend on itself")
                self.graph.add_edge(parent, node_name)

        # 验证 DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycle = nx.find_cycle(self.graph, orientation="original")
                raise DagCycleError(f"The graph contains a cycle: {cycle}")
            except nx.NetworkXNoCycle:
                raise DagCycleError("The graph contains cycles and is not a valid DAG")

    def _build_node_infos(self) -> None:
        """构建 NodeInfo（parents/children/ancestors/descendants 等缓存）"""
        self.activities = {}

        for node_name, node_config in self.config.items():
            parents = list(self.graph.predecessors(node_name))
            children = list(self.graph.successors(node_name))

            ancestors = list(nx.ancestors(self.graph, node_name))
            descendants = list(nx.descendants(self.graph, node_name))

            if self.deterministic:
                parents.sort()
                children.sort()
                ancestors.sort()
                descendants.sort()

            self.activities[node_name] = NodeInfo(
                node_name=node_name,
                options=node_config.options,
                execute_params=node_config.execute_params,
                input=node_config.input,
                parents=parents,
                children=children,
                ancestors=ancestors,
                descendants=descendants,
            )

    def has_node(self, node_name: NodeName) -> bool:
        return node_name in self.graph

    def get_node_info(self, node_name: NodeName) -> NodeInfo:
        if node_name not in self.activities:
            raise DagConfigError(f"Node '{node_name}' not found")
        return self.activities[node_name]

    def get_root_nodes(self) -> List[NodeName]:
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        if self.deterministic:
            roots.sort()
        return roots

    def get_leaf_nodes(self) -> List[NodeName]:
        leaves = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        if self.deterministic:
            leaves.sort()
        return leaves

    def get_node_dependencies(self, node_name: NodeName) -> Dict[str, List[NodeName]]:
        info = self.get_node_info(node_name)
        return {
            "parents": list(info.parents),
            "children": list(info.children),
            "ancestors": list(info.ancestors),
            "descendants": list(info.descendants),
        }

    def get_layers(self) -> List[List[NodeName]]:
        """
        按层级（level）返回节点：List[List[str]]

        - 第 0 层：所有入度为 0 的根节点
        - 第 k 层：所有依赖节点都在 <k 层里已出现的节点

        每一层内的节点可以并发执行。
        deterministic=True 时：每层节点按字典序排序，保证稳定可复现。
        """
        in_deg = {n: self.graph.in_degree(n) for n in self.graph.nodes()}

        ready = [n for n, d in in_deg.items() if d == 0]
        if self.deterministic:
            ready.sort()

        layers: List[List[NodeName]] = []
        visited_count = 0

        while ready:
            # 当前层就是当前所有 ready 节点
            layer = list(ready)
            layers.append(layer)
            visited_count += len(layer)

            # 计算下一层
            next_ready: List[NodeName] = []
            for n in layer:
                for child in self.graph.successors(n):
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        next_ready.append(child)

            if self.deterministic:
                next_ready.sort()
            ready = next_ready

        # 如果没覆盖所有节点，说明图有环（理论上 build_graph 已校验，这里做防御）
        if visited_count != self.graph.number_of_nodes():
            raise DagCycleError("Unexpected: layering failed (cycle suspected)")

        return layers


    def is_node_ready(self, node_name: NodeName, completed_nodes: Set[NodeName]) -> bool:
        if node_name not in self.graph:
            raise DagConfigError(f"Node '{node_name}' not found in graph")
        for parent in self.graph.predecessors(node_name):
            if parent not in completed_nodes:
                return False
        return True

    def get_ready_nodes(self, completed_nodes: Set[NodeName]) -> List[NodeName]:
        ready: List[NodeName] = []
        for node in self.graph.nodes():
            if node in completed_nodes:
                continue
            if self.is_node_ready(node, completed_nodes):
                ready.append(node)
        if self.deterministic:
            ready.sort()
        return ready
