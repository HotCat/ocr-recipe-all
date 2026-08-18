from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import copy
from time import perf_counter
from typing import Any, Callable


class PortType(str, Enum):
    FRAME = "frame"
    IMAGE = "image"
    CAMERA_SETTINGS = "camera_settings"
    TRIGGER = "trigger"
    TEXT = "text"
    VERDICT = "verdict"
    RESULT = "result"


@dataclass(frozen=True)
class PortSpec:
    name: str
    data_type: PortType
    required: bool = True
    multiple: bool = False


@dataclass
class NodeResult:
    value: Any
    preview: Any = None
    summary: str = ""
    elapsed_ms: float = 0.0
    skipped: bool = False


Processor = Callable[[dict[str, Any], dict[str, Any]], NodeResult]


@dataclass(frozen=True)
class NodeDefinition:
    type_name: str
    title: str
    category: str
    inputs: tuple[PortSpec, ...]
    output: PortSpec
    default_params: dict[str, Any]
    processor: Processor
    realtime_safe: bool = False


@dataclass
class RecipeNode:
    node_id: str
    type_name: str
    title: str
    x: float
    y: float
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    target_port: str


class GraphError(ValueError):
    pass


class RecipeGraph:
    FORMAT = "openfrp-vision/recipe/v1"

    def __init__(self, definitions: dict[str, NodeDefinition], revision: int = 0) -> None:
        self.definitions = definitions
        self.revision = revision
        self.nodes: dict[str, RecipeNode] = {}
        self.edges: list[Edge] = []
        self.results: dict[str, NodeResult] = {}

    def add_node(self, node: RecipeNode) -> None:
        if node.node_id in self.nodes:
            raise GraphError(f"Duplicate node id: {node.node_id}")
        if node.type_name not in self.definitions:
            raise GraphError(f"Unknown node type: {node.type_name}")
        defaults = self.definitions[node.type_name].default_params
        node.params = {**defaults, **node.params}
        self.nodes[node.node_id] = node
        self.revision += 1

    def remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes.pop(node_id, None)
            self.edges = [edge for edge in self.edges if edge.source != node_id and edge.target != node_id]
            self.results.pop(node_id, None)
            self.revision += 1

    def set_node_enabled(self, node_id: str, enabled: bool) -> None:
        node = self.nodes.get(node_id)
        if node is not None and node.enabled != enabled:
            node.enabled = enabled
            self.results.clear()
            self.revision += 1

    def connect(self, source: str, target: str, target_port: str) -> None:
        if source == target:
            raise GraphError("A node cannot connect to itself")
        if source not in self.nodes or target not in self.nodes:
            raise GraphError("Both connection endpoints must exist")

        source_def = self.definitions[self.nodes[source].type_name]
        target_def = self.definitions[self.nodes[target].type_name]
        port = next((item for item in target_def.inputs if item.name == target_port), None)
        if port is None:
            raise GraphError(f"{target_def.title} has no input named {target_port}")
        if source_def.output.data_type != port.data_type:
            raise GraphError(f"{source_def.output.data_type.value} cannot connect to {port.data_type.value}")

        old_edges = list(self.edges)
        if not port.multiple:
            self.edges = [edge for edge in self.edges if not (edge.target == target and edge.target_port == target_port)]
        candidate = Edge(source, target, target_port)
        changed = candidate not in self.edges
        if candidate not in self.edges:
            self.edges.append(candidate)
        try:
            self.topological_order()
        except GraphError:
            self.edges = old_edges
            raise
        if changed:
            self.revision += 1

    def disconnect(self, edge: Edge) -> None:
        if edge in self.edges:
            self.edges.remove(edge)
            self.revision += 1

    def topological_order(self) -> list[str]:
        indegree = {node_id: 0 for node_id in self.nodes}
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            if edge.source not in indegree or edge.target not in indegree:
                raise GraphError("Graph contains an edge with a missing endpoint")
            indegree[edge.target] += 1
            outgoing[edge.source].append(edge.target)

        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(order) != len(self.nodes):
            raise GraphError("Connections would create a cycle")
        return order

    def validate(self) -> list[str]:
        errors: list[str] = []
        try:
            self.topological_order()
        except GraphError as exc:
            errors.append(str(exc))
        for node in self.nodes.values():
            if not node.enabled:
                continue
            definition = self.definitions[node.type_name]
            for port in definition.inputs:
                incoming = [edge for edge in self.edges if edge.target == node.node_id and edge.target_port == port.name]
                if port.required and not incoming:
                    errors.append(f"{node.title}: required input '{port.name}' is not connected")
                if not port.multiple and len(incoming) > 1:
                    errors.append(f"{node.title}: input '{port.name}' accepts one connection")
        return errors

    def execute(self) -> dict[str, NodeResult]:
        errors = self.validate()
        if errors:
            raise GraphError("\n".join(errors))

        results: dict[str, NodeResult] = {}
        for node_id in self.topological_order():
            node = self.nodes[node_id]
            if not node.enabled:
                results[node_id] = NodeResult(None, summary="DISABLED", skipped=True)
                continue
            definition = self.definitions[node.type_name]
            inputs: dict[str, Any] = {}
            skip_reason = ""
            for port in definition.inputs:
                incoming = [edge for edge in self.edges if edge.target == node_id and edge.target_port == port.name]
                source_results = [results[edge.source] for edge in incoming]
                values = [result.value for result in source_results if not result.skipped]
                if port.required and not values:
                    skipped_sources = [edge.source for edge, result in zip(incoming, source_results) if result.skipped]
                    suffix = f" from {', '.join(skipped_sources)}" if skipped_sources else ""
                    skip_reason = f"SKIPPED missing {port.name}{suffix}"
                    break
                if port.multiple:
                    inputs[port.name] = values
                elif values:
                    inputs[port.name] = values[0]
                else:
                    inputs[port.name] = None

            if skip_reason:
                results[node_id] = NodeResult(None, summary=skip_reason, skipped=True)
                continue

            started = perf_counter()
            result = definition.processor(inputs, dict(node.params))
            result.elapsed_ms = (perf_counter() - started) * 1000
            results[node_id] = result
        self.results = results
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "revision": self.revision,
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.type_name,
                    "title": node.title,
                    "position": [node.x, node.y],
                    "enabled": node.enabled,
                    "params": {key: copy.deepcopy(value) for key, value in node.params.items() if key != "snapshot"},
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {"source": edge.source, "target": edge.target, "target_port": edge.target_port}
                for edge in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, definitions: dict[str, NodeDefinition], data: dict[str, Any]) -> "RecipeGraph":
        if data.get("format") != cls.FORMAT:
            raise GraphError(f"Unsupported recipe format: {data.get('format')}")

        graph = cls(definitions, int(data.get("revision", 0)))
        for item in data.get("nodes", []):
            node_id = str(item["id"])
            type_name = str(item["type"])
            title = str(item.get("title") or definitions[type_name].title)
            position = item.get("position", [0, 0])
            params = dict(item.get("params", {}))
            enabled = bool(item.get("enabled", True))
            graph.add_node(RecipeNode(node_id, type_name, title, float(position[0]), float(position[1]), params, enabled))

        for item in data.get("edges", []):
            graph.connect(str(item["source"]), str(item["target"]), str(item["target_port"]))
        graph.revision = int(data.get("revision", graph.revision))
        graph.results.clear()
        return graph
