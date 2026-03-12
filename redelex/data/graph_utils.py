import ast
import inspect
import itertools
import re
from collections import defaultdict

from torch_geometric.data import HeteroData

from relbench.datasets import get_dataset

from redelex.db import DBInspector
from redelex.db.utils import get_db_connection


def graph_max_degree(data: HeteroData):
	"""Calculate maximum and average degree in the graph."""
	degrees = defaultdict(int)
	for src, _, _ in data.edge_types:
		degrees[src] += 1

	max_degree = max(degrees.values()) if degrees else 0
	avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
	return max_degree, avg_degree


def graph_density(data: HeteroData):
	"""Calculate graph density based on node/edge types."""
	num_nodes = len(data.node_types)
	num_edges = int(len(data.edge_types) / 2)

	if num_nodes <= 1:
		return 0
	return num_edges / (num_nodes * (num_nodes - 1))


def graph_diameter(data: HeteroData):
	"""Calculate graph diameter and average pairwise reachable distance."""

	def bfs(adj, start):
		"""Breadth-first search to compute distances from start node."""
		dist = {start: 0}
		queue = [start]
		head = 0

		while head < len(queue):
			u = queue[head]
			head += 1

			for v in adj[u]:
				if v not in dist:
					dist[v] = dist[u] + 1
					queue.append(v)

		return dist

	adj = defaultdict(list)
	for src, _, dst in data.edge_types:
		adj[src].append(dst)

	diameter = 0
	all_distances = []
	for node in data.node_types:
		dist = bfs(adj, node)
		if dist.values():
			diameter = max(diameter, max(dist.values()))
			all_distances.extend(dist.values())

	avg_diameter = sum(all_distances) / len(all_distances) if all_distances else 0
	return diameter, avg_diameter


def graph_node2edge_ratio(data: HeteroData):
	"""Calculate ratio of node types to edge types (forward edges only)."""
	num_nodes = len(data.node_types)
	num_edges = int(len(data.edge_types) / 2)

	if num_edges == 0:
		return 0
	return num_nodes / num_edges


def number_of_nodes_edges(data: HeteroData):
	"""Count node types and edge types (forward edges only)."""
	num_nodes = len(data.node_types)
	num_edges = int(len(data.edge_types) / 2)
	return num_nodes, num_edges


def number_of_hubs_bridges_from_fkey_maps(fkey_maps: dict[str, dict[str, str]]):
	"""Count hub (3+ FK) and bridge (2 FK) tables from FK maps."""
	num_hubs = 0
	num_bridges = 0
	for fkey_map in fkey_maps.values():
		if len(fkey_map) >= 3:
			num_hubs += 1
		if len(fkey_map) == 2:
			num_bridges += 1
	return num_hubs, num_bridges


def number_of_hubs_bridges_from_inspector(inspector: DBInspector):
	"""Count hub and bridge tables using DB inspector schema."""
	fkey_maps = {}
	for table in inspector.get_tables():
		fkeys = inspector.get_foreign_keys(table)
		fkey_maps[table] = {fk.src_columns[0]: fk.ref_table for fk in fkeys}
	return number_of_hubs_bridges_from_fkey_maps(fkey_maps)


def create_HeteroData_from_inspector(inspector: DBInspector) -> HeteroData:
	"""Create default HeteroData from database schema via DBInspector."""
	data = HeteroData()
	for table in inspector.get_tables():
		fkeys = inspector.get_foreign_keys(table)
		data[table]
		for fk in fkeys:
			edge_type = (table, f"f2p_{fk.src_columns[0]}", fk.ref_table)
			rev_edge_type = (fk.ref_table, f"rev_f2p_{fk.src_columns[0]}", table)
			data[edge_type]
			data[rev_edge_type]
	return data


def create_HeteroData_from_inspector_with_hub_bridge(
	inspector: DBInspector,
	process_hub: bool = False,
	process_bridge: bool = False,
	keep_hub_table: bool = False,
	keep_bridge_table: bool = False,
) -> HeteroData:
	"""Create HeteroData via DBInspector with optional hub/bridge processing."""
	data = HeteroData()
	cannot_delete = {table: False for table in inspector.get_tables()}

	for table in inspector.get_tables():
		for fk in inspector.get_foreign_keys(table):
			cannot_delete[fk.ref_table] = True

	for table in inspector.get_tables():
		fkeys = inspector.get_foreign_keys(table)
		fkey_dict = {fk.src_columns[0]: fk.ref_table for fk in fkeys}

		if process_hub and len(fkey_dict) >= 3:
			for (fk1, ref1), (fk2, ref2) in itertools.combinations(fkey_dict.items(), 2):
				relation_label = f"p2p_{table}_{fk1}_{fk2}"
				rev_relation_label = f"rev_p2p_{table}_{fk1}_{fk2}"
				data[(ref1, relation_label, ref2)]
				data[(ref2, rev_relation_label, ref1)]

			if keep_hub_table or cannot_delete[table]:
				data[table]
				for fk in fkeys:
					edge_type = (table, f"f2p_{fk.src_columns[0]}", fk.ref_table)
					rev_edge_type = (fk.ref_table, f"rev_f2p_{fk.src_columns[0]}", table)
					data[edge_type]
					data[rev_edge_type]

		elif process_bridge and len(fkey_dict) == 2:
			(fk1, ref1), (fk2, ref2) = list(fkey_dict.items())
			relation_label = f"p2p_{table}_{fk1}_{fk2}"
			rev_relation_label = f"rev_p2p_{table}_{fk1}_{fk2}"
			data[(ref1, relation_label, ref2)]
			data[(ref2, rev_relation_label, ref1)]

			if keep_bridge_table or cannot_delete[table]:
				data[table]
				for fk in fkeys:
					edge_type = (table, f"f2p_{fk.src_columns[0]}", fk.ref_table)
					rev_edge_type = (fk.ref_table, f"rev_f2p_{fk.src_columns[0]}", table)
					data[edge_type]
					data[rev_edge_type]

		else:
			data[table]
			for fk in fkeys:
				edge_type = (table, f"f2p_{fk.src_columns[0]}", fk.ref_table)
				rev_edge_type = (fk.ref_table, f"rev_f2p_{fk.src_columns[0]}", table)
				data[edge_type]
				data[rev_edge_type]

	return data


def extract_table_blocks_from_source(dataset_name: str) -> dict[str, str]:
	"""Extract ``Table(...)`` source code blocks from a RelBench dataset class."""
	dataset = get_dataset(dataset_name, download=False)
	source_text = inspect.getsource(dataset.__class__)
	tree = ast.parse(source_text)

	parent = {}
	for node in ast.walk(tree):
		for child in ast.iter_child_nodes(node):
			parent[child] = node

	def is_table_call(node: ast.AST) -> bool:
		if not isinstance(node, ast.Call):
			return False
		func = node.func
		return (
			isinstance(func, ast.Name) and func.id == "Table"
		) or (
			isinstance(func, ast.Attribute) and func.attr == "Table"
		)

	def table_name_from_parent(node: ast.AST):
		p = parent.get(node)

		if isinstance(p, ast.Dict):
			for key_node, value_node in zip(p.keys, p.values):
				if value_node is node and isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
					return key_node.value

		if isinstance(p, ast.Assign):
			for target in p.targets:
				if not isinstance(target, ast.Subscript):
					continue
				key = target.slice
				if isinstance(key, ast.Constant) and isinstance(key.value, str):
					return key.value

		return None

	tables = {}
	counter = 0
	for node in ast.walk(tree):
		if not is_table_call(node):
			continue

		counter += 1
		table_name = table_name_from_parent(node)
		if table_name is None:
			table_name = f"table_{counter:03d}"

		block = ast.get_source_segment(source_text, node) or ""
		if table_name in tables:
			table_name = f"{table_name}_{counter:03d}"

		tables[table_name] = block

	return tables


def _parse_table_call(table_block: str):
	try:
		node = ast.parse(table_block, mode="eval").body
	except SyntaxError:
		return None
	return node if isinstance(node, ast.Call) else None


def _parse_fkey_map_from_table_block(table_block: str) -> dict[str, str]:
	call = _parse_table_call(table_block)
	if call is None:
		return {}

	for kw in call.keywords:
		if kw.arg != "fkey_col_to_pkey_table":
			continue
		try:
			value = ast.literal_eval(kw.value)
		except Exception:
			return {}
		if isinstance(value, dict):
			return {str(k): str(v) for k, v in value.items()}
		return {}

	return {}


def _extract_name_like(expr: ast.AST):
	if isinstance(expr, ast.Name):
		return expr.id
	if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
		return expr.value
	if isinstance(expr, ast.Subscript):
		index = expr.slice
		if isinstance(index, ast.Constant) and isinstance(index.value, str):
			return index.value
		if isinstance(index, ast.Name):
			return index.id
	if isinstance(expr, ast.Attribute):
		return expr.attr
	return None


def _parse_node_name_from_df_arg(table_block: str):
	call = _parse_table_call(table_block)
	if call is None:
		return None

	for kw in call.keywords:
		if kw.arg != "df":
			continue

		df_expr = kw.value
		if isinstance(df_expr, ast.Call):
			func = df_expr.func
			is_dataframe_ctor = (
				(isinstance(func, ast.Attribute) and func.attr == "DataFrame")
				or (isinstance(func, ast.Name) and func.id == "DataFrame")
			)
			if is_dataframe_ctor and df_expr.args:
				extracted = _extract_name_like(df_expr.args[0])
				if extracted:
					return extracted
				if hasattr(ast, "unparse"):
					return ast.unparse(df_expr.args[0])
				return None

		extracted = _extract_name_like(df_expr)
		if extracted:
			return extracted

		if hasattr(ast, "unparse"):
			return ast.unparse(df_expr)

		return None

	return None


def _is_generated_table_key(table_key: str):
	return re.fullmatch(r"table_\d{3}(?:_\d{3})?", table_key) is not None


def _build_tables_meta(tables_dict: dict[str, str]):
	meta = {}
	used_node_names = set()

	for table_key, table_block in tables_dict.items():
		parsed_name = _parse_node_name_from_df_arg(table_block)

		if not _is_generated_table_key(table_key):
			node_name = table_key
		else:
			node_name = parsed_name or table_key

		if node_name in used_node_names:
			node_name = f"{node_name}__{table_key}"
		used_node_names.add(node_name)

		meta[table_key] = {
			"node_name": node_name,
			"parsed_name": parsed_name,
			"fkey_map": _parse_fkey_map_from_table_block(table_block),
		}

	key_to_node = {k: v["node_name"] for k, v in meta.items()}
	ref_to_node = dict(key_to_node)
	for _, info in meta.items():
		parsed_name = info["parsed_name"]
		if parsed_name:
			ref_to_node[parsed_name] = info["node_name"]

	return meta, ref_to_node


def _add_fkey_edges(data: HeteroData, src_node: str, fkey_map: dict[str, str], ref_to_node: dict[str, str]):
	for fk_col, ref_table_key in fkey_map.items():
		dst_node = ref_to_node.get(ref_table_key)
		if dst_node is None:
			continue

		edge_type = (src_node, f"f2p_{fk_col}", dst_node)
		rev_edge_type = (dst_node, f"rev_f2p_{fk_col}", src_node)
		data[edge_type]
		data[rev_edge_type]


def create_HeteroData_from_tables_dict(tables_dict: dict[str, str]) -> HeteroData:
	"""Create default HeteroData from extracted ``Table(...)`` code blocks."""
	data = HeteroData()
	meta, ref_to_node = _build_tables_meta(tables_dict)

	for info in meta.values():
		data[info["node_name"]]

	for _, info in meta.items():
		src_node = info["node_name"]
		_add_fkey_edges(data, src_node, info["fkey_map"], ref_to_node)

	return data


def create_HeteroData_from_tables_dict_with_hub_bridge(
	tables_dict: dict[str, str],
	process_hub: bool = False,
	process_bridge: bool = False,
	keep_hub_table: bool = False,
	keep_bridge_table: bool = False,
) -> HeteroData:
	"""Create HeteroData from source-extracted tables with optional hub/bridge logic."""
	data = HeteroData()
	meta, ref_to_node = _build_tables_meta(tables_dict)

	cannot_delete = {info["node_name"]: False for info in meta.values()}
	for info in meta.values():
		for ref_table_key in info["fkey_map"].values():
			ref_node = ref_to_node.get(ref_table_key)
			if ref_node is not None:
				cannot_delete[ref_node] = True

	for _, info in meta.items():
		src_node = info["node_name"]
		fkey_map = info["fkey_map"]

		resolved_items = []
		for fk_name, ref_key in fkey_map.items():
			ref_node = ref_to_node.get(ref_key)
			if ref_node is None:
				continue
			resolved_items.append((fk_name, ref_node))

		if process_hub and len(resolved_items) >= 3:
			for (fk1, n1), (fk2, n2) in itertools.combinations(resolved_items, 2):
				data[n1]
				data[n2]
				data[(n1, f"p2p_{src_node}_{fk1}_{fk2}", n2)]
				data[(n2, f"rev_p2p_{src_node}_{fk1}_{fk2}", n1)]

			if keep_hub_table or cannot_delete.get(src_node, False):
				data[src_node]
				_add_fkey_edges(data, src_node, fkey_map, ref_to_node)

		elif process_bridge and len(resolved_items) == 2:
			(fk1, n1), (fk2, n2) = resolved_items
			data[n1]
			data[n2]
			data[(n1, f"p2p_{src_node}_{fk1}_{fk2}", n2)]
			data[(n2, f"rev_p2p_{src_node}_{fk1}_{fk2}", n1)]

			if keep_bridge_table or cannot_delete.get(src_node, False):
				data[src_node]
				_add_fkey_edges(data, src_node, fkey_map, ref_to_node)

		else:
			data[src_node]
			_add_fkey_edges(data, src_node, fkey_map, ref_to_node)

	return data


def create_HeteroData_from_relbench_source(
	dataset_name: str,
	process_hub: bool = False,
	process_bridge: bool = False,
	keep_hub_table: bool = False,
	keep_bridge_table: bool = False,
):
	"""Create default/processed HeteroData and hub/bridge counts from source code only."""
	tables_dict = extract_table_blocks_from_source(dataset_name)

	default_data = create_HeteroData_from_tables_dict(tables_dict)
	processed_data = create_HeteroData_from_tables_dict_with_hub_bridge(
		tables_dict,
		process_hub=process_hub,
		process_bridge=process_bridge,
		keep_hub_table=keep_hub_table,
		keep_bridge_table=keep_bridge_table,
	)

	fkey_maps = {
		table_name: _parse_fkey_map_from_table_block(table_block)
		for table_name, table_block in tables_dict.items()
	}
	num_hubs, num_bridges = number_of_hubs_bridges_from_fkey_maps(fkey_maps)

	return default_data, processed_data, num_hubs, num_bridges


def create_HeteroData_from_ctu_inspector(
	dataset_name: str,
	process_hub: bool = False,
	process_bridge: bool = False,
	keep_hub_table: bool = False,
	keep_bridge_table: bool = False,
):
	"""Create default/processed HeteroData and hub/bridge counts for CTU datasets."""
	dataset = get_dataset(dataset_name)
	inspector = DBInspector(get_db_connection(dataset.remote_url))
	try:
		default_data = create_HeteroData_from_inspector(inspector)
		processed_data = create_HeteroData_from_inspector_with_hub_bridge(
			inspector,
			process_hub=process_hub,
			process_bridge=process_bridge,
			keep_hub_table=keep_hub_table,
			keep_bridge_table=keep_bridge_table,
		)
		num_hubs, num_bridges = number_of_hubs_bridges_from_inspector(inspector)
	finally:
		inspector.connection.close()

	return default_data, processed_data, num_hubs, num_bridges


