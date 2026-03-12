import pandas as pd
import itertools
from collections import defaultdict
from torch_geometric.data import HeteroData

from relbench.datasets import get_dataset, get_dataset_names

from redelex.db import DBInspector
from redelex.db.utils import get_db_connection




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


def graph_max_degree(data: HeteroData):
    """Calculate maximum and average degree in the graph."""
    edge_types = data.edge_types
    degrees = defaultdict(int)

    for edge_key in edge_types:
        src, edg, dst = edge_key
        degrees[src] += 1

    max_degree = max(degrees.values()) if degrees else 0
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    return max_degree, avg_degree


def graph_density(data: HeteroData):
    """Calculate graph density."""
    num_nodes = len(data.node_types)
    num_edges = int(len(data.edge_types) / 2)  # Divide by 2 for reverse edges

    if num_nodes <= 1:
        return 0
    return num_edges / (num_nodes * (num_nodes - 1))


def graph_diameter(data: HeteroData):
    """Calculate graph diameter and average diameter."""
    edge_types = data.edge_types
    adj = defaultdict(list)

    for edge_key in edge_types:
        src, edg, dst = edge_key
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
    num_nodes = len(data.node_types)
    num_edges = int(len(data.edge_types) / 2)

    if num_edges == 0:
        return 0
    return num_nodes / num_edges


def create_HeteroData(inspector: DBInspector) -> HeteroData:
    """Create default HeteroData from database schema."""
    data = HeteroData()
    for t in inspector.get_tables():
        fkeys = inspector.get_foreign_keys(t)
        data[t]
        for fk in fkeys:
            edge_type = (t, f"f2p_{fk.src_columns[0]}", fk.ref_table)
            rev_edge_type = (fk.ref_table, f"rev_f2p_{fk.src_columns[0]}", t)
            data[edge_type]
            data[rev_edge_type]
    return data


def create_HeteroData_with_hub_bridge(
    inspector: DBInspector, 
    process_hub: bool = False, 
    process_bridge: bool = False,
    keep_hub_table: bool = False,
    keep_bridge_table: bool = False
) -> HeteroData:
    """Create HeteroData with optional hub/bridge processing."""
    
    data = HeteroData()
    cannot_delete = {}
    
    # Determine which tables cannot be deleted
    for table in inspector.get_tables():
        cannot_delete[table] = False
    
    for table in inspector.get_tables():
        fkeys = inspector.get_foreign_keys(table)
        for fk in fkeys:
            cannot_delete[fk.ref_table] = True
    
    # Create nodes and edges
    for table in inspector.get_tables():
        fkeys = inspector.get_foreign_keys(table)
        fkey_dict = {fk.src_columns[0]: fk.ref_table for fk in fkeys}
        
        # Process hub tables (3+ foreign keys)
        if process_hub and len(fkey_dict) >= 3:
            fkey_pairs = list(itertools.combinations(fkey_dict.items(), 2))
            for (fkey_name_1, ref_table_1), (fkey_name_2, ref_table_2) in fkey_pairs:
                relation_label = f"p2p_{table}_{fkey_name_1}_{fkey_name_2}"
                rev_relation_label = f"rev_p2p_{table}_{fkey_name_1}_{fkey_name_2}"
                edge_type_1 = (ref_table_1, relation_label, ref_table_2)
                edge_type_2 = (ref_table_2, rev_relation_label, ref_table_1)
                data[edge_type_1]
                data[edge_type_2]
            
            if keep_hub_table or cannot_delete[table]:
                data[table]
                for fk in fkeys:
                    edge_type = (table, f"f2p_{fk.src_columns[0]}", fk.ref_table)
                    rev_edge_type = (fk.ref_table, f"rev_f2p_{fk.src_columns[0]}", table)
                    data[edge_type]
                    data[rev_edge_type]
        
        # Process bridge tables (exactly 2 foreign keys)
        elif process_bridge and len(fkey_dict) == 2:
            fkeys_list = list(fkey_dict.items())
            fkey_name_1, ref_table_1 = fkeys_list[0]
            fkey_name_2, ref_table_2 = fkeys_list[1]
            
            relation_label = f"p2p_{table}_{fkey_name_1}_{fkey_name_2}"
            rev_relation_label = f"rev_p2p_{table}_{fkey_name_1}_{fkey_name_2}"
            edge_type_1 = (ref_table_1, relation_label, ref_table_2)
            edge_type_2 = (ref_table_2, rev_relation_label, ref_table_1)
            data[edge_type_1]
            data[edge_type_2]
            
            if keep_bridge_table or cannot_delete[table]:
                data[table]
                for fk in fkeys:
                    edge_type = (table, f"f2p_{fk.src_columns[0]}", fk.ref_table)
                    rev_edge_type = (fk.ref_table, f"rev_f2p_{fk.src_columns[0]}", table)
                    data[edge_type]
                    data[rev_edge_type]
        
        # Default: create standard primary-foreign key edges
        else:
            data[table]
            for fk in fkeys:
                edge_type = (table, f"f2p_{fk.src_columns[0]}", fk.ref_table)
                rev_edge_type = (fk.ref_table, f"rev_f2p_{fk.src_columns[0]}", table)
                data[edge_type]
                data[rev_edge_type]
    
    return data


def number_of_nodes_edges(data: HeteroData):
    """Count nodes and edges in HeteroData."""
    num_nodes = len(data.node_types)
    num_edges = int(len(data.edge_types) / 2)  # Divide by 2 for reverse edges
    return num_nodes, num_edges


def number_of_hubs_bridges(inspector: DBInspector):
    """Count hub and bridge tables in database schema."""
    num_hubs = 0
    num_bridges = 0
    for t in inspector.get_tables():
        fkeys = inspector.get_foreign_keys(t)
        fkey_dict = {fk.src_columns[0]: fk.ref_table for fk in fkeys}
        
        if len(fkey_dict) >= 3:
            num_hubs += 1
        if len(fkey_dict) == 2:
            num_bridges += 1
    
    return num_hubs, num_bridges


def compute_metrics_for_dataset(name: str, process_hub: bool, process_bridge: bool):
    """Compute all graph metrics for a single dataset."""
    print(f"Processing {name}...")
    
    dataset = get_dataset(name)
    inspector = DBInspector(get_db_connection(dataset.remote_url))

    # Create graphs
    data_default = create_HeteroData(inspector)
    data_processed = create_HeteroData_with_hub_bridge(
        inspector, 
        process_hub=process_hub, 
        process_bridge=process_bridge
    )

    # Basic counts
    default_nodes, default_edges = number_of_nodes_edges(data_default)
    processed_nodes, processed_edges = number_of_nodes_edges(data_processed)
    default_hubs, default_bridges = number_of_hubs_bridges(inspector)

    # Graph metrics - default
    default_diameter, default_avg_diameter = graph_diameter(data_default)
    default_degree, default_avg_degree = graph_max_degree(data_default)
    default_density = graph_density(data_default)
    default_node2edge_ratio = graph_node2edge_ratio(data_default)

    # Graph metrics - processed
    processed_diameter, processed_avg_diameter = graph_diameter(data_processed)
    processed_degree, processed_avg_degree = graph_max_degree(data_processed)
    processed_density = graph_density(data_processed)
    processed_node2edge_ratio = graph_node2edge_ratio(data_processed)

    # Calculate deltas
    delta_nodes = processed_nodes - default_nodes
    delta_edges = processed_edges - default_edges
    delta_diameter = processed_diameter - default_diameter
    delta_avg_diameter = processed_avg_diameter - default_avg_diameter
    delta_degree = processed_degree - default_degree
    delta_avg_degree = processed_avg_degree - default_avg_degree
    delta_density = processed_density - default_density
    delta_node2edge_ratio = processed_node2edge_ratio - default_node2edge_ratio

    # Hub/bridge ratios (avoid division by zero)
    # Count only relevant structures number based on process_hub and process_bridge
    if process_hub and process_bridge:
        relevant_count = default_hubs + default_bridges
    elif process_hub and not process_bridge:
        relevant_count = default_hubs
    elif process_bridge and not process_hub:
        relevant_count = default_bridges
    else:
        relevant_count = 0
    
    if default_nodes - processed_nodes != 0:
        processed_hubs_bridges_ratio = relevant_count / (default_nodes - processed_nodes)
    else:
        processed_hubs_bridges_ratio = None
    
    if default_nodes != 0:
        normalized_hubs_bridges_ratio = relevant_count / default_nodes
    else:
        normalized_hubs_bridges_ratio = None

    inspector.connection.close()

    return {
        'dataset': name,
        'num_hubs': default_hubs,
        'num_bridges': default_bridges,
        'default_nodes': default_nodes,
        'default_edges': default_edges,
        'processed_nodes': processed_nodes,
        'processed_edges': processed_edges,
        'delta_nodes': delta_nodes,
        'delta_edges': delta_edges,
        'default_diameter': default_diameter,
        'default_avg_diameter': round(default_avg_diameter, 2),
        'processed_diameter': processed_diameter,
        'processed_avg_diameter': round(processed_avg_diameter, 2),
        'delta_diameter': delta_diameter,
        'delta_avg_diameter': round(delta_avg_diameter, 2),
        'default_max_degree': default_degree,
        'default_avg_degree': round(default_avg_degree, 2),
        'processed_max_degree': processed_degree,
        'processed_avg_degree': round(processed_avg_degree, 2),
        'delta_max_degree': delta_degree,
        'delta_avg_degree': round(delta_avg_degree, 2),
        'default_density': round(default_density, 2),
        'processed_density': round(processed_density, 2),
        'delta_density': round(delta_density, 2),
        'default_node2edge_ratio': round(default_node2edge_ratio, 2),
        'processed_node2edge_ratio': round(processed_node2edge_ratio, 2),
        'delta_node2edge_ratio': round(delta_node2edge_ratio, 2),
        'processed_hubs_bridges_ratio': round(processed_hubs_bridges_ratio, 2) if processed_hubs_bridges_ratio is not None else None,
        'normalized_hubs_bridges_ratio': round(normalized_hubs_bridges_ratio, 2) if normalized_hubs_bridges_ratio is not None else None,
    }


def main():
    """Main function to compute metrics for all datasets and save to CSV."""
    # Configuration
    process_hub = True
    process_bridge = False
    output_file = "graph_metrics_info_hubs.csv"
    dataset_start_index = 7  # Start from index 7 in dataset list

    names = get_dataset_names()[dataset_start_index:]
    results = []

    for name in names:
        try:
            metrics = compute_metrics_for_dataset(name, process_hub, process_bridge)
            results.append(metrics)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print()
    print("=" * 80)
    print(f"Results saved to {output_file}")
    print(f"Total datasets processed: {len(results)}")
    print("=" * 80)
    


if __name__ == "__main__":
    main()
