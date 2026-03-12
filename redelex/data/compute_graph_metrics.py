import pandas as pd

from relbench.datasets import get_dataset_names

from redelex.data.graph_utils import (
    create_HeteroData_from_ctu_inspector,
    create_HeteroData_from_relbench_source,
    graph_density,
    graph_diameter,
    graph_max_degree,
    graph_node2edge_ratio,
    number_of_nodes_edges,
)


def create_dataset_HeteroData(
    dataset_name: str,
    process_hub: bool = False,
    process_bridge: bool = False,
    keep_hub_table: bool = False,
    keep_bridge_table: bool = False,
):
    """
    Creates HeteroData object for RelBench and CTU datasets based on the dataset name prefix.
    - RelBench - "rel-": from source-code extraction (no DB download).
    - CTU      - "ctu-": from DB inspector.
    """
    if dataset_name.startswith("rel-"):
        return create_HeteroData_from_relbench_source(
            dataset_name,
            process_hub=process_hub,
            process_bridge=process_bridge,
            keep_hub_table=keep_hub_table,
            keep_bridge_table=keep_bridge_table,
        )

    if dataset_name.startswith("ctu-"):
        return create_HeteroData_from_ctu_inspector(
            dataset_name,
            process_hub=process_hub,
            process_bridge=process_bridge,
            keep_hub_table=keep_hub_table,
            keep_bridge_table=keep_bridge_table,
        )

    raise ValueError(
        f"Unsupported dataset prefix for '{dataset_name}'. Expected 'rel-' or 'ctu-'."
    )


def compute_metrics_for_dataset(name: str, process_hub: bool, process_bridge: bool):
    """Compute all graph metrics for a single dataset."""
    print(f"Processing {name}...")

    data_default, data_processed, default_hubs, default_bridges = create_dataset_HeteroData(
        name,
        process_hub=process_hub,
        process_bridge=process_bridge,
    )

    # Basic counts
    default_nodes, default_edges = number_of_nodes_edges(data_default)
    processed_nodes, processed_edges = number_of_nodes_edges(data_processed)

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
    process_hub = False
    process_bridge = True
    output_file = "graph_metrics_test_bridges.csv"

    names = get_dataset_names()
    results = []

    for name in names:
        if not (name.startswith("rel-") or name.startswith("ctu-")):
            print(f"Skipping {name}: unsupported prefix (expected rel- or ctu-)")
            continue
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
