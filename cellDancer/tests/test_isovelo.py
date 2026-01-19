"""
Test script for IsoVelo: Isoform-level RNA velocity estimation.

This script demonstrates:
1. How to create sample input data in the correct format
2. How to run isoform-level velocity estimation
3. How to compute cell velocity from isoform-level results

Usage:
    python test_isovelo.py
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import celldancer as cd


def create_sample_isovelo_data(n_cells=100, n_genes=3, max_isoforms=3, seed=42):
    """
    Create sample isoform-level data for testing.
    
    The data format matches the expected IsoVelo input:
    - gene_name: Gene identifier
    - isoform_name: Isoform identifier (unique per gene)
    - unsplice: Unspliced counts (same for all isoforms of a gene in a cell)
    - splice: Spliced counts (different for each isoform)
    - cellID: Cell identifier
    - clusters: Cell type/cluster
    - embedding1: UMAP/embedding x-coordinate
    - embedding2: UMAP/embedding y-coordinate
    """
    np.random.seed(seed)
    
    rows = []
    
    # Create genes with different numbers of isoforms
    gene_isoform_counts = {
        'GeneA': 2,
        'GeneB': 3,
        'GeneC': 2
    }
    
    # Sample cluster assignments
    clusters = ['TypeA', 'TypeB', 'TypeC']
    cell_clusters = np.random.choice(clusters, n_cells)
    
    # Generate UMAP-like embedding (clustered)
    embedding1 = np.zeros(n_cells)
    embedding2 = np.zeros(n_cells)
    for i, cluster in enumerate(clusters):
        mask = cell_clusters == cluster
        n_in_cluster = mask.sum()
        embedding1[mask] = np.random.normal(i * 3, 0.5, n_in_cluster)
        embedding2[mask] = np.random.normal(i * 2, 0.5, n_in_cluster)
    
    for gene, n_isoforms in gene_isoform_counts.items():
        # Generate base unsplice for this gene (shared across isoforms)
        # Simulate temporal dynamics
        base_unsplice = np.abs(np.random.normal(5, 2, n_cells))
        
        for k in range(n_isoforms):
            isoform_name = f'{gene}_isoform_{k+1}'
            
            # Each isoform has different splice dynamics
            # Splice is related to unsplice but with isoform-specific factors
            beta_k = np.random.uniform(0.3, 0.8)
            gamma_k = np.random.uniform(0.2, 0.6)
            
            splice = base_unsplice * beta_k / gamma_k + np.random.normal(0, 0.5, n_cells)
            splice = np.maximum(splice, 0)  # Ensure non-negative
            
            for cell_idx in range(n_cells):
                rows.append({
                    'gene_name': gene,
                    'isoform_name': isoform_name,
                    'unsplice': base_unsplice[cell_idx],
                    'splice': splice[cell_idx],
                    'cellID': f'cell_{cell_idx}',
                    'clusters': cell_clusters[cell_idx],
                    'embedding1': embedding1[cell_idx],
                    'embedding2': embedding2[cell_idx]
                })
    
    df = pd.DataFrame(rows)
    return df


def test_data_format():
    """Test that the sample data has the correct format."""
    print("=" * 60)
    print("Testing data format...")
    print("=" * 60)
    
    df = create_sample_isovelo_data(n_cells=50, n_genes=2)
    
    # Check required columns
    required_cols = ['gene_name', 'isoform_name', 'unsplice', 'splice', 
                     'cellID', 'clusters', 'embedding1', 'embedding2']
    
    missing = set(required_cols) - set(df.columns)
    assert len(missing) == 0, f"Missing columns: {missing}"
    print(f"[PASS] All required columns present: {required_cols}")
    
    # Check that unsplice is same for all isoforms of a gene in a cell
    for gene in df.gene_name.unique():
        gene_data = df[df.gene_name == gene]
        for cell in gene_data.cellID.unique():
            cell_gene_data = gene_data[gene_data.cellID == cell]
            unsplice_values = cell_gene_data.unsplice.unique()
            assert len(unsplice_values) == 1, \
                f"Unsplice should be same for all isoforms of {gene} in {cell}"
    print("[PASS] Unsplice is consistent across isoforms for each gene-cell pair")
    
    print(f"\nSample data shape: {df.shape}")
    print(f"Genes: {df.gene_name.unique().tolist()}")
    print(f"Isoforms per gene:")
    for gene in df.gene_name.unique():
        isoforms = df[df.gene_name == gene].isoform_name.unique()
        print(f"  {gene}: {len(isoforms)} isoforms - {isoforms.tolist()}")
    
    print("\nFirst few rows:")
    print(df.head(10).to_string())
    
    return df


def test_isovelo_velocity(df=None):
    """Test the isoform-level velocity estimation."""
    print("\n" + "=" * 60)
    print("Testing IsoVelo velocity estimation...")
    print("=" * 60)
    
    if df is None:
        df = create_sample_isovelo_data(n_cells=100, n_genes=2)
    
    # Run IsoVelo (with minimal epochs for testing)
    print("\nRunning isovelo_velocity with test parameters...")
    print("(This is a minimal test - use more epochs for real data)")
    
    try:
        loss_df, isovelo_df = cd.isovelo_velocity(
            isovelo_u_s=df,
            gene_list=['GeneA', 'GeneB'],  # Test with 2 genes
            max_epoches=10,  # Minimal for testing
            check_val_every_n_epoch=5,
            patience=2,
            learning_rate=0.001,
            dt=0.5,
            n_neighbors=10,
            permutation_ratio=0.5,
            speed_up=False,  # No downsampling for small test
            norm_u_s=True,
            norm_cell_distribution=False,
            loss_func='cosine',
            n_jobs=1,  # Single job for testing
            save_path=os.path.join(os.path.dirname(__file__), 'test_output')
        )
        
        if isovelo_df is not None:
            print("\n[PASS] IsoVelo velocity estimation completed!")
            print(f"Result shape: {isovelo_df.shape}")
            print(f"Columns: {isovelo_df.columns.tolist()}")
            
            # Check output has isoform_name column
            assert 'isoform_name' in isovelo_df.columns, "isoform_name should be in output"
            print("[PASS] Output contains isoform_name column")
            
            # Check that alpha is gene-specific (same for all isoforms)
            for gene in isovelo_df.gene_name.unique():
                gene_data = isovelo_df[isovelo_df.gene_name == gene]
                for cell in gene_data.cellIndex.unique()[:5]:  # Check first 5 cells
                    cell_gene_data = gene_data[gene_data.cellIndex == cell]
                    if len(cell_gene_data) > 1:
                        alpha_values = cell_gene_data.alpha.unique()
                        # Alpha should be the same across isoforms
                        assert len(alpha_values) == 1, \
                            f"Alpha should be same for all isoforms of {gene} in cell {cell}"
            print("[PASS] Alpha is consistent across isoforms (gene-specific)")
            
            print("\nSample output rows:")
            print(isovelo_df.head(10).to_string())
            
            return isovelo_df
        else:
            print("[FAIL] IsoVelo returned None - check error messages above")
            return None
            
    except Exception as e:
        print(f"[FAIL] Error during IsoVelo estimation: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_compute_isovelo_velocity(isovelo_df=None):
    """Test the cell velocity computation from isoform results."""
    print("\n" + "=" * 60)
    print("Testing compute_isovelo_velocity...")
    print("=" * 60)
    
    if isovelo_df is None:
        print("No IsoVelo results provided - creating mock data...")
        # Create mock isovelo output for testing
        df = create_sample_isovelo_data(n_cells=50, n_genes=2)
        # Add mock prediction columns
        df['cellIndex'] = df.groupby(['gene_name', 'isoform_name']).cumcount()
        df['unsplice_predict'] = df['unsplice'] * 1.1
        df['splice_predict'] = df['splice'] * 1.05
        df['alpha'] = 1.0
        df['beta'] = 0.5
        df['gamma'] = 0.3
        df['loss'] = 0.1
        isovelo_df = df
    
    try:
        result_df = cd.compute_isovelo_velocity(
            isovelo_df,
            gene_list=None,
            speed_up=(10, 10),
            projection_neighbor_size=20,
            aggregate_method='sum'
        )
        
        print("[PASS] compute_isovelo_velocity completed!")
        
        # Check velocity columns
        assert 'velocity1' in result_df.columns, "velocity1 should be in output"
        assert 'velocity2' in result_df.columns, "velocity2 should be in output"
        print("[PASS] Output contains velocity1 and velocity2 columns")
        
        # Check that some cells have velocity
        has_velocity = result_df['velocity1'].notna().sum()
        print(f"Cells with computed velocity: {has_velocity}")
        
        print("\nSample output with velocity:")
        print(result_df[result_df['velocity1'].notna()].head(5).to_string())
        
        return result_df
        
    except Exception as e:
        print(f"[FAIL] Error during velocity computation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("IsoVelo Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Data format
    df = test_data_format()
    
    # Test 2: IsoVelo velocity estimation
    # Note: This will take a moment even with minimal epochs
    isovelo_df = test_isovelo_velocity(df)
    
    # Test 3: Compute cell velocity
    if isovelo_df is not None:
        test_compute_isovelo_velocity(isovelo_df)
    else:
        # Test with mock data
        test_compute_isovelo_velocity(None)
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
