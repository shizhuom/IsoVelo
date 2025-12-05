import argparse
import anndata

parser = argparse.ArgumentParser(description='Isovelo arguments')
parser.add_argument('--adata',type=str,help="Path to the adata file")

def main():
    global args
    args = parser.parse_args()

    if not args.adata:
        parser.error("--adata is required")

    adata = anndata.read_h5ad(args.adata)
    print(adata)

    U = adata.layers['unspliced']
    S = adata.layers['spliced']
    P = adata.uns['proportions_observed']
    I = 
   




