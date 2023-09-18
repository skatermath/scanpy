from typing import Optional, Union
from anndata import AnnData
from .. import logging as logg
import numpy as np

def pacmap(
    adata: AnnData,
    init_data: str = 'X_scVI',
    copy: bool = False,
    n_components: int = 2,
    n_neighbors: int = 10,
    MN_ratio: float = 0.5,
    FP_ratio: float = 2.0,
    pair_neighbors: Optional[np.ndarray] = None,
    pair_MN: Optional[np.ndarray] = None,
    pair_FP: Optional[np.ndarray] = None,
    distance: str = "euclidean",
    lr: float = 1.0,
    num_iters: int = 450,
    verbose: bool = False,
    apply_pca: bool = True,
    intermediate: bool = False,
    intermediate_snapshots: Optional[list[int]] = None,
    random_state: Optional[int] = None,
    save_tree: bool = False
) -> Optional[AnnData]:
    """Pairwise Controlled Manifold Approximation (PACMAP).

    Maps high-dimensional dataset to a low-dimensional embedding using PACMAP.
    This class inherits the sklearn BaseEstimator, and we tried our best to
    follow the sklearn api. For details of this method, please refer to our publication:
    https://www.jmlr.org/papers/volume22/20-1061/20-1061.pdf

    Parameters:
    - adata: AnnData
        Annotated data matrix.
    - init_data: str, optional (default: 'X_scVI')
        Key in adata.obsm to use as the initial data for PACMAP.
    - copy: bool, optional (default: False)
        Whether to return a copy of adata with the PACMAP embedding.
    - n_components: int, optional (default: 2)
        Dimensions of the embedded space (2 or 3 recommended).
    - n_neighbors: int, optional (default: 10)
        Number of neighbors considered for nearest neighbor pairs.
    - MN_ratio: float, optional (default: 0.5)
        Ratio of mid-near pairs to nearest neighbor pairs.
    - FP_ratio: float, optional (default: 2.0)
        Ratio of further pairs to nearest neighbor pairs.
    - pair_neighbors: np.ndarray, optional (default: None)
        Nearest neighbor pairs constructed from a previous run.
    - pair_MN: np.ndarray, optional (default: None)
        Mid-near pairs constructed from a previous run.
    - pair_FP: np.ndarray, optional (default: None)
        Further pairs constructed from a previous run.
    - distance: str, optional (default: "euclidean")
        Distance metric used for high-dimensional space. Allowed metrics include euclidean, manhattan, angular, hamming.
    - lr: float, optional (default: 1.0)
        Learning rate of the optimizer for embedding.
    - num_iters: int, optional (default: 450)
        Number of iterations for the optimization of embedding.
    - verbose: bool, optional (default: False)
        Whether to print additional information during initialization and fitting.
    - apply_pca: bool, optional (default: True)
        Whether to apply PCA on the data before pair construction.
    - intermediate: bool, optional (default: False)
        Whether to return intermediate snapshots of the embedding during optimization.
    - intermediate_snapshots: list[int], optional (default: None)
        Indices of steps where intermediate snapshots of the embedding are taken.
    - random_state: int, optional (default: None)
        Random state for reproducibility.
    - save_tree: bool, optional (default: False)
        Whether to save the index tree after finding nearest neighbor pairs.

    Returns:
    - adata: AnnData
        Annotated data matrix with PACMAP coordinates in adata.obsm['X_pacmap'].
    """

    # Error handling for invalid input data
    if init_data not in adata.obsm:
        raise ValueError(f"Invalid init_data: '{init_data}' is not found in adata.obsm.")

    # ... Add more input validation and error handling as needed ...

    adata = adata.copy() if copy else adata

    start = logg.info('computing PaCMAP')

    embedding_data = adata.obsm[init_data]

    try:
        import pacmap as pacmap
    except ImportError:
        raise ImportError(
            'You need to install the package `pacmap`: please run `pip install '
            'pacmap` in a terminal.'
    )


    try:
        map = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=MN_ratio,
            FP_ratio=FP_ratio,
            pair_neighbors=pair_neighbors,
            pair_MN=pair_MN,
            pair_FP=pair_FP,
            distance=distance,
            lr=lr,
            num_iters=num_iters,
            verbose=verbose,
            apply_pca=apply_pca,
            intermediate=intermediate,
            intermediate_snapshots=intermediate_snapshots,
            random_state=random_state,
            save_tree=save_tree
        )

        X_pacmap = map.fit_transform(embedding_data)
        adata.obsm['X_pacmap'] = X_pacmap

        logg.info(
            '    finished',
            time=start,
            deep=("added\n" "    'X_pacmap', pacmap coordinates (adata.obsm)"),
        )

    except Exception as e:
        logg.error(f'Error in PACMAP computation: {str(e)}')
        raise e

    return adata if copy else None
