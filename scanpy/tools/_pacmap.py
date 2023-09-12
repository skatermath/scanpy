from typing import Optional 
from anndata import AnnData

import pacmap as pacmap
from pacmap import PaCMAP

from .. import logging as logg


def pacmap(
    adata: AnnData,
    init_data: str = 'X_scVI',
    copy: bool = False,

    n_components=2,
    n_neighbors=10,
    MN_ratio=0.5,
    FP_ratio=2.0,
    pair_neighbors=None,
    pair_MN=None,
    pair_FP=None,
    distance="euclidean",
    lr=1.0,
    num_iters=450,
    verbose=False,
    apply_pca=True,
    intermediate=False,
    intermediate_snapshots=[
        0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450],
    random_state=None,
    save_tree=False
                 

) -> Optional[AnnData]:
    """Pairwise Controlled Manifold Approximation.\

    Maps high-dimensional dataset to a low-dimensional embedding.
    This class inherits the sklearn BaseEstimator, and we tried our best to
    follow the sklearn api. For details of this method, please refer to our publication:
    https://www.jmlr.org/papers/volume22/20-1061/20-1061.pdf

    Parameters
    ---------
    n_components: int, default=2
        Dimensions of the embedded space. We recommend to use 2 or 3.

    n_neighbors: int, default=10
        Number of neighbors considered for nearest neighbor pairs for local structure preservation.

    MN_ratio: float, default=0.5
        Ratio of mid near pairs to nearest neighbor pairs (e.g. n_neighbors=10, MN_ratio=0.5 --> 5 Mid near pairs)
        Mid near pairs are used for global structure preservation.

    FP_ratio: float, default=2.0
        Ratio of further pairs to nearest neighbor pairs (e.g. n_neighbors=10, FP_ratio=2 --> 20 Further pairs)
        Further pairs are used for both local and global structure preservation.

    pair_neighbors: numpy.ndarray, optional
        Nearest neighbor pairs constructed from a previous run or from outside functions.

    pair_MN: numpy.ndarray, optional
        Mid near pairs constructed from a previous run or from outside functions.

    pair_FP: numpy.ndarray, optional
        Further pairs constructed from a previous run or from outside functions.

    distance: string, default="euclidean"
        Distance metric used for high-dimensional space. Allowed metrics include euclidean, manhattan, angular, hamming.

    lr: float, default=1.0
        Learning rate of the Adam optimizer for embedding.

    num_iters: int, default=450
        Number of iterations for the optimization of embedding. 
        Due to the stage-based nature, we suggest this parameter to be greater than 250 for all three stages to be utilized.

    verbose: bool, default=False
        Whether to print additional information during initialization and fitting.

    apply_pca: bool, default=True
        Whether to apply PCA on the data before pair construction.

    intermediate: bool, default=False
        Whether to return intermediate state of the embedding during optimization.
        If True, returns a series of embedding during different stages of optimization.

    intermediate_snapshots: list[int], optional
        The index of step where an intermediate snapshot of the embedding is taken.
        If intermediate sets to True, the default value will be [0, 10, 30, 60, 100, 120, 140, 170, 200, 250, 300, 350, 450]

    random_state: int, optional
        Random state for the pacmap instance.
        Setting random state is useful for repeatability.

    save_tree: bool, default=False
        Whether to save the annoy index tree after finding the nearest neighbor pairs.
        Default to False for memory saving. Setting this option to True can make `transform()` method faster.
    """
    
    adata = adata.copy() if copy else adata

    start = logg.info('computing PaCMAP')


    embedding_data = adata.obsm[init_data]

    map = PaCMAP(
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

    adata.obsm['X_pacmap'] = X_pacmap  # annotate samples with UMAP coordinates
    logg.info(
        '    finished',
        time=start,
        deep=('added\n' "    'X_pacmap', pacmap coordinates (adata.obsm)"),
    )

    return adata if copy else None