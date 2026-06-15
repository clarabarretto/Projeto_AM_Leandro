"""Placeholder do modelo principal do grupo.

Cada grupo deve substituir `build_group_model` pelo wrapper sklearn-compativel
do modelo atribuido. Exemplos comentados para os 10 modelos abaixo.

Padrao esperado: a funcao retorna um estimador com .fit(X, y) e .predict_proba(X).
Se o modelo nao tem API sklearn, envolva em sklearn.base.BaseEstimator.
"""

from __future__ import annotations


def build_group_model(seed: int = 42):
    # 2) TabICL v2
    from tabicl import TabICLClassifier
    import os
    
    # Create the cache directory for disk offloading if it doesn't exist
    os.makedirs("./cache/tabicl_offload", exist_ok=True)
    
    return TabICLClassifier(
        random_state=seed, 
        device="cuda", 
        batch_size=4, 
        offload_mode="disk", 
        disk_offload_dir="./cache/tabicl_offload", 
        kv_cache="repr"
    )
