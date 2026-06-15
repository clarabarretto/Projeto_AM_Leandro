"""Modelos do projeto: baselines, AutoML e modelo do grupo.

Importacoes sao feitas diretamente dos submodulos para evitar puxar dependencias
pesadas (como autogluon) quando nao sao necessarias. Exemplo:

    from src.models.baselines import build_lightgbm
    from src.models.automl import build_autogluon  # so se autogluon esta instalado
    from src.models.group_model import build_group_model
"""
