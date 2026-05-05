"""fast_fact: 8-K event-driven stock direction prediction.

Pipelines for collecting SEC 8-K filings, computing horizon-window abnormal
returns vs SPY, and training/comparing several text classifiers (LoRA SFT,
frozen-encoder RAG, and DPO-style preference tuning).
"""

from fast_fact.config import DataConfig, ModelConfig, SECConfig

__all__ = ["DataConfig", "ModelConfig", "SECConfig", "__version__"]
__version__ = "0.1.0"
