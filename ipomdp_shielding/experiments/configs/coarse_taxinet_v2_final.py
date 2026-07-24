"""Final coarseness experiment configuration for TaxiNetV2."""

from .base_config import CoarseExperimentConfig
from ...CaseStudies.TaxiNetV2 import build_taxinet_v2_ipomdp
from ...Propagators import LikelihoodSamplingStrategy, PruningStrategy


config = CoarseExperimentConfig(
    case_study_name="taxinet_v2",
    build_ipomdp_fn=build_taxinet_v2_ipomdp,
    seed=42,
    num_trajectories=100,
    trajectory_length=20,
    initial_state=(0, 0),
    initial_action=0,
    sampler_budget=500,
    sampler_k=100,
    sampler_likelihood_strategy=LikelihoodSamplingStrategy.HYBRID,
    sampler_pruning_strategy=PruningStrategy.FARTHEST_POINT,
    results_path="results/taxinet_v2/coarse_taxinet_v2_results.json",
    ipomdp_kwargs={
        "confidence_method": "Clopper_Pearson",
        "alpha": 0.05,
        "confidence_level": "0.95",
        "error": 0.1,
        "smoothing": True,
    },
)
