"""Metrics infrastructure for template comparison evaluation.

Provides a modular system for computing and tracking metrics during
belief propagation runs.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

from ..Models import IPOMDP
from ..Propagators import LFPPropagator, BeliefPolytope, Template, compute_volume

# temporary
from .test_volume import print_polytope_info

# ============================================================
# Metrics Infrastructure
# ============================================================

@dataclass
class MetricValue:
    """A single metric value with metadata for plotting."""
    name: str
    value: float
    display_name: str = ""
    ylabel: str = ""
    use_log_scale: bool = False
    ylim: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()
        if not self.ylabel:
            self.ylabel = self.display_name


class MetricsCollector(ABC):
    """Abstract base class for metrics collection.

    Subclasses define which metrics to compute from an rt_shield object.
    """

    @abstractmethod
    def compute(self, rt_shield, step: int) -> Dict[str, MetricValue]:
        """Compute all metrics for the current step.

        Returns a dict mapping metric name to MetricValue.
        """
        pass

    @abstractmethod
    def metric_names(self) -> List[str]:
        """Return list of metric names this collector produces."""
        pass

    @abstractmethod
    def get_plot_config(self, metric_name: str) -> Dict:
        """Return plotting configuration for a metric.

        Returns dict with keys: display_name, ylabel, use_log_scale, ylim
        """
        pass

    def reset(self):
        """Reset internal state for a new run. Override if needed."""
        pass


@dataclass
class StepMetrics:
    """Container for metrics at a single step."""
    step: int
    values: Dict[str, MetricValue] = field(default_factory=dict)

    def __getattr__(self, name):
        if name in ('step', 'values') or name.startswith('_'):
            return super().__getattribute__(name)
        if name in self.values:
            return self.values[name].value
        raise AttributeError(f"No metric named '{name}'")

    def get(self, name: str, default: float = 0.0) -> float:
        """Get metric value by name with default."""
        if name in self.values:
            return self.values[name].value
        return default


# ============================================================
# Standalone compute functions (reusable by any MetricsCollector)
# ============================================================

def compute_template_spread(polytope: BeliefPolytope, template: Template) -> float:
    """Compute sum of spreads across all template directions.

    Parameters
    ----------
    polytope : BeliefPolytope
        The current belief polytope
    template : Template
        The template defining directions to measure

    Returns
    -------
    float
        Total spread (sum of upper - lower for each template direction)
    """
    total_spread = 0.0
    for k in range(template.K):
        v = template.V[k]
        try:
            max_val = polytope.maximize_linear(v)
            min_val = -polytope.maximize_linear(-v)
            total_spread += max_val - min_val
        except RuntimeError:
            total_spread += 1.0
    return total_spread


def compute_volume_proxy(polytope: BeliefPolytope, template: Template) -> float:
    """Compute polytope volume using vertex enumeration and ConvexHull.

    Projects the polytope onto (n-1) dimensions and computes the exact volume.
    Returns volume as a fraction of the full simplex (0 to 1 scale).

    Parameters
    ----------
    polytope : BeliefPolytope
        The current belief polytope
    template : Template
        The template (kept for API compatibility, not used in computation)

    Returns
    -------
    float
        Volume as a fraction of simplex volume (0 to 1)
    """
    _ = template  # Unused, kept for API compatibility
    print_polytope_info("current",polytope)
    return compute_volume(polytope)


def compute_safest_action_prob(polytope: BeliefPolytope, ipomdp: IPOMDP) -> float:
    """Compute the maximum minimum safe probability across all actions.

    For each action, computes the minimum probability of being in a state
    where that action is safe. Returns the maximum across all actions.

    Parameters
    ----------
    polytope : BeliefPolytope
        The current belief polytope
    ipomdp : IPOMDP
        The interval POMDP model

    Returns
    -------
    float
        Maximum (over actions) of minimum safe probability
    """
    states = list(ipomdp.states)
    actions = list(ipomdp.actions) if isinstance(ipomdp.actions, (list, set)) else list(ipomdp.actions)
    n = len(states)

    safest_prob = 0.0
    for action in actions:
        safe_indices = []
        for i, s in enumerate(states):
            next_states = ipomdp.T.get((s, action), {})
            if "FAIL" not in next_states or next_states.get("FAIL", 0) < 0.5:
                safe_indices.append(i)

        if not safe_indices:
            continue

        indicator = np.zeros(n)
        for i in safe_indices:
            indicator[i] = 1.0

        try:
            min_safe_prob = -polytope.maximize_linear(-indicator)
            min_safe_prob = max(0.0, min(1.0, min_safe_prob))
        except RuntimeError:
            min_safe_prob = 0.0

        safest_prob = max(safest_prob, min_safe_prob)

    return safest_prob


# ============================================================
# ApproximationMetrics_1 - Metrics collector using standalone functions
# ============================================================

class ApproximationMetrics_1(MetricsCollector):
    """Metrics collector for LFP belief approximation quality.

    Computes:
    - template_spread: Sum of (upper - lower) for each template direction
    - volume_proxy: Product of spreads (proxy for polytope volume)
    - safest_action_prob: Max over actions of min safe probability
    - normalized_spread: Spread relative to initial (tracks decay)
    """

    def __init__(self):
        self._initial_spread: Optional[float] = None
        self._metric_configs = {
            "template_spread": {
                "display_name": "Template Spread",
                "ylabel": "Template Spread",
                "use_log_scale": False,
                "ylim": None
            },
            "volume_proxy": {
                "display_name": "Volume Proxy",
                "ylabel": "Volume Proxy (log scale)",
                "use_log_scale": True,
                "ylim": None
            },
            "safest_action_prob": {
                "display_name": "Safest Action Probability",
                "ylabel": "Safest Action Probability",
                "use_log_scale": False,
                "ylim": (-0.05, 1.05)
            },
            "normalized_spread": {
                "display_name": "Normalized Spread",
                "ylabel": "Normalized Spread",
                "use_log_scale": False,
                "ylim": None
            }
        }

    def reset(self):
        """Reset internal state for a new run."""
        self._initial_spread = None

    def metric_names(self) -> List[str]:
        return list(self._metric_configs.keys())

    def get_plot_config(self, metric_name: str) -> Dict:
        return self._metric_configs.get(metric_name, {
            "display_name": metric_name,
            "ylabel": metric_name,
            "use_log_scale": False,
            "ylim": None
        })

    def compute(self, rt_shield, step: int) -> Dict[str, MetricValue]:
        """Compute all metrics from the current rt_shield state."""
        assert isinstance(rt_shield.ipomdp_belief, LFPPropagator)

        polytope = rt_shield.ipomdp_belief.belief
        template = rt_shield.ipomdp_belief.template
        ipomdp = rt_shield.ipomdp_belief.ipomdp

        # Compute raw values using standalone functions
        spread = compute_template_spread(polytope, template)
        volume = compute_volume_proxy(polytope, template)
        safest_prob = compute_safest_action_prob(polytope, ipomdp)

        # Track initial spread for normalization
        if self._initial_spread is None or self._initial_spread == 0:
            self._initial_spread = spread if spread > 0 else 1.0

        normalized = spread / self._initial_spread

        # Build metric values with plot configs
        metrics = {}
        for name in self.metric_names():
            config = self._metric_configs[name]
            if name == "template_spread":
                value = spread
            elif name == "volume_proxy":
                value = max(volume, 1e-20)  # Avoid log(0)
            elif name == "safest_action_prob":
                value = safest_prob
            elif name == "normalized_spread":
                value = normalized
            else:
                continue

            metrics[name] = MetricValue(
                name=name,
                value=value,
                display_name=config["display_name"],
                ylabel=config["ylabel"],
                use_log_scale=config["use_log_scale"],
                ylim=config["ylim"]
            )

        return metrics


# ============================================================
# GroundTruthComparisonMetrics - Per-action and per-state bounds
# ============================================================

@dataclass
class StepPredictions:
    """Detailed per-action and per-state predictions at a single step."""
    step: int
    action_min_probs: Dict[Any, float]  # action -> min P(safe states)
    state_bounds: Dict[int, Tuple[float, float]]  # state_idx -> (lower, upper)


class GroundTruthComparisonMetrics(MetricsCollector):
    """Collects per-action safety bounds and per-state probability bounds.

    For each step, computes:
    - Per-action: minimum probability of being in safe states (inv_shield[a])
    - Per-state: [lower, upper] probability bounds via LP on unit vectors

    Stores detailed predictions internally for later aggregation against
    Monte Carlo ground truth.
    """

    def __init__(self):
        self.step_predictions: List[StepPredictions] = []
        self._metric_configs = {
            "mean_action_min_prob": {
                "display_name": "Mean Action Min P(safe)",
                "ylabel": "Mean Min P(safe)",
                "use_log_scale": False,
                "ylim": (-0.05, 1.05)
            },
            "mean_state_bound_width": {
                "display_name": "Mean State Bound Width",
                "ylabel": "Mean Bound Width",
                "use_log_scale": False,
                "ylim": None
            }
        }

    def reset(self):
        """Reset internal state for a new run."""
        self.step_predictions = []

    def metric_names(self) -> List[str]:
        return list(self._metric_configs.keys())

    def get_plot_config(self, metric_name: str) -> Dict:
        return self._metric_configs.get(metric_name, {
            "display_name": metric_name,
            "ylabel": metric_name,
            "use_log_scale": False,
            "ylim": None
        })

    def compute(self, rt_shield, step: int) -> Dict[str, MetricValue]:
        """Compute per-action and per-state bounds from the current rt_shield state.

        Stores detailed predictions in self.step_predictions for later
        ground truth comparison.
        """
        assert isinstance(rt_shield.ipomdp_belief, LFPPropagator)

        polytope = rt_shield.ipomdp_belief.belief
        n = polytope.n

        # Per-action: min P(safe states) for each action
        action_min_probs = {}
        for action in rt_shield.actions:
            allowed_states = rt_shield.inv_shield[action]
            if allowed_states:
                try:
                    min_prob = polytope.minimum_allowed_prob(allowed_states)
                    min_prob = max(0.0, min(1.0, min_prob))
                except RuntimeError:
                    min_prob = 0.0
            else:
                min_prob = 0.0
            action_min_probs[action] = min_prob

        # Per-state: [lower, upper] probability bounds
        state_bounds = {}
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            try:
                upper = polytope.maximize_linear(e_i)
                upper = max(0.0, min(1.0, upper))
            except RuntimeError:
                upper = 1.0
            try:
                lower = -polytope.maximize_linear(-e_i)
                lower = max(0.0, min(1.0, lower))
            except RuntimeError:
                lower = 0.0
            state_bounds[i] = (lower, upper)

        # Store detailed predictions
        self.step_predictions.append(StepPredictions(
            step=step,
            action_min_probs=action_min_probs,
            state_bounds=state_bounds
        ))

        # Compute scalar summaries for MetricValue compatibility
        mean_action_prob = (
            np.mean(list(action_min_probs.values()))
            if action_min_probs else 0.0
        )
        bound_widths = [ub - lb for lb, ub in state_bounds.values()]
        mean_bound_width = np.mean(bound_widths) if bound_widths else 0.0

        metrics = {}
        metrics["mean_action_min_prob"] = MetricValue(
            name="mean_action_min_prob",
            value=float(mean_action_prob),
            display_name="Mean Action Min P(safe)",
            ylabel="Mean Min P(safe)",
            use_log_scale=False,
            ylim=(-0.05, 1.05)
        )
        metrics["mean_state_bound_width"] = MetricValue(
            name="mean_state_bound_width",
            value=float(mean_bound_width),
            display_name="Mean State Bound Width",
            ylabel="Mean Bound Width",
            use_log_scale=False,
            ylim=None
        )

        return metrics
