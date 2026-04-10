"""Coarseness evaluator comparing LFP over-approximation with sampled under-approximation.

Measures the gap between the LFP over-approximation and a sampling-based
under-approximation of the true reachable belief set.

Mathematical basis:
    P_sampled (under) ⊆ P_true ⊆ P_lfp (over)
    min_allowed_lfp ≤ min_allowed_true ≤ min_allowed_sampled
    max_disallowed_sampled ≤ max_disallowed_true ≤ max_disallowed_lfp
    safe_gap = min_allowed_sampled - min_allowed_lfp  (sound upper bound on coarseness)
    unsafe_gap = max_disallowed_lfp - max_disallowed_sampled  (sound upper bound on coarseness)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..Propagators.lfp_propagator import LFPPropagator
from ..Propagators.forward_sampled_belief import ForwardSampledBelief
from .metrics import MetricsCollector, MetricValue


# ============================================================
# Data classes
# ============================================================

@dataclass
class CoarsenessStepResult:
    """Per-action coarseness result at a single timestep.

    Attributes
    ----------
    action : Any
        The action being evaluated.
    min_allowed_lfp : float
        Minimum allowed probability from LFP over-approximation.
    min_allowed_sampled : float
        Minimum allowed probability from sampled under-approximation.
    max_disallowed_lfp : float
        Maximum disallowed probability from LFP over-approximation.
    max_disallowed_sampled : float
        Maximum disallowed probability from sampled under-approximation.
    """
    action: Any
    min_allowed_lfp: float
    min_allowed_sampled: float
    max_disallowed_lfp: float
    max_disallowed_sampled: float

    @property
    def safe_gap(self) -> float:
        """Sound upper bound on true coarseness for safety check.

        safe_gap = min_allowed_sampled - min_allowed_lfp >= 0
        """
        return max(0.0, self.min_allowed_sampled - self.min_allowed_lfp)

    @property
    def unsafe_gap(self) -> float:
        """Sound upper bound on true coarseness for unsafety check.

        unsafe_gap = max_disallowed_lfp - max_disallowed_sampled >= 0
        """
        return max(0.0, self.max_disallowed_lfp - self.max_disallowed_sampled)


@dataclass
class CoarsenessSnapshot:
    """Per-timestep collection of CoarsenessStepResult for all actions.

    Attributes
    ----------
    step : int
        The timestep index.
    action_results : List[CoarsenessStepResult]
        Coarseness results for each action at this timestep.
    """
    step: int
    action_results: List[CoarsenessStepResult] = field(default_factory=list)

    @property
    def max_safe_gap(self) -> float:
        """Maximum safe gap across all actions."""
        if not self.action_results:
            return 0.0
        return max(r.safe_gap for r in self.action_results)

    @property
    def max_unsafe_gap(self) -> float:
        """Maximum unsafe gap across all actions."""
        if not self.action_results:
            return 0.0
        return max(r.unsafe_gap for r in self.action_results)

    @property
    def mean_safe_gap(self) -> float:
        """Mean safe gap across all actions."""
        if not self.action_results:
            return 0.0
        return float(np.mean([r.safe_gap for r in self.action_results]))

    @property
    def mean_unsafe_gap(self) -> float:
        """Mean unsafe gap across all actions."""
        if not self.action_results:
            return 0.0
        return float(np.mean([r.unsafe_gap for r in self.action_results]))


@dataclass
class CoarsenessReport:
    """Full trajectory of coarseness snapshots with summary statistics.

    Attributes
    ----------
    snapshots : List[CoarsenessSnapshot]
        Per-timestep coarseness snapshots.
    """
    snapshots: List[CoarsenessSnapshot] = field(default_factory=list)

    @property
    def max_safe_gaps(self) -> List[float]:
        """Time series of max safe gap per timestep."""
        return [s.max_safe_gap for s in self.snapshots]

    @property
    def max_unsafe_gaps(self) -> List[float]:
        """Time series of max unsafe gap per timestep."""
        return [s.max_unsafe_gap for s in self.snapshots]

    @property
    def mean_safe_gaps(self) -> List[float]:
        """Time series of mean safe gap per timestep."""
        return [s.mean_safe_gap for s in self.snapshots]

    @property
    def mean_unsafe_gaps(self) -> List[float]:
        """Time series of mean unsafe gap per timestep."""
        return [s.mean_unsafe_gap for s in self.snapshots]

    @property
    def overall_max_safe_gap(self) -> float:
        """Maximum safe gap across all timesteps and actions."""
        if not self.snapshots:
            return 0.0
        return max(s.max_safe_gap for s in self.snapshots)

    @property
    def overall_max_unsafe_gap(self) -> float:
        """Maximum unsafe gap across all timesteps and actions."""
        if not self.snapshots:
            return 0.0
        return max(s.max_unsafe_gap for s in self.snapshots)

    @property
    def overall_mean_safe_gap(self) -> float:
        """Mean safe gap across all timesteps."""
        if not self.snapshots:
            return 0.0
        return float(np.mean([s.mean_safe_gap for s in self.snapshots]))

    @property
    def overall_mean_unsafe_gap(self) -> float:
        """Mean unsafe gap across all timesteps."""
        if not self.snapshots:
            return 0.0
        return float(np.mean([s.mean_unsafe_gap for s in self.snapshots]))


# ============================================================
# CoarsenessEvaluator
# ============================================================

class CoarsenessEvaluator:
    """Parallel comparison of LFP over-approximation and sampled under-approximation.

    Holds both an LFPPropagator and a ForwardSampledBelief, propagates
    them in tandem, and computes per-action coarseness gaps.

    Parameters
    ----------
    lfp : LFPPropagator
        The LFP-based over-approximation propagator.
    sampler : ForwardSampledBelief
        The sampling-based under-approximation propagator.
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions.
    """

    def __init__(
        self,
        lfp: LFPPropagator,
        sampler: ForwardSampledBelief,
        pp_shield: Dict,
    ):
        self.lfp = lfp
        self.sampler = sampler
        self.pp_shield = pp_shield

        # Build inverted shield maps: action -> list of safe state indices
        states = list(lfp.ipomdp.states)
        actions = list(lfp.ipomdp.actions)
        state_to_idx = {s: i for i, s in enumerate(states)}

        self.actions = actions
        self.inv_shield: Dict[Any, List[int]] = {
            a: [state_to_idx[s] for s in states if a in pp_shield[s]]
            for a in actions
        }
        self.inv_shield_complement: Dict[Any, List[int]] = {
            a: [state_to_idx[s] for s in states if a not in pp_shield[s]]
            for a in actions
        }

    def step(self, action, obs) -> CoarsenessSnapshot:
        """Propagate both propagators and compute per-action coarseness.

        Parameters
        ----------
        action : Action
            The action taken.
        obs : Observation
            The observation received.

        Returns
        -------
        CoarsenessSnapshot
            Coarseness results for this timestep.
        """
        # Propagate both
        self.lfp.propagate(action, obs)
        self.sampler.propagate(action, obs)

        # Compute per-action coarseness
        results = []
        for a in self.actions:
            allowed = self.inv_shield[a]
            disallowed = self.inv_shield_complement[a]

            min_allowed_lfp = self.lfp.minimum_allowed_probability(allowed) if allowed else 0.0
            min_allowed_sampled = self.sampler.minimum_allowed_probability(allowed) if allowed else 0.0
            max_disallowed_lfp = self.lfp.maximum_disallowed_probability(disallowed) if disallowed else 0.0
            max_disallowed_sampled = self.sampler.maximum_disallowed_probability(disallowed) if disallowed else 0.0

            results.append(CoarsenessStepResult(
                action=a,
                min_allowed_lfp=min_allowed_lfp,
                min_allowed_sampled=min_allowed_sampled,
                max_disallowed_lfp=max_disallowed_lfp,
                max_disallowed_sampled=max_disallowed_sampled,
            ))

        # Step counter based on accumulated snapshots isn't tracked here;
        # caller provides step in run_trajectory.
        return CoarsenessSnapshot(step=0, action_results=results)

    def run_trajectory(self, history: List[Tuple]) -> CoarsenessReport:
        """Run full (obs, action) sequence and return coarseness report.

        Parameters
        ----------
        history : List[Tuple]
            Sequence of (observation, action) pairs.

        Returns
        -------
        CoarsenessReport
            Full trajectory of coarseness measurements.
        """
        report = CoarsenessReport()
        for t, (obs, action) in enumerate(history):
            snapshot = self.step(action, obs)
            snapshot.step = t
            report.snapshots.append(snapshot)
        return report

    def restart(self):
        """Reset both propagators."""
        self.lfp.restart()
        self.sampler.restart()


# ============================================================
# CoarsenessMetricsCollector
# ============================================================

class CoarsenessMetricsCollector(MetricsCollector):
    """Adapts CoarsenessEvaluator to the MetricsCollector interface.

    Emits metrics: max_safe_gap, max_unsafe_gap, mean_safe_gap, num_sample_points.

    Parameters
    ----------
    sampler : ForwardSampledBelief
        The sampling-based under-approximation propagator.
    pp_shield : dict
        Perfect perception shield: state -> set of safe actions.
    """

    def __init__(self, sampler: ForwardSampledBelief, pp_shield: Dict):
        self.sampler = sampler
        self.pp_shield = pp_shield
        self._inv_shield: Optional[Dict] = None
        self._inv_shield_complement: Optional[Dict] = None
        self._metric_configs = {
            "max_safe_gap": {
                "display_name": "Max Safe Gap",
                "ylabel": "Max Safe Gap",
                "use_log_scale": False,
                "ylim": (-0.05, 1.05),
            },
            "max_unsafe_gap": {
                "display_name": "Max Unsafe Gap",
                "ylabel": "Max Unsafe Gap",
                "use_log_scale": False,
                "ylim": (-0.05, 1.05),
            },
            "mean_safe_gap": {
                "display_name": "Mean Safe Gap",
                "ylabel": "Mean Safe Gap",
                "use_log_scale": False,
                "ylim": (-0.05, 1.05),
            },
            "num_sample_points": {
                "display_name": "Sample Points",
                "ylabel": "Number of Sample Points",
                "use_log_scale": False,
                "ylim": None,
            },
        }

    def _ensure_shield_maps(self, rt_shield):
        """Build inverted shield maps from rt_shield if not already built."""
        if self._inv_shield is None:
            self._inv_shield = rt_shield.inv_shield
            self._inv_shield_complement = rt_shield.inv_shield_compliment

    def reset(self):
        """Reset internal state."""
        self._inv_shield = None
        self._inv_shield_complement = None

    def metric_names(self) -> List[str]:
        return list(self._metric_configs.keys())

    def get_plot_config(self, metric_name: str) -> Dict:
        return self._metric_configs.get(metric_name, {
            "display_name": metric_name,
            "ylabel": metric_name,
            "use_log_scale": False,
            "ylim": None,
        })

    def compute(self, rt_shield, step: int) -> Dict[str, MetricValue]:
        """Compute coarseness metrics from the current rt_shield state.

        The rt_shield's ipomdp_belief (LFPPropagator) provides the
        over-approximation; self.sampler provides the under-approximation.

        NOTE: The caller must ensure that self.sampler.propagate() has
        already been called with the same (action, obs) as rt_shield
        before calling this method.
        """
        self._ensure_shield_maps(rt_shield)

        lfp_belief = rt_shield.ipomdp_belief

        action_results = []
        for action in rt_shield.actions:
            allowed = self._inv_shield[action]
            disallowed = self._inv_shield_complement[action]

            min_allowed_lfp = lfp_belief.minimum_allowed_probability(allowed) if allowed else 0.0
            min_allowed_sampled = self.sampler.minimum_allowed_probability(allowed) if allowed else 0.0
            max_disallowed_lfp = lfp_belief.maximum_disallowed_probability(disallowed) if disallowed else 0.0
            max_disallowed_sampled = self.sampler.maximum_disallowed_probability(disallowed) if disallowed else 0.0

            action_results.append(CoarsenessStepResult(
                action=action,
                min_allowed_lfp=min_allowed_lfp,
                min_allowed_sampled=min_allowed_sampled,
                max_disallowed_lfp=max_disallowed_lfp,
                max_disallowed_sampled=max_disallowed_sampled,
            ))

        snapshot = CoarsenessSnapshot(step=step, action_results=action_results)

        metrics = {}
        for name in self.metric_names():
            config = self._metric_configs[name]
            if name == "max_safe_gap":
                value = snapshot.max_safe_gap
            elif name == "max_unsafe_gap":
                value = snapshot.max_unsafe_gap
            elif name == "mean_safe_gap":
                value = snapshot.mean_safe_gap
            elif name == "num_sample_points":
                value = float(self.sampler.points.shape[0])
            else:
                continue

            metrics[name] = MetricValue(
                name=name,
                value=value,
                display_name=config["display_name"],
                ylabel=config["ylabel"],
                use_log_scale=config["use_log_scale"],
                ylim=config["ylim"],
            )

        return metrics
