"""LFP-specific report runners for template comparison.

Provides reporters that collect metrics during LFP belief propagation runs.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..Propagators import LFPPropagator, BeliefPolytope, Template

from .report_runner import ReportRunner, ScriptedReportRunner
from .runtime_shield import RuntimeImpShield
from .metrics import (
    MetricsCollector, StepMetrics, ApproximationMetrics_1,
    GroundTruthComparisonMetrics, StepPredictions,
)


@dataclass
class ComparisonResult:
    """Results from comparing templates on a single run."""
    template_name: str
    template: Template
    history: List[Tuple]
    metrics: List[StepMetrics]
    final_polytope: BeliefPolytope
    # Ground truth comparison data (populated when GroundTruthComparisonMetrics is used)
    true_states: Optional[List[Any]] = field(default=None)
    action_predictions: Optional[List[Dict[Any, float]]] = field(default=None)
    state_bound_predictions: Optional[List[Dict[int, Tuple[float, float]]]] = field(default=None)


class LFPReporter(ReportRunner):
    """Report runner that collects metrics during LFP belief propagation.

    Accepts a MetricsCollector to allow flexible metric computation.
    """

    def __init__(
        self,
        ipomdp,
        perception,
        rt_shield: RuntimeImpShield,
        action_selector,
        length: int,
        initial,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        super().__init__(ipomdp, perception, rt_shield, action_selector, length, initial)
        self.metrics_collector = metrics_collector or ApproximationMetrics_1()

    def initialize(self):
        self.metrics: List[StepMetrics] = []
        self.history: List[Tuple] = []
        self.metrics_collector.reset()

    def report(self, step, state, obs, action, rt_shield):
        metric_values = self.metrics_collector.compute(rt_shield, step)
        step_metrics = StepMetrics(step=step + 1, values=metric_values)
        self.metrics.append(step_metrics)
        self.history.append((step, obs, action))

    def final(self, rt_shield) -> ComparisonResult:
        assert isinstance(rt_shield.ipomdp_belief, LFPPropagator)

        template = rt_shield.ipomdp_belief.template
        polytope = rt_shield.ipomdp_belief.belief

        return ComparisonResult(
            template_name=getattr(template, 'name', 'unnamed'),
            template=template,
            history=self.history,
            metrics=self.metrics,
            final_polytope=polytope
        )


class ScriptedLFPReporter(ScriptedReportRunner):
    """LFP reporter that replays a pre-recorded script.

    Collects metrics on a fixed trajectory, enabling fair comparison
    between templates. Accepts a MetricsCollector for flexible metrics.

    When using GroundTruthComparisonMetrics, also stores true states and
    detailed per-action/per-state predictions for ground truth comparison.
    """

    def __init__(
        self,
        script,
        rt_shield: RuntimeImpShield,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        super().__init__(script, rt_shield)
        self.metrics_collector = metrics_collector or ApproximationMetrics_1()

    def initialize(self):
        self.metrics: List[StepMetrics] = []
        self.history: List[Tuple] = []
        self.shield_decisions: List[Dict] = []
        self.true_states: List[Any] = []
        self.metrics_collector.reset()

    def report(self, step, state, obs, scripted_action, allowed_actions, rt_shield):
        metric_values = self.metrics_collector.compute(rt_shield, step)
        step_metrics = StepMetrics(step=step + 1, values=metric_values)
        self.metrics.append(step_metrics)
        self.history.append((step, obs, scripted_action))
        self.true_states.append(state)
        self.shield_decisions.append({
            "step": step,
            "state": state,
            "scripted_action": scripted_action,
            "allowed_actions": allowed_actions,
            "would_block": scripted_action not in allowed_actions if allowed_actions else True
        })

    def final(self, rt_shield) -> ComparisonResult:
        assert isinstance(rt_shield.ipomdp_belief, LFPPropagator)

        template = rt_shield.ipomdp_belief.template
        polytope = rt_shield.ipomdp_belief.belief

        # Extract detailed predictions if using GroundTruthComparisonMetrics
        action_predictions = None
        state_bound_predictions = None
        true_states = self.true_states if self.true_states else None

        if isinstance(self.metrics_collector, GroundTruthComparisonMetrics):
            predictions = self.metrics_collector.step_predictions
            action_predictions = [p.action_min_probs for p in predictions]
            state_bound_predictions = [p.state_bounds for p in predictions]

        return ComparisonResult(
            template_name=getattr(template, 'name', 'unnamed'),
            template=template,
            history=self.history,
            metrics=self.metrics,
            final_polytope=polytope,
            true_states=true_states,
            action_predictions=action_predictions,
            state_bound_predictions=state_bound_predictions,
        )
