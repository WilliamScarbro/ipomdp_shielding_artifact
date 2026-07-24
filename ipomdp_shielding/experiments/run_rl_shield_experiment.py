"""RL Shielding Experiment: Compare shield strategies for RL action selection.

Evaluates RL action selection under four shielding strategies, with
uniform-random and adversarial-optimized perception realizations.

Usage:
    python -m ipomdp_shielding.experiments.run_rl_shield_experiment <config_module>

Example:
    python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim
"""

import os
import sys
import json
import time
import importlib
import random as random_module

from .experiment_io import build_metadata, add_rate_cis, save_experiment_results

from ..Evaluation.runtime_shield import RuntimeImpShield
from ..Propagators import LFPPropagator, BeliefPolytope, TemplateFactory, ForwardSampledBelief
from ..Propagators.lfp_propagator import default_solver
from ..Models.pomdp import POMDP_Belief
from ..MonteCarlo import (
    UniformPerceptionModel,
    AdversarialPerceptionModel,
    FixedRealizationPerceptionModel,
    train_optimal_realization,
    RandomActionSelector,
    BeliefSelector,
    NeuralActionSelector,
    ActionSelector,
    RandomInitialState,
    run_monte_carlo_trials,
    compute_safety_metrics,
)


# ============================================================
# Shield classes
# ============================================================

class NoShield:
    """Passthrough shield: allows all actions."""

    def __init__(self, all_actions):
        self.all_actions = list(all_actions)
        self.stuck_count = 0
        self.error_count = 0

    def next_actions(self, _evidence):
        return list(self.all_actions)

    def get_action_probs(self):
        return []

    def restart(self):
        self.stuck_count = 0
        self.error_count = 0

    def initialize(self, _initial_state):
        self.restart()


class ObservationShield:
    """Observation-based shield using a probability threshold.

    At each step computes a posterior P(s | obs) using the midpoint
    observation model O_mid(obs|s) = (P_lower + P_upper) / 2 with a
    uniform prior, then allows action a iff P(a safe | obs) >= threshold.

    This is a memoryless shield: it uses only the current observation with
    no belief propagation over history.  It is equivalent to SingleBeliefShield
    reset to the uniform prior at every step.
    """

    def __init__(self, ipomdp, pp_shield, threshold=0.8):
        self.pp_shield = pp_shield
        self.threshold = threshold
        self.all_actions = list(ipomdp.actions)
        self.stuck_count = 0
        self.error_count = 0

        # Precompute P(s | obs) for every observation (uniform prior,
        # midpoint observation probabilities).
        self._posterior = {}
        for obs in ipomdp.observations:
            weights = {}
            for s in ipomdp.states:
                p_mid = (ipomdp.P_lower[s].get(obs, 0.0)
                         + ipomdp.P_upper[s].get(obs, 0.0)) / 2.0
                if p_mid > 0:
                    weights[s] = p_mid
            total = sum(weights.values())
            if total > 0:
                self._posterior[obs] = {s: w / total for s, w in weights.items()}

    def _p_safe(self, obs, action):
        """P(action safe | obs) = sum_{s: action in pp_shield[s]} P(s | obs)."""
        posterior = self._posterior.get(obs)
        if posterior is None:
            return 1.0  # unknown observation — allow all
        return sum(
            p for s, p in posterior.items()
            if action in self.pp_shield.get(s, set())
        )

    def next_actions(self, evidence):
        obs, _action = evidence
        allowed = [a for a in self.all_actions if self._p_safe(obs, a) >= self.threshold]
        if not allowed:
            self.stuck_count += 1
        return allowed

    def get_action_probs(self):
        return []

    def restart(self):
        self.stuck_count = 0
        self.error_count = 0

    def initialize(self, _initial_state):
        self.restart()


class SingleBeliefShield:
    """Shield using single-point POMDP belief propagation.

    Maintains a standard POMDP belief (point estimate) and filters
    actions based on P(action allowed | belief) >= threshold.
    """

    def __init__(self, pomdp, pp_shield, threshold=0.8):
        self.pomdp = pomdp
        self.pp_shield = pp_shield
        self.threshold = threshold
        self.belief = POMDP_Belief(pomdp)
        self.stuck_count = 0
        self.error_count = 0

    def _allowance_probability(self, action):
        """P(action allowed | belief) = sum_{s: action in pp_shield[s]} belief(s)."""
        return sum(
            self.belief.belief.get(s, 0.0)
            for s, allowed_set in self.pp_shield.items()
            if action in allowed_set
        )

    def next_actions(self, evidence):
        self.belief.propogate(evidence)
        allowed = [
            a for a in self.pomdp.actions
            if self._allowance_probability(a) >= self.threshold
        ]
        if not allowed:
            self.stuck_count += 1
        return allowed

    def get_action_probs(self):
        """Return (action, allowed_prob, disallowed_prob) triples."""
        return [
            (a, self._allowance_probability(a), 1.0 - self._allowance_probability(a))
            for a in self.pomdp.actions
        ]

    def restart(self):
        self.belief.restart()
        self.stuck_count = 0
        self.error_count = 0

    def initialize(self, _initial_state):
        self.restart()


# ============================================================
# Shield factories
# ============================================================

def create_no_shield_factory(all_actions):
    def factory():
        return NoShield(all_actions)
    return factory


def create_observation_shield_factory(ipomdp, pp_shield, threshold=0.8):
    def factory():
        return ObservationShield(ipomdp, pp_shield, threshold)
    return factory


def create_single_belief_shield_factory(pomdp, pp_shield, threshold=0.8):
    def factory():
        return SingleBeliefShield(pomdp, pp_shield, threshold)
    return factory


def create_envelope_shield_factory(ipomdp, pp_shield, threshold=0.8):
    def factory():
        n = len(ipomdp.states)
        template = TemplateFactory.canonical(n)
        polytope = BeliefPolytope.uniform_prior(n)
        propagator = LFPPropagator(ipomdp, template, default_solver(), polytope)
        return RuntimeImpShield(pp_shield, propagator, action_shield=threshold)
    return factory


def create_forward_sampling_shield_factory(ipomdp, pp_shield, threshold=0.8,
                                           budget=500, K_samples=100):
    """Forward-sampled belief envelope shield.

    Uses ForwardSampledBelief to maintain N concrete belief points as an
    inner approximation of the reachable belief set.  Action filtering uses
    the same probability threshold as the envelope shield but without LP solves.
    """
    def factory():
        fsb = ForwardSampledBelief(ipomdp=ipomdp, budget=budget, K_samples=K_samples)
        return RuntimeImpShield(pp_shield, fsb, action_shield=threshold)
    return factory


class ConformalPredictionShield:
    """Memoryless shield using conformal prediction-set observations.

    Designed for case studies (e.g. TaxiNetV2) where the observation is a
    conformal prediction set: a tuple of axis-wise sets
    ``(cte_set, he_set)`` enumerating possible true state values on each axis.

    An action is allowed iff it is safe for *every* state in the Cartesian
    product of the conformal set.  This is the exact runtime counterpart of
    the Scarbro PRISM-based shielding approach, evaluated empirically rather
    than via formal model checking.

    If an axis is empty (the conformal set omitted that axis), the shield
    falls back to the full axis range supplied at construction time, which is
    the most conservative choice.  The string sentinel "FAIL" as observation
    is treated as stuck (no actions allowed).
    """

    def __init__(self, pp_shield, all_actions, cte_states=None, he_states=None):
        self.pp_shield = pp_shield
        self.all_actions = list(all_actions)
        self._cte_states = list(cte_states) if cte_states else []
        self._he_states = list(he_states) if he_states else []
        self.stuck_count = 0
        self.error_count = 0

    def next_actions(self, evidence):
        obs, _action = evidence

        if obs == "FAIL":
            self.stuck_count += 1
            return []

        cte_set, he_set = obs

        effective_cte = list(cte_set) if cte_set else self._cte_states
        effective_he = list(he_set) if he_set else self._he_states

        if not effective_cte or not effective_he:
            self.stuck_count += 1
            return []

        allowed = set(self.all_actions)
        for cte in effective_cte:
            for he in effective_he:
                allowed &= self.pp_shield.get((cte, he), set())

        if not allowed:
            self.stuck_count += 1
        return list(allowed)

    def get_action_probs(self):
        return []

    def restart(self):
        self.stuck_count = 0
        self.error_count = 0

    def initialize(self, _initial_state):
        self.restart()


def create_conformal_shield_factory(pp_shield, all_actions,
                                    cte_states=None, he_states=None):
    """Factory for ConformalPredictionShield.

    Parameters
    ----------
    pp_shield : dict
        Perfect-perception shield: state -> set of safe actions.
    all_actions : list
        Full action space.
    cte_states : list, optional
        All possible CTE state values (fallback for empty conformal axis).
    he_states : list, optional
        All possible HE state values (fallback for empty conformal axis).
    """
    def factory():
        return ConformalPredictionShield(pp_shield, all_actions, cte_states, he_states)
    return factory


# ============================================================
# Action selector wrapper for RL
# ============================================================

class ShieldCompliantSelector(ActionSelector):
    """Wraps a selector that ignores allowed_actions (e.g. NeuralActionSelector)
    to respect the shield's filtering.

    If the primary selector's preferred action is in allowed_actions, use it.
    Otherwise, fall back to random choice from allowed_actions.

    This keeps action selection cleanly separated from shielding --
    the fallback is shield-agnostic.
    """

    def __init__(self, primary_selector, all_actions):
        self.primary = primary_selector
        self.all_actions = all_actions
        self.primary_count = 0
        self.fallback_count = 0

    def select(self, history, allowed_actions, context=None):
        if not allowed_actions:
            return self.primary.select(history, self.all_actions, context)

        preferred = self.primary.select(history, self.all_actions, context)
        if preferred in allowed_actions:
            self.primary_count += 1
            return preferred
        self.fallback_count += 1
        return random_module.choice(allowed_actions)

    def reset_stats(self):
        self.primary_count = 0
        self.fallback_count = 0


# ============================================================
# Setup: train/cache RL agent & optimized realization
# ============================================================

def setup(ipomdp, pp_shield, config):
    """Train or load RL agent and optimized realizations.

    RL agent is trained with adversarial-greedy perception (per plan).
    Returns (rl_selector, optimized_perceptions_by_target, setup_info).
    """
    # --- RL Agent (trained with adversarial-greedy perception) ---
    print("\n" + "=" * 70)
    print("SETUP: RL AGENT")
    print("=" * 70)

    if os.path.exists(config.rl_cache_path):
        print(f"Loading cached RL agent from {config.rl_cache_path}")
        rl_selector = NeuralActionSelector.load(config.rl_cache_path, ipomdp)
    else:
        print(f"Training RL agent ({config.rl_episodes} episodes, adversarial-greedy perception)...")
        rl_selector = NeuralActionSelector(
            actions=list(ipomdp.actions),
            observations=ipomdp.observations,
            maximize_safety=True,
        )
        train_metrics = rl_selector.train(
            ipomdp=ipomdp,
            perception=AdversarialPerceptionModel(pp_shield),
            num_episodes=config.rl_episodes,
            episode_length=config.rl_episode_length,
            verbose=True,
        )
        rl_selector.save(config.rl_cache_path)
        print(f"RL agent saved to {config.rl_cache_path}")
        print(f"  Final safe rate: {train_metrics['final_safe_rate']:.2%}")

    rl_selector.exploration_rate = 0.0

    # --- Optimized Realizations (fixed interval realizations) ---
    print("\n" + "=" * 70)
    print("SETUP: OPTIMIZED REALIZATIONS")
    print("=" * 70)

    def _cache_path_for_target(base_path: str, target: str) -> str:
        if target == "envelope":
            return base_path
        if base_path.endswith(".json"):
            return base_path[:-5] + f"_{target}.json"
        return base_path + f"_{target}"

    targets = getattr(config, "adversarial_opt_targets", ["envelope"])
    targets = list(dict.fromkeys(targets))  # preserve order, de-dup

    pomdp = ipomdp.to_pomdp()
    optimized_perceptions = {}
    opt_cache_paths = {}
    opt_cached = {}

    for target in targets:
        cache_path = _cache_path_for_target(config.opt_cache_path, target)
        opt_cache_paths[target] = cache_path
        opt_cached[target] = os.path.exists(cache_path)

        if opt_cached[target]:
            print(f"Loading cached optimized realization ({target}) from {cache_path}")
            optimized_perceptions[target] = FixedRealizationPerceptionModel.load(cache_path)
            continue

        print(f"Training optimized realization ({target}) ({config.opt_iterations} iterations)...")
        if target == "envelope":
            rt_shield_factory = create_envelope_shield_factory(
                ipomdp, pp_shield, threshold=config.shield_threshold
            )
        elif target == "single_belief":
            rt_shield_factory = create_single_belief_shield_factory(
                pomdp, pp_shield, threshold=config.shield_threshold
            )
        else:
            raise ValueError(
                f"Unknown adversarial_opt_target={target!r}. Supported: 'envelope', 'single_belief'."
            )

        optimized_perception = train_optimal_realization(
            ipomdp=ipomdp,
            pp_shield=pp_shield,
            rt_shield_factory=rt_shield_factory,
            action_selector=ShieldCompliantSelector(rl_selector, list(ipomdp.actions)),
            initial_generator=RandomInitialState(),
            num_candidates=config.opt_candidates,
            num_trials_per_candidate=config.opt_trials_per_candidate,
            max_iterations=config.opt_iterations,
            trial_length=config.trial_length,
            save_path=cache_path,
            verbose=True,
        )
        optimized_perceptions[target] = optimized_perception
        print(f"Optimized realization ({target}) saved to {cache_path}")
        score = optimized_perception.metadata.get("objective_score", "N/A")
        print(f"  Best score ({target}): {score}")

    setup_info = {
        "rl_agent_cached": os.path.exists(config.rl_cache_path),
        "rl_cache_path": config.rl_cache_path,
        "adversarial_opt_targets": targets,
        "opt_realization_cached_by_target": opt_cached,
        "opt_cache_paths_by_target": opt_cache_paths,
    }
    return rl_selector, optimized_perceptions, setup_info


# ============================================================
# Build experiment grid
# ============================================================

def build_grid(ipomdp, pp_shield, rl_selector, optimized_perceptions, config):
    """Build the 3-factor experiment grid.

    Factors:
      - Perception: uniform, adversarial_opt
      - Selector: random, best, rl
      - Shield: none, observation, single_belief, envelope, forward_sampling

    Returns list of (p_name, s_name, sh_name,
                     perception, selector, shield_factory) tuples.
    """
    all_actions = list(ipomdp.actions)
    pomdp = ipomdp.to_pomdp()

    # Factor 1: Perception models
    uniform_perception = UniformPerceptionModel()

    def perception_for(p_name: str, sh_name: str):
        if p_name == "uniform":
            return uniform_perception
        if p_name != "adversarial_opt":
            raise ValueError(f"Unknown perception regime: {p_name!r}")
        if not optimized_perceptions:
            raise ValueError("No optimized perceptions available for adversarial_opt regime.")
        # If a realization was optimized against this shield, use it.
        if sh_name in optimized_perceptions:
            return optimized_perceptions[sh_name]
        # Otherwise default to envelope if present, else first available.
        if "envelope" in optimized_perceptions:
            return optimized_perceptions["envelope"]
        return next(iter(optimized_perceptions.values()))

    # Factor 2: Action selectors (independent of shield)
    selectors = {
        "random": RandomActionSelector(),
        "best": BeliefSelector(mode="best"),
        "rl": ShieldCompliantSelector(rl_selector, all_actions),
    }

    # Factor 3: Shield strategies (independent of selector)
    shields = {
        "none": create_no_shield_factory(all_actions),
        "observation": create_observation_shield_factory(ipomdp, pp_shield, config.shield_threshold),
        "single_belief": create_single_belief_shield_factory(
            pomdp, pp_shield, config.shield_threshold),
        "envelope": create_envelope_shield_factory(
            ipomdp, pp_shield, config.shield_threshold),
        "forward_sampling": create_forward_sampling_shield_factory(
            ipomdp, pp_shield, config.shield_threshold),
    }

    grid = []
    for p_name in ["uniform", "adversarial_opt"]:
        for s_name, selector in selectors.items():
            for sh_name, shield_factory in shields.items():
                grid.append((p_name, s_name, sh_name,
                             perception_for(p_name, sh_name), selector, shield_factory))

    return grid


# ============================================================
# Run experiment
# ============================================================

def run_experiment(ipomdp, pp_shield, grid, config):
    """Run all grid combinations, collecting per-trial results.

    Returns (results, trial_data, intervention_stats) where:
      - results: dict mapping (perception, selector, shield) -> MCSafetyMetrics
      - trial_data: dict mapping same keys -> list of SafetyTrialResult
      - intervention_stats: dict mapping same keys -> {primary, fallback, rate}
    """
    results = {}
    trial_data = {}
    intervention_stats = {}
    total = len(grid)

    for i, (p_name, s_name, sh_name, perception, selector, sh_factory) in enumerate(grid):
        label = f"{p_name}/{s_name}/{sh_name}"
        print(f"\n[{i+1}/{total}] Running: {label} ...", end=" ", flush=True)

        # Reset intervention counters for RL selector
        if isinstance(selector, ShieldCompliantSelector):
            selector.reset_stats()

        t0 = time.time()
        trial_results = run_monte_carlo_trials(
            ipomdp=ipomdp,
            pp_shield=pp_shield,
            perception=perception,
            rt_shield_factory=sh_factory,
            action_selector=selector,
            initial_generator=RandomInitialState(),
            num_trials=config.num_trials,
            trial_length=config.trial_length,
            seed=config.seed,
        )
        metrics = compute_safety_metrics(trial_results)
        elapsed = time.time() - t0

        key = (p_name, s_name, sh_name)
        results[key] = metrics
        trial_data[key] = trial_results

        # Collect intervention stats
        if isinstance(selector, ShieldCompliantSelector):
            total_decisions = selector.primary_count + selector.fallback_count
            rate = selector.fallback_count / total_decisions if total_decisions > 0 else 0.0
            intervention_stats[key] = {
                "primary_count": selector.primary_count,
                "fallback_count": selector.fallback_count,
                "intervention_rate": rate,
            }

        print(f"fail={metrics.fail_rate:.1%}  stuck={metrics.stuck_rate:.1%}  "
              f"safe={metrics.safe_rate:.1%}  ({elapsed:.1f}s)")

    return results, trial_data, intervention_stats


# ============================================================
# Per-timestep analysis
# ============================================================

def compute_timestep_outcomes(trial_results, trial_length):
    """Compute per-timestep cumulative outcome probabilities.

    At timestep t:
      P(fail) = fraction of trials that failed at or before step t
      P(stuck) = fraction of trials that got stuck at or before step t
      P(safe) = 1 - P(fail) - P(stuck)

    Returns dict with keys 'fail', 'stuck', 'safe', each a list of
    length trial_length.
    """
    n = len(trial_results)
    if n == 0:
        return {"fail": [0.0] * trial_length,
                "stuck": [0.0] * trial_length,
                "safe": [1.0] * trial_length}

    fail_by_t = [0] * trial_length
    stuck_by_t = [0] * trial_length

    for trial in trial_results:
        if trial.outcome == "fail" and trial.fail_step is not None:
            for t in range(trial.fail_step, trial_length):
                fail_by_t[t] += 1
        elif trial.outcome == "stuck":
            for t in range(trial.steps_completed, trial_length):
                stuck_by_t[t] += 1

    return {
        "fail": [f / n for f in fail_by_t],
        "stuck": [s / n for s in stuck_by_t],
        "safe": [1.0 - fail_by_t[t] / n - stuck_by_t[t] / n
                 for t in range(trial_length)],
    }


# ============================================================
# Plotting
# ============================================================

def plot_results(trial_data, config, intervention_stats=None):
    """Generate figures: P(fail) and P(stuck) per perception, plus intervention rate.

    Replaces the old 3-panel (fail/stuck/safe) layout — safe is redundant
    since safe = 1 - fail - stuck.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(config.figures_dir, exist_ok=True)

    perceptions = ["uniform", "adversarial_opt"]
    outcomes = ["fail", "stuck"]
    shield_order = ["none", "observation", "single_belief", "envelope", "forward_sampling", "conformal_prediction"]

    shield_labels = {
        "none": "No Shield",
        "observation": "Observation Shield",
        "single_belief": "Single-Belief Shield",
        "envelope": "Envelope Shield",
        "forward_sampling": "Forward-Sampling Shield",
        "conformal_prediction": "Conf. Pred. Shield",
    }
    shield_colors = {
        "none": "gray",
        "observation": "orange",
        "single_belief": "blue",
        "envelope": "green",
        "forward_sampling": "teal",
        "conformal_prediction": "red",
    }
    perception_labels = {
        "uniform": "Uniform Random",
        "adversarial_opt": "Adversarial Optimized",
    }
    selector_styles = {
        "rl": ("-", 2.0, 1.0),
        "best": ("--", 1.5, 0.8),
        "random": (":", 1.2, 0.5),
    }

    timesteps = list(range(config.trial_length))

    # Main outcome plots (fail + stuck only)
    for p_name in perceptions:
        for outcome in outcomes:
            fig, ax = plt.subplots(figsize=(10, 6))

            for sh_name in shield_order:
                color = shield_colors[sh_name]
                for s_name, (ls, lw, alpha) in selector_styles.items():
                    key = (p_name, s_name, sh_name)
                    if key not in trial_data:
                        continue
                    ts = compute_timestep_outcomes(trial_data[key], config.trial_length)
                    label = f"{s_name.upper()} + {shield_labels[sh_name]}"
                    ax.plot(timesteps, ts[outcome],
                            label=label, color=color,
                            linestyle=ls, linewidth=lw, alpha=alpha)

            ax.set_xlabel("Timestep")
            ax.set_ylabel(f"P({outcome})")
            ax.set_title(f"{outcome.capitalize()} Rate by Timestep\n"
                         f"Perception: {perception_labels[p_name]} ({config.case_study_name.upper()})")
            ax.legend(loc="best", fontsize=7)
            ax.set_xlim(0, config.trial_length - 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            fname = f"{p_name}_{outcome}.png"
            fig.savefig(os.path.join(config.figures_dir, fname),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {fname}")

    # Intervention rate bar chart (RL selector only)
    if intervention_stats:
        for p_name in perceptions:
            fig, ax = plt.subplots(figsize=(8, 5))
            shields_with_data = []
            rates = []
            for sh_name in shield_order:
                key = (p_name, "rl", sh_name)
                if key in intervention_stats:
                    shields_with_data.append(shield_labels[sh_name])
                    rates.append(intervention_stats[key]["intervention_rate"])

            if shields_with_data:
                bars = ax.bar(shields_with_data, rates,
                              color=[shield_colors[s] for s in shield_order
                                     if (p_name, "rl", s) in intervention_stats])
                ax.set_ylabel("Intervention Rate")
                ax.set_title(f"RL Shield Intervention Rate\n"
                             f"Perception: {perception_labels[p_name]} ({config.case_study_name.upper()})")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3, axis="y")
                for bar, rate in zip(bars, rates):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{rate:.1%}", ha="center", fontsize=9)

                fname = f"{p_name}_intervention_rate.png"
                fig.savefig(os.path.join(config.figures_dir, fname),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved {fname}")


# ============================================================
# Results table and analysis
# ============================================================

def print_results_table(results, config):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"RL SHIELDING EXPERIMENT RESULTS - {config.case_study_name.upper()}")
    print("=" * 90)
    header = (f"{'Perception':<17} {'Selector':<10} {'Shield':<16} "
              f"{'Fail%':>7} {'Stuck%':>7} {'Safe%':>7} {'Steps':>7}")
    print(header)
    print("-" * 90)

    for key in sorted(results.keys()):
        p_name, s_name, sh_name = key
        m = results[key]
        print(f"{p_name:<17} {s_name:<10} {sh_name:<16} "
              f"{m.fail_rate:>6.1%} {m.stuck_rate:>6.1%} {m.safe_rate:>6.1%} "
              f"{m.mean_steps:>7.1f}")

    # Shield comparison for RL
    print("\n" + "=" * 90)
    print("ANALYSIS: SHIELD COMPARISON FOR RL")
    print("=" * 90)
    for p_name in ["uniform", "adversarial_opt"]:
        print(f"\n  Perception: {p_name}")
        for sh_name in ["none", "observation", "single_belief", "envelope", "forward_sampling", "conformal_prediction"]:
            key = (p_name, "rl", sh_name)
            if key in results:
                m = results[key]
                print(f"    {sh_name:<16} fail={m.fail_rate:.1%}  "
                      f"stuck={m.stuck_rate:.1%}  safe={m.safe_rate:.1%}")

    # Envelope vs single-belief under different perceptions
    print("\n" + "=" * 90)
    print("ANALYSIS: ENVELOPE vs SINGLE-BELIEF (expected: envelope better under adversarial)")
    print("=" * 90)
    for s_name in ["rl", "random", "best"]:
        for p_name in ["uniform", "adversarial_opt"]:
            sb_key = (p_name, s_name, "single_belief")
            env_key = (p_name, s_name, "envelope")
            if sb_key in results and env_key in results:
                sb_fail = results[sb_key].fail_rate
                env_fail = results[env_key].fail_rate
                diff = env_fail - sb_fail
                print(f"  {s_name}/{p_name}: envelope - single_belief = "
                      f"{diff:+.1%} fail rate")


def save_results(results, config, setup_info=None):
    """Save results to JSON with full metadata and CIs."""
    serializable = {}
    tidy_rows = []
    for (p, s, sh), m in results.items():
        key = f"{p}/{s}/{sh}"
        cell = {
            "fail_rate": m.fail_rate,
            "stuck_rate": m.stuck_rate,
            "safe_rate": m.safe_rate,
            "mean_steps": m.mean_steps,
            "num_trials": m.num_trials,
        }
        add_rate_cis(cell, m.num_trials)
        serializable[key] = cell
        tidy_rows.append({
            "perception": p,
            "selector": s,
            "shield": sh,
            **cell,
        })

    extra_meta = {}
    if setup_info:
        extra_meta.update(setup_info)
    extra_meta["perception_regimes"] = {
        "uniform": "UniformPerceptionModel",
        "adversarial_opt": (
            "FixedRealizationPerceptionModel (trained via train_optimal_realization; "
            "if multiple adversarial_opt_targets are provided, the adversarial_opt "
            "realization is selected per shield when available)"
        ),
    }
    extra_meta["selectors"] = {
        "random": "RandomActionSelector",
        "best": "BeliefSelector(mode='best')",
        "rl": "NeuralActionSelector wrapped by ShieldCompliantSelector",
    }
    extra_meta["shields"] = {
        "none": "NoShield (passthrough)",
        "observation": f"ObservationShield (midpoint posterior, threshold={config.shield_threshold})",
        "single_belief": f"SingleBeliefShield (POMDP belief, threshold={config.shield_threshold})",
        "envelope": f"RuntimeImpShield (LFP polytope, threshold={config.shield_threshold})",
        "forward_sampling": (
            f"ForwardSamplingShield (forward-sampled belief envelope, threshold={config.shield_threshold})"
        ),
        "conformal_prediction": (
            "ConformalPredictionShield (memoryless; allows action iff safe for all states "
            "in the conformal set; TaxiNetV2 only)"
        ),
    }

    metadata = build_metadata(config, extra=extra_meta)
    save_experiment_results(config.results_path, serializable, metadata, tidy_rows)
    print(f"\nResults saved to {config.results_path}")


# ============================================================
# Main
# ============================================================

def run_rl_experiment(config):
    """Run the full RL shielding experiment programmatically.

    Parameters
    ----------
    config : RLShieldExperimentConfig
        Full experiment configuration.

    Returns
    -------
    (results, trial_data, metadata) where:
        results : dict mapping (perception, selector, shield) -> MCSafetyMetrics
        trial_data : dict mapping same keys -> list of SafetyTrialResult
        metadata : dict with full experiment metadata
    """
    print("=" * 70)
    print(f"RL SHIELDING EXPERIMENT - {config.case_study_name.upper()}")
    print(f"Trials: {config.num_trials}, Length: {config.trial_length}, Seed: {config.seed}")
    print(f"Shield threshold: {config.shield_threshold}")
    print("=" * 70)

    # Load IPOMDP
    print(f"\nLoading {config.case_study_name.upper()} IPOMDP...")
    ipomdp, pp_shield, _, _ = config.build_ipomdp_fn(**config.ipomdp_kwargs)
    print(f"  States: {len(ipomdp.states)}, Actions: {len(ipomdp.actions)}, "
          f"Observations: {len(ipomdp.observations)}")

    # Setup: train/load RL agent and optimized realizations
    rl_selector, optimized_perceptions, setup_info = setup(ipomdp, pp_shield, config)

    # Build 3-factor grid
    grid = build_grid(ipomdp, pp_shield, rl_selector, optimized_perceptions, config)
    print(f"\nExperiment grid: {len(grid)} combinations "
          f"(2 perceptions x 3 selectors x 5 shields)")

    # Run all combinations
    t0 = time.time()
    results, trial_data, intervention_stats = run_experiment(ipomdp, pp_shield, grid, config)
    total_time = time.time() - t0

    # Results
    print_results_table(results, config)
    save_results(results, config, setup_info={**setup_info, "intervention_stats": {
        f"{k[0]}/{k[1]}/{k[2]}": v for k, v in intervention_stats.items()
    }})

    # Plots
    print("\nGenerating figures...")
    plot_results(trial_data, config, intervention_stats=intervention_stats)

    print(f"\nTotal experiment time: {total_time:.1f}s")
    print(f"Figures saved to {config.figures_dir}")
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    metadata = build_metadata(config, extra={**setup_info, "total_time_s": total_time})
    return results, trial_data, metadata


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ipomdp_shielding.experiments.run_rl_shield_experiment <config_module>")
        print("Example: python -m ipomdp_shielding.experiments.run_rl_shield_experiment configs.rl_shield_taxinet_prelim")
        sys.exit(1)

    # Import config
    config_module_name = sys.argv[1]
    try:
        config_module = importlib.import_module(f".{config_module_name}", package="ipomdp_shielding.experiments")
        config = config_module.config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    run_rl_experiment(config)


if __name__ == "__main__":
    main()
