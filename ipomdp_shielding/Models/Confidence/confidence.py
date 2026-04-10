"""Unified confidence interval wrapper for IMDP construction."""

from typing import Iterable, Hashable, Tuple

from ..imdp import IMDP
from .clopper_pearson import estimate_clopper_pearson
from .sidak_wilson import estimate_sidak_wilson_hybrid
from .goodman import estimate_goodman


class ConfidenceInterval:
    """
    Wrapper class unifying three confidence interval methods for
    constructing interval MDPs from observation data.
    """

    def __init__(self, data: Iterable[Tuple[Hashable, Hashable]]):
        self.data = list(data)

    def Sidak_Wilson_Hybrid(self, alpha: float):
        """Tightest intervals, relies on binomial independence assumption."""
        return estimate_sidak_wilson_hybrid(self.data, alpha=alpha)

    def Goodman(self, alpha: float):
        """Captures multinomial interaction, assumes chi-squared model."""
        return estimate_goodman(self.data, alpha=alpha)

    def Clopper_Pearson(self, alpha: float):
        """Wider intervals but stronger guarantee (weaker distributional assumption)."""
        return estimate_clopper_pearson(self.data, alpha=alpha)

    def produce_imdp(self, states, percieve_action, confidence_method: str, alpha: float) -> IMDP:
        """
        Construct an IMDP perception model from confidence intervals.

        Parameters
        ----------
        states : list
            The state space.
        percieve_action : hashable
            The action label for perception transitions.
        confidence_method : str
            One of "Goodman", "Clopper_Pearson", "Sidak_Wilson_Hybrid".
        alpha : float
            Significance level.

        Returns
        -------
        IMDP
            The interval MDP with perception intervals.
        """
        methods = {
            "Goodman": self.Goodman,
            "Clopper_Pearson": self.Clopper_Pearson,
            "Sidak_Wilson_Hybrid": self.Sidak_Wilson_Hybrid
        }

        if confidence_method not in methods:
            raise ValueError(f"Unknown confidence method {confidence_method}")

        result = methods[confidence_method](alpha)

        imdp_low = {}
        imdp_high = {}

        for s1 in result:
            s2_map_low = {}
            s2_map_high = {}
            for s2 in result[s1]:
                s2_map_low[s2] = result[s1][s2]["lower"]
                s2_map_high[s2] = result[s1][s2]["upper"]
            imdp_low[(s1, percieve_action)] = s2_map_low
            imdp_high[(s1, percieve_action)] = s2_map_high

        actions = {s: [percieve_action] for s in states}
        return IMDP(states, actions, imdp_low, imdp_high)
