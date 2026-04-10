"""Runtime imperfect perception shield."""

from typing import Dict, Set, Tuple, Any

from ..Propagators import IPOMDP_Belief


class RuntimeImpShield:
    """
    Runtime shield enforcing safety via belief propagation.

    Uses a pre-computed perfect-perception shield and probability threshold
    to filter actions based on current belief state.
    """

    def __init__(
        self,
        pp_shield: "Dict State -> (Col Action)",
        ipomdp_belief: IPOMDP_Belief,
        action_shield: float,
        default_action: "Action" = None
    ):
        """
        Initialize runtime shield.
 
        Parameters
        ----------
        pp_shield : dict
            Perfect perception shield: state -> set of safe actions
        ipomdp_belief : IPOMDP_Belief
            Belief propagator for tracking state uncertainty
        action_shield : float
            Minimum required probability for an action to be allowed
        default_action : any
            Action to use when no actions pass the threshold
        """
        self.actions = ipomdp_belief.ipomdp.actions
        self.states = list(ipomdp_belief.ipomdp.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.ipomdp_belief = ipomdp_belief
        self.action_shield = action_shield
        self.default_action = default_action
        self.stuck_count = 0
        self.error_count = 0

        # Invert shield: action -> set of state indices where action is safe
        self.inv_shield = {
            a: [self.state_to_idx[s] for s in self.states if a in pp_shield[s]]
            for a in self.actions
        }

        self.inv_shield_compliment = {
            a: [self.state_to_idx[s] for s in self.states if not a in pp_shield[s]]
            for a in self.actions
        }

    
    def get_action_probs(self) -> "[Action]":
        '''
        Computes allowed/disallowed probailities from ipomdp_belief
        Pure (no effects on ipomdp_belief)
        '''
        action_shielded_prob = []
        for action in self.actions:
            allowed_states = self.inv_shield[action]
            disallowed_states = self.inv_shield_compliment[action]
            allowed_prob = self.ipomdp_belief.minimum_allowed_probability(allowed_states)
            disallowed_prob = self.ipomdp_belief.maximum_disallowed_probability(disallowed_states)
            action_shielded_prob.append((action, allowed_prob, disallowed_prob))
        return action_shielded_prob
    
    def next_actions(self, evidence: Tuple[Any, Any]):
        """
        Update belief and return allowed actions.

        Parameters
        ----------
        evidence : tuple
            (observation, action) pair

        Returns
        -------
        list
            List of actions that meet the safety threshold.
            Returns empty list (or default action) if propagation fails
            due to numerical errors.
        """
        obs, action = evidence
        propagation_success = self.ipomdp_belief.propagate(action, obs)

        if not propagation_success:
            # Numerical error in propagation - treat as stuck
            self.error_count += 1
            return [] if self.default_action is None else [self.default_action]

        action_shielded_prob = self.get_action_probs()

        # if the constr sum_b(s)=1 is maintained, the two checks are equivalent
        allowed_actions = [
            action for action, ap, dp in action_shielded_prob
            if ap >= self.action_shield or dp <= 1-self.action_shield
        ]

        if not allowed_actions:
            self.stuck_count += 1
            allowed_actions = [] if self.default_action is None else [self.default_action]

        return allowed_actions

    def restart(self):
        """Reset shield state."""
        self.stuck_count = 0
        self.error_count = 0
        self.ipomdp_belief.restart()

    def initialize(self, initial_state):
        self.restart() # doesn't use initial state
