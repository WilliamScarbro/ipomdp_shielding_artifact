" Evaluation Runner - generic method for running experiments "

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..Models import IPOMDP
from .runtime_shield import RuntimeImpShield

if TYPE_CHECKING:
    from .script_library import RunScript

@dataclass
class ReportRunner:
    ipomdp : IPOMDP
    perception : "State -> Obs"
    rt_shield : RuntimeImpShield
    action_selector : "{Actions} -> Action"
    length : int
    initial : "State x Action"


    def run(self):

        self.initialize()

        self.rt_shield.initialize(self.initial)

        state, action = self.initial

        obs = self.perception(state)

        for step in range(self.length):
            # Stop if we've reached FAIL state
            if state == "FAIL":
                break

            # get next action
            actions = self.rt_shield.next_actions((obs, action))
            action = self.action_selector(actions)

            # evaluator sees cur state, obs, and chosen action
            self.report(step, state, obs, action, self.rt_shield)

            # update state with action
            state = self.ipomdp.evolve(state, action)

            # update obs with new state
            obs = self.perception(state)

        return self.final(self.rt_shield)
    
    def initialize(self):
        pass
    
    def report(self, step, state, obs, action, rt_shield):
        pass


    def final(self, rt_shield):
        pass


@dataclass
class ScriptedReportRunner:
    """Report runner that replays a pre-recorded script.

    Unlike ReportRunner which generates states/observations at runtime,
    this replays a fixed sequence allowing fair comparison between
    different templates/propagators on identical trajectories.
    """
    script: "RunScript"
    rt_shield: RuntimeImpShield

    def run(self):
        """Run the evaluation using the scripted trajectory."""
        self.initialize()

        self.rt_shield.initialize(self.script.initial)

        for step, (state, obs, scripted_action) in enumerate(self.script.steps):
            if state == "FAIL":
                break

            # Get shield's allowed actions given (obs, previous_action)
            prev_action = self.script.initial[1] if step == 0 else self.script.steps[step - 1][2]
            allowed_actions = self.rt_shield.next_actions((obs, prev_action))

            # Report uses the scripted action (from perfect perception shield)
            # but we track what the runtime shield would have allowed
            self.report(step, state, obs, scripted_action, allowed_actions, self.rt_shield)

        return self.final(self.rt_shield)

    def initialize(self):
        pass

    def report(self, step, state, obs, scripted_action, allowed_actions, rt_shield):
        """Report on a single step.

        Parameters
        ----------
        step : int
            Step number
        state : any
            True state (from script)
        obs : any
            Observation (from script)
        scripted_action : any
            Action chosen by perfect perception shield (from script)
        allowed_actions : list
            Actions allowed by the runtime shield
        rt_shield : RuntimeImpShield
            The runtime shield instance
        """
        pass

    def final(self, rt_shield):
        pass

