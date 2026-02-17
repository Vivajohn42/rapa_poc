from state.schema import ZA
from kernel.interfaces import StreamA


class AgentA(StreamA):
    def infer_zA(self, obs) -> ZA:
        return ZA(
            width=obs.width,
            height=obs.height,
            agent_pos=obs.agent_pos,
            goal_pos=obs.goal_pos,
            obstacles=list(obs.obstacles),
            hint=obs.hint
        )
