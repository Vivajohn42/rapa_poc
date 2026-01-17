from state.schema import ZA

class AgentA:
    def infer_zA(self, obs) -> ZA:
        return ZA(
            width=obs.width,
            height=obs.height,
            agent_pos=obs.agent_pos,
            goal_pos=obs.goal_pos,
            obstacles=list(obs.obstacles),
            hint=obs.hint
        )
