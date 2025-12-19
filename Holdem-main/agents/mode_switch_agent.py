# agents/mode_switch_agent.py
from agents.rl_only_agent import RlOnlyAgent, RLOpt
from risk.risk_manager import filter_actions
from controller.mode_switch import decide_mode

class ModeSwitchAgent:
    def __init__(self, model_path, start_bankroll=100, seed=0):
        self.start_bankroll = start_bankroll
        self.bankroll = start_bankroll
        self.peak = start_bankroll

        self.rl = RlOnlyAgent(RLOpt(
            model_path=model_path,
            explore=False,
            seed=seed
        ))

    def act(self, state, legal_actions):
        # drawdown 计算
        self.peak = max(self.peak, self.bankroll)
        drawdown = self.peak - self.bankroll

        # 模式决策
        mode = decide_mode(
            bankroll=self.bankroll,
            start_bankroll=self.start_bankroll,
            drawdown=drawdown,
            state=state
        )

        # 风控过滤
        filtered = filter_actions(
            legal_actions=legal_actions,
            state=state,
            mode=mode
        )

        # RL 选动作（在过滤后的动作空间）
        action = self.rl.act(state, filtered)
        return action

    def observe_reward(self, reward):
        # 在 evaluate/self-play 里调用
        self.bankroll += reward
