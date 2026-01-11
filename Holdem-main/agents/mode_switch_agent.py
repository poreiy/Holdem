# agents/mode_switch_agent.py

from agents.rl_only_agent import RlOnlyAgent, RLOpt
from risk.risk_manager import filter_actions
from controller.mode_switch import decide_mode
from controller.opponent_detection import OpponentDetector   # ★ 新增

class ModeSwitchAgent:
    def __init__(self, model_path, start_bankroll=100, seed=0, hero_id=0):
        self.start_bankroll = start_bankroll
        self.bankroll = start_bankroll
        self.peak = start_bankroll

        self.hero_id = hero_id
        self.detector = OpponentDetector(hero_id=hero_id)      # ★ 新增

        self.rl = RlOnlyAgent(RLOpt(
            model_path=model_path,
            explore=False,
            seed=seed
        ))

    def act(self, state, legal_actions):
        # === 1. 读取对手画像 ===
        opp_profile = self.detector.get_opponent_profile()
        opp_type = opp_profile.get("type", "UNKNOWN")

        # === 2. drawdown ===
        self.peak = max(self.peak, self.bankroll)
        drawdown = self.peak - self.bankroll

        # === 3. 模式决策（加入 opponent_type）===
        mode = decide_mode(
            bankroll=self.bankroll,
            start_bankroll=self.start_bankroll,
            drawdown=drawdown,
            state=state,
            opponent_type=opp_type,        # ★ 新增
        )

        # === 4. 风控过滤 ===
        filtered = filter_actions(
            legal_actions=legal_actions,
            state=state,
            mode=mode
        )

        # === 5. RL 选动作 ===
        action = self.rl.act(state, filtered)
        return action

    def observe_reward(self, reward):
        self.bankroll += reward

    # 在 evaluate/self-play 中每步调用
    def observe_opponent(self, state, action, actor_id):
        self.detector.observe(state, action, actor_id)
