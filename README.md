# Holdem
poker_ai/  
├─ README.md  
├─ requirements.txt  
├─ run.py                      # 一键入口：训练/评估/对战（命令行参数）  
│  
├─ configs/  
│  ├─ default.yaml             # 盲注、筹码、动作离散、阈值、训练轮数等  
│  ├─ ablation_no_risk.yaml  
│  ├─ ablation_no_mode.yaml  
│  └─ ablation_rl_only.yaml  
│  
├─ env/                        # 游戏环境（地基）   
│  ├─ __init__.py    
│  ├─ poker_env.py             # 核心：step/reset/legal_actions/reward/terminal    
│  ├─ rules.py                 # 盲注、下注规则、行动顺序、raise 合法性  
│  ├─ deck.py                  # Card/Suit/Rank/Deck  
│  ├─ hand_eval.py             # 手牌评估、胜率/牌力           
│  └─ state.py                 # GameState 数据结构（dataclass）  
│  
├─ features/                   # 感知层：把局面变成特征向量  
│  ├─ __init__.py  
│  ├─ extractor.py             # extract_features(state, player_id) -> np.array  
│  ├─ equity.py                # Monte Carlo equity 近似（可缓存）  
│  └─ normalizer.py            # pot/stack/SPR 等归一化  
│  
├─ opponent/                   # 对手建模（贝叶斯/启发式都放这里）  
│  ├─ __init__.py  
│  ├─ profiler.py              # tight/loose/aggressive/passive 分类  
│  ├─ range_model.py           # 对手范围（初期可用简化分布）  
│  └─ history.py               # Event 记录与统计  
│  
├─ strategy/                   # 策略引擎（EV + RL + 组合逻辑）  
│  ├─ __init__.py  
│  ├─ ev_preflop.py            # 翻前 EV(fold/call/raise) 估计  
│  ├─ rl_flop_qlearn.py        # 翻牌 Q-learning（近似线性Q或tabular）  
│  ├─ policy_mix.py            # 把 EV/RL 输出合并成 action distribution  
│  └─ action_space.py          # fold/call/raise + raise sizes 离散定义  
│  
├─ risk/                       # 风控层（你的亮点）  
│  ├─ __init__.py  
│  ├─ risk_manager.py          # Kelly缩放、drawdown、stop-loss、过滤危险动作  
│  └─ bankroll.py              # bankroll、最大回撤、Sharpe 等统计  
│  
├─ controller/                 # 模式切换控制器（核心创新点）  
│  ├─ __init__.py  
│  └─ mode_switch.py           # decide_mode(...) & apply_mode_to_policy(...)  
│  
├─ agents/                     # 不同智能体封装（用于 baseline 和消融）  
│  ├─ __init__.py  
│  ├─ base_agent.py            # Agent 接口：act(obs)->action  
│  ├─ random_agent.py  
│  ├─ rule_tight_aggressive.py  
│  ├─ ev_only_agent.py  
│  ├─ rl_only_agent.py  
│  └─ mode_switch_agent.py     # 你的最终完整系统（调用所有模块）  
│  
├─ training/                   # 训练入口（自我对弈）  
│  ├─ __init__.py  
│  ├─ self_play.py             # 自我对弈生成数据/更新 Q  
│  ├─ replay.py                # （可选）缓存 transitions  
│  └─ schedules.py             # epsilon 衰减、学习率调度  
│  
├─ evaluation/                 # 评估入口（TA会看）  
│  ├─ __init__.py  
│  ├─ evaluate.py              # 跑N手 vs baselines，输出表格  
│  ├─ metrics.py               # winrate、BB/100、drawdown、Sharpe  
│  └─ ablation.py              # 自动跑消融：no_risk/no_mode/no_ev/...  
│  
├─ scripts/                    # 小工具  
│  ├─ quick_sanity.py          # 快速自检（合法动作/奖励/终局）  
│  ├─ seed_test.py             # 复现性检查  
│  └─ profile_opponents.py     # 输出对手分类统计  
│  
└─ outputs/  
   ├─ logs/  
   ├─ models/                  # q_table / weights 存档  
   └─ reports/                 # 自动导出的结果表格 csv  
