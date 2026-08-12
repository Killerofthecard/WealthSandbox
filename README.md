# WealthSandBox — Career & Life Simulation Sandbox

一个研究 LLM agent 长期经济决策行为的沙盒环境。agent 从无业青年出发，在真实的宏观经济周期（FRED 历史数据驱动）和有限寿命约束下，通过职业选择、技能投资、银行理财、借贷等决策，最大化最终净资产。

---

## 快速开始

```bash
# Mock agent（确定性，无需 API key）
python run_wealthsandbox.py --agent mock --months 12

# Random baseline
python run_wealthsandbox.py --agent random --months 24 --seed 42

# LLM agent（需要 OPENAI_API_KEY）
python run_wealthsandbox.py --agent llm --months 12

# 指定宏观周期
python run_wealthsandbox.py --agent llm --months 24 --cycle recession

# 完整参数
python run_wealthsandbox.py --agent llm --months 480 --seed 42 \
    --start-age 20 --end-age 60 --cycle random \
    --save trajectories/run_001.json
```

环境变量（`.env` 或 shell）：

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_MODEL=deepseek-v4-pro
```

---

## 核心概念

### 时间

- 1 步 = 1 个月，12 步 = 1 年
- 默认 20 岁开局，60 岁强制退休终止
- 一局最多 `(end_age - start_age) × 12` 步 = 480 个月

### 状态变量（AgentState）

| 变量 | 范围 | 说明 |
|---|---|---|
| `cash` | 0 ~ ∞ | 流动资产，**现金+储蓄 ≤ $0 立刻破产** |
| `savings` | 0 ~ ∞ | 银行存款，按 FEDFUNDS / 12 生息，不自动花 |
| `loan_balance` | 0 ~ ∞ | 银行贷款，利息 (FEDFUNDS+2%)/月，每月自动还 2%（最低$50） |
| `general_skill` | 1 ~ 10 | 通用能力，决定可进入的职业，换职业时部分保留 |
| `occupation_skills` | per-job 1 ~ 10 | 各职业的专项经验，控制 tier 晋升，换职业清零 |
| `health` | 0.0 ~ 1.0 | 随年龄加速衰减，低于职业阈值被强制辞职，≤ 0 死亡 |
| `energy` | 0.0 ~ 1.0 | 短期耐力，upskill/intensive_work 消耗 40%，培训期每月 -15%，休息期每月 +10% |
| `tenure_months` | 0 ~ ∞ | 当前职业在职月数，与 occ_skill 共同决定 tier 晋升 |
| `current_tier` | 0 ~ N | 当前职业 ladder 索引（Junior → Mid → Senior → ...） |

### 二维技能系统

```
general_skill (可转移)         occupation_skills[occ_id] (职业绑定)
     │                              │
     决定收入 + 职业准入门槛          决定 tier 晋升 + 薪资乘数
     │                              │
     换职业: 同行业 ≈80%保留          换职业: 清零
     跨行业: ≈20%保留               失业再就业: 保留原值
     upskill 投资提升                intensive_work 加速提升
                                    tenure 每12月自动+1
```

### 职业（7 种，附 tier ladder）

| ID | 行业 | 基础月薪 | Tier 晋升链（×薪资乘数） | Gen≥ | Health≥ | 入职成本 | 培训期 |
|---|---|---|---|---|---|---|---|
| `manufacturing_worker` | 制造业 | $3,800 | Apprentice×1.0 → Skilled×1.30 → Supervisor×1.70 → Manager×2.20 | 1 | 0.6 | $0 | 0 |
| `civil_servant` | 政府 | $3,800 | Jr Officer×1.0 → Sr Officer×1.24 → Director×1.58 | 2 | 0.3 | $3,000 | 2月 |
| `investment_banker` | 金融 | $6,500 | Analyst×1.0 → Assoc×1.40 → VP×2.10 → MD×2.73 | 5 | 0.4 | $14,000 | 6月 |
| `financial_analyst` | 金融 | $6,500 | Junior×1.0 → Mid×1.31 → Senior×1.70 → Lead×2.22 | 3 | 0.3 | $7,000 | 3月 |
| `nurse` | 医疗 | $6,900 | Staff×1.0 → Senior×1.20 → Head×1.81 | 2 | 0.5 | $6,000 | 3月 |
| `data_scientist` | 科技 | $7,100 | Junior×1.0 → Mid×1.40 → Senior×1.85 → Principal×2.33 | 4 | 0.3 | $12,000 | 4月 |
| `software_engineer` | 科技 | $8,800 | Junior×1.0 → Mid×1.35 → Senior×1.74 → Principal×2.13 | 4 | 0.3 | $12,000 | 4月 |

> 校准锚：**BLS OEWS May 2025 National** — `base_monthly = P25 ÷ 12`，`tier_multiplier(k) = P(k) ÷ P25`。
> manufacturing_worker（SOC 51-0000）和 civil_servant（SOC 43-0000 proxy）为粗估，置信度较低。

**manufacturing_worker 是零门槛安全网**：永远免费入职，无需培训，即使 $0 现金也可以。

### 收入公式

```
gross = base_monthly × (1 + skill_sensitivity × (general_skill − 3)) × tier_multiplier
after_tax = gross × (1 − 0.15)
```

- 参考技能等级为 3：gen_skill=3 时拿 base_monthly × tier_multiplier
- 统一 15% 所得税

### 行业亲密度矩阵（换职业时技能转移率）

| 从→到 | tech | finance | healthcare | manufacturing | gov |
|---|---|---|---|---|---|
| **tech** | 100% | 50% | 20% | 10% | 10% |
| **finance** | 50% | 100% | 20% | 10% | 20% |
| **healthcare** | 20% | 20% | 100% | 20% | 30% |
| **manufacturing** | 10% | 10% | 20% | 100% | 20% |
| **gov** | 10% | 20% | 30% | 20% | 100% |

保留率 = `max(20%, 亲密度 × 80%)`。首次就业（从失业状态）保持 100%。

---

## 宏观经济系统

MacroLayer 从 FRED 历史数据驱动经济周期。`raw_data/` 下有三个周期的真实月度数据：

| 周期 | 目录 | 示例时期 | 特征 |
|---|---|---|---|
| boom | `raw_data/boom/` | 1997-2000, 2016-2019 | 低失业率，经济扩张 |
| normal | `raw_data/normal/` | 1983-1989, 2010-2015 | 正常波动 |
| recession | `raw_data/recession/` | 2008-2009, 2020 | 高失业率，NBER 衰退 |

每个 CSV 包含列：`year, month, UNRATE, USREC, FEDFUNDS`

| 指标 | 来源 | 影响 |
|---|---|---|
| UNRATE（失业率） | FRED | 行业收入乘数、裁员概率 |
| USREC（衰退标志） | FRED/NBER | 0=正常, 1=衰退，影响裁员和再就业 |
| FEDFUNDS（联邦基金利率） | FRED | 储蓄利率、贷款利率 |

**agent 不直接看到任何宏观数字。** 宏观指标只驱动底层机制，agent 通过具体状态变化感受经济——被裁员、收入下降、生活成本变化。Observation 中只展示定性的 `economy_status`（HEALTHY / SLUGGISH / WEAK / RECESSION）。

---

## 八种可用工具

agent 不调工具 = 自动工作赚钱。工具用于**主动改变现状**。每月可提交**多个工具**（原子 bundle，按序执行）。

### 职业类

| 工具 | 效果 | 成本 |
|---|---|---|
| `switch_occupation(id)` | 换职业，进入培训期（继续原工作），培训结束后转职 | $2,000 base + 职业 entry_cost，制造业永远免费 |
| `upskill` | general_skill +1（上限 10） | $5,000 + 40% energy，6 个月（继续工作） |
| `intensive_work` | 当前职业 occ_skill +1（上限 10） | 40% energy，3 个月（继续工作），无现金成本 |
| `quit_job` | 立即辞职，收入归零 | 免费 |

### 银行类

| 工具 | 效果 | 限制 |
|---|---|---|
| `deposit(amount)` | 现金 → 储蓄，按月生息 | 保留 ≥$2,000 现金 |
| `withdraw(amount)` | 储蓄 → 现金 | amount ≤ 储蓄余额 |
| `borrow(amount)` | 银行贷款 | 就职限额 = 12×月入，无职限额 = $8,000 |
| `repay(amount)` | 提前还贷 | amount ≤ 现金，amount ≤ 贷款余额 |

---

## 守卫系统（ActionValidator）

所有合法性判断集中在 `ActionValidator`，守卫链 fail-fast 执行。每个 guard 是纯函数：

```python
def guard_xxx(state: AgentState, career: CareerSystem) -> GuardResult:
    if condition_not_met:
        return GuardResult.reject(event_key, human_readable_message)
    return GuardResult.ok()
```

### 守卫注册表

| CareerMove | Guards |
|---|---|
| `SWITCH_OCCUPATION` | 不在培训中 → 目标有效 → gen_skill ≥ 门槛 → health ≥ 门槛 → cash ≥ 总成本 + buffer（制造业跳过） |
| `UPSKILL` | gen_skill < 10 → 未在进行中 → energy ≥ 40% → cash ≥ $7,000 |
| `INTENSIVE_WORK` | 必须就职 → occ_skill < 10 → 未在进行中 → energy ≥ 40% |
| `QUIT_JOB` | 必须就职 |
| `DEPOSIT` | cash ≥ $2,000 → amount > 0 → amount ≤ cash → cash − amount ≥ $2,000 |
| `WITHDRAW` | savings > 0 → amount > 0 → amount ≤ savings |
| `BORROW` | loan < limit → amount > 0 → loan + amount ≤ limit |
| `REPAY` | loan > 0 → cash > 0 → amount > 0 → amount ≤ cash → amount ≤ loan |

### 动态工具过滤

每回合只暴露当前合法的工具给 LLM——agent 不可能 call 出不允许的 action。Observation 文本中显式展示：

```
## Available Actions (you CAN call these)
  switch_occupation
  deposit
  borrow

## Unavailable
  upskill — Insufficient cash to upskill...
  quit_job — You are not currently employed.
  intensive_work — Must be employed to do intensive work.
```

---

## 月度循环详解（env.step）

三阶段执行，被拒绝立即返回，不推进月份：

```
Phase 1 — VALIDATE + EXECUTE
  ├─ 对 deepcopy 状态逐 action 验证（_validate_action 按 CareerMove 分派）
  ├─ 任一被拒 → 原状态返回 obs + rejection 信息，月份不推进
  └─ 全部通过 → 在真实状态上执行所有 action

Phase 2 — TICK（每月仅执行一次）
  ├─ CareerSystem.tick:   自动发工资、layoff 判定、timer 推进
  ├─ EnergySystem.tick:   培训能量消耗 / 休息能量恢复
  └─ HealthSystem.tick:   年龄加速健康衰减

Phase 3 — FINALISE
  ├─ LivingExpenseSystem: 扣除月生活费（现金不足自动从储蓄扣）
  ├─ BankSystem.finalize: 储蓄利息、贷款利息、贷款自动还款
  ├─ macro.step():        推进月份，加载下月宏观数据
  ├─ AgingSystem:         年龄递增
  ├─ check_dead():        破产/死亡/超龄判定
  └─ cash = max(0, cash)
```

关键设计：
- **工作自动** — 不需每个月决定上班，只有改变现状才调工具
- **拒绝不推进月份** — 被拒当场反馈，agent 可立即重试
- **Tick 在 Execute 之后** — agent 的下一个 observation 反映最新 tick 结果（包括刚发生的裁员）
- **原子 bundle** — 多工具在同月内按序执行，产生干净的 (s, CoT, [actions], s') 训练数据

---

## Agent 决策流程

```
Observation (自然语言)
    │
    ▼
LLMAgent.decide()
    │
    ├─ _format_observation(obs)
    │   ├─ Your Status     — age, cash, health, energy, skills, job
    │   ├─ Available Actions — 当前合法工具列表（从 Validator 实时计算）
    │   ├─ Unavailable     — 不可用工具及原因
    │   ├─ Economy Status  — HEALTHY/SLUGGISH/WEAK/RECESSION（定性，无数值）
    │   ├─ Available Occupations — 职业详情（仅失业时显示）
    │   ├─ Recent Events   — 上月事件列表
    │   └─ Narrative       — 自然语言摘要
    │
    ├─ messages.append(user_msg)
    │
    ├─ _filter_legal_tools(available_actions)
    │   只保留当前合法的 tool definitions → 传给 API
    │
    └─ _call_llm(tools=legal_tools)
        单次 API 调用，tool_choice="auto"
        返回 (reasoning, tool_calls)
```

### 对话历史

```
messages = [
    {role: "system",    content: "<游戏规则 prompt>"},
    {role: "user",      content: "<第1月 observation>"},
    {role: "assistant", content: "<第1月推理>"},
    {role: "user",      content: "<第2月 observation>"},
    ...
]
```

- 工具调用不存入对话史（OpenAI API 协议兼容性）
- System prompt 由 `_build_system_prompt()` 动态生成，所有数字从 `EnvConfig` 注入

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  run_wealthsandbox.py              ← CLI runner + 重试循环       │
│                                                                  │
│  ┌──────────────────────────┐    ┌───────────────────────────┐  │
│  │  WealthSandBoxEnv         │    │  LLMAgent                  │  │
│  │                           │    │                            │  │
│  │  MacroLayer  ← FRED 数据  │    │  system_prompt ← 规则     │  │
│  │  MicroLayer  ← 状态容器   │    │  messages[]   ← 对话史    │  │
│  │  Systems[]   ← 6 子系统   │    │  tools[]      ← 动态过滤  │  │
│  │  Validator   ← 守卫链     │    │  client       ← API 调用  │  │
│  │  history[]   ← 轨迹存档   │    │                            │  │
│  └──────────────────────────┘    └───────────────────────────┘  │
│            ↑                               │                     │
│            │        Observation             │                     │
│            └───────────────────────────────┘                     │
│                            │                                     │
│                     Decision(tool_calls)                         │
│                            ↓                                     │
│                     env.step()                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 六大子系统（实现 `BaseSystem` 协议：tick / finalize / handle_action / check_dead）

| 系统 | 职责 |
|---|---|
| **CareerSystem** | 职业注册表、收入计算、转职/培训/upskill/intensive_work/辞职、tier 晋升、裁员判定 |
| **EnergySystem** | 能量消耗（upskill/intensive_work）、培训月衰减、休息恢复 |
| **LivingExpenseSystem** | 月生活费扣除，不足时自动从储蓄提取 |
| **BankSystem** | 存取款、借贷、还款、月利息结算、自动还款 |
| **HealthSystem** | 年龄加速健康衰减（20-29: 0.03%/月, 30-39: 0.1%, 40-49: 0.3%, 50+: 0.6%） |
| **AgingSystem** | 年龄递增 + 超龄终止判定 |

---

## 配置（EnvConfig）

单一配置源，所有参数从 `EnvConfig` 流向各子系统、prompt 和 tool definitions：

```python
EnvConfig(
    # 时间
    end_age=60, start_year=2024, start_month=1,

    # 职业成本
    monthly_living_expense=2000.0,
    upskill_cost=5000.0, upskill_months=6,
    max_skill_level=10,
    switch_occupation_base_cost=2000.0,

    # 健康（分年龄段）
    health_decline_20_29=0.0003, health_decline_30_39=0.001,
    health_decline_40_49=0.003,  health_decline_50_plus=0.006,

    # 能量
    energy_cost_per_upskill=0.4,
    energy_decline_per_training_month=0.15,
    energy_recovery_per_month=0.10,
    energy_threshold_for_upskill=0.4,

    # 职业内增长
    intensive_work_months=3,
    occ_skill_passive_months=12,

    # 宏观
    macro_data_dir="raw_data",
    macro_cycle="",      # "" = 随机, "boom", "normal", "recession"
    macro_cycle_file="", # 指定具体 CSV, e.g. "2008_2009.csv"
    layoff_base_rate=0.05,

    # Agent 初始条件
    profile=AgentProfile(age=20, initial_cash=10000.0, ...),
    seed=42,
)
```

---

## 轨迹文件格式

```json
{
  "run": {"timestamp": "...", "model": "deepseek-v4-pro", "seed": 42, "macro_cycle": "random", "macro_cycle_file": "2008_2009.csv"},
  "config": { ... EnvConfig 完整参数 ... },
  "profile": { "age": 20, "initial_cash": 10000.0, ... },
  "active_systems": ["CareerSystem", "EnergySystem", "LivingExpenseSystem", "BankSystem", "HealthSystem", "AgingSystem"],
  "initial_state": { ... 第一步前的状态 ... },
  "system_prompt": "<完整 prompt 文本>",
  "total_steps": 480,
  "trajectory": [
    {
      "step": 1,
      "decision": {
        "reasoning": "I am unemployed with gen_skill=1...",
        "tool_calls": [{"tool_name": "switch_occupation", "parameters": {"occupation_id": "manufacturing_worker"}}]
      },
      "state_after": {
        "month": 1, "year": 2024, "age": 20,
        "cash": 6036.20, "savings": 0.0, "loan_balance": 0.0,
        "occupation_id": "manufacturing_worker", "general_skill": 1, "occ_skill": 1,
        "tenure_months": 1, "current_tier": 0,
        "monthly_after_tax_income": 3036.20, "job_status": "employed",
        "health": 1.0, "energy": 0.7,
        "upskill_months_remaining": 5, "intensive_work_months_remaining": 0,
        "training_months_remaining": 0,
        "events": ["Switched to Manufacturing Worker...", "Earned $3,036...", "Paid $2,000..."],
        "macro": {"unrate": 0.05, "usrecm": 1, "fedfunds": 3.94, "cycle_label": "recession", "economy_status": "The economy is in RECESSION..."}
      }
    }
  ],
  "final_summary": {
    "months_played": 480, "total_reward": 1234567.89,
    "final_cash": 50000.0, "final_savings": 20000.0,
    "final_occupation_id": "software_engineer",
    "final_general_skill": 8, "final_occ_skill": 7,
    "final_health": 0.85, "final_job_status": "employed",
    "termination_reason": "age_limit", "age": 60
  }
}
```

---

## 项目结构

```
sandbox/
├── run_wealthsandbox.py              # CLI 入口 + Runner
├── README.md
├── raw_data/                         # FRED 历史宏观数据
│   ├── UNRATE.csv, USREC.csv, FEDFUNDS.csv
│   ├── boom/                         # 经济扩张周期
│   ├── normal/                       # 正常周期
│   └── recession/                    # 衰退周期
├── trajectories/                     # 轨迹输出
│
└── wealthsandbox/
    ├── __init__.py                   # 导出 WealthSandBoxEnv, EnvConfig
    ├── config.py                     # 常量 + EnvConfig dataclass
    ├── types.py                      # Action, Observation, AgentState, CareerMove, JobStatus, Tier
    ├── profile.py                    # AgentProfile（agent 初始条件）
    ├── macro_layer.py                # 日历 + FRED 宏观数据驱动
    ├── micro_layer.py                # AgentState 工厂
    ├── validator.py                  # ActionValidator + 全部 guard 函数
    ├── env.py                        # 主环境：三阶段 step/reset/observation
    │
    ├── systems/
    │   ├── base.py                   # BaseSystem 抽象类
    │   ├── career.py                 # CareerSystem（职业/收入/技能/tier/转职/upskill/裁员）
    │   ├── energy.py                 # EnergySystem（精力消耗与恢复）
    │   ├── living.py                 # LivingExpenseSystem（月生活费）
    │   ├── bank.py                   # BankSystem（存取款/借贷/利息）
    │   ├── health.py                 # HealthSystem（年龄加速健康衰减）
    │   └── aging.py                  # AgingSystem（年龄 + 超龄终止）
    │
    ├── agents/
    │   ├── tools.py                  # build_tools() + ToolCall, Decision dataclass
    │   └── llm_agent.py              # LLMAgent: prompt 生成 + API 调用 + 动态工具过滤
    │
    └── tests/
        ├── test_env.py               # 环境集成测试
        ├── test_career.py            # CareerSystem 单元测试
        ├── test_validator.py         # 守卫单元测试
        └── test_tools.py             # 工具解析测试
```

### 扩展点

- **新增职业**：在 `career.py` 的 `DEFAULT_OCCUPATIONS` 中添加 `Occupation(...)`
- **新增工具/动作**：在 `tools.py` 添加工具定义 → `types.py` 添加 `CareerMove` → `validator.py` 注册 guard → 对应 system 添加 `handle_action` 逻辑
- **新增子系统**：实现 `BaseSystem` 协议（tick / finalize / handle_action / check_dead）→ `env.systems.append(...)`
- **新增宏观指标**：在 `raw_data/` CSV 中添加列 → `macro_layer.py` 读取 → 对应 system 消费该指标
- **调整参数**：修改 `EnvConfig`，所有子系统、prompt、tool definitions 自动同步
