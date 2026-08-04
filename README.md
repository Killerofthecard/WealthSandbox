# WealthSandBox — Career Simulation

一个基于 LLM agent 的职业与经济决策沙盒。agent 从失业状态出发，在有限的寿命和资金约束下，通过换职业、提升技能、辞职等决策最大化最终净资产。

---

## 目录

- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [架构总览](#架构总览)
- [月度循环详解](#月度循环详解)
- [Agent 决策流程](#agent-决策流程)
- [三种可用工具](#三种可用工具)
- [守卫系统](#守卫系统)
- [LLM Agent 集成](#llm-agent-集成)
- [配置 & 数据流](#配置--数据流)
- [轨迹文件格式](#轨迹文件格式)
- [项目结构](#项目结构)

---

## 快速开始

```bash
# Mock agent（确定性，无需 API）
python run_wealthsandbox.py --agent mock --months 12

# Random baseline
python run_wealthsandbox.py --agent random --months 24 --seed 42

# LLM agent（需要 OPENAI_API_KEY）
python run_wealthsandbox.py --agent llm --months 12 --start-age 30 --end-age 31

# 自定义保存路径
python run_wealthsandbox.py --agent llm --months 12 --save results/traj.json
```

环境变量（`.env` 文件或 shell）：

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # 可选
DEFAULT_MODEL=gpt-4.1-mini                    # 可选
```

---

## 核心概念

### 时间

- 每 **1 步 = 1 个月**，12 步 = 1 年
- agent 起始年龄可配置（默认 20），到达 `end_age`（默认 60）时游戏结束
- 因此一局最多 `(end_age - start_age) × 12` 步

### 状态变量

| 变量 | 范围 | 说明 |
|---|---|---|
| `cash` | 0 ~ ∞ | 流动资产，**降到 0 立刻破产** |
| `health` | 0.0 ~ 1.0 | 每月自然衰减 0.001，到 0 死亡 |
| `energy` | 0.0 ~ 1.0 | 培训期间每月 -15%，非培训期间每月 +2% |
| `skill_level` | 1 ~ 10 | 通用能力，影响所有职业的薪资 |
| `occupation_id` | string | 当前职业，空字符串表示失业 |
| `job_status` | `employed` / `unemployed` | 就业状态 |

### 职业（Occupation）

7 种预设职业，各有不同的准入门槛：

| ID | 行业 | 基础月薪 | 技能敏感度 | 最低技能 | 入职费 | 培训期 |
|---|---|---|---|---|---|---|
| `manufacturing_worker` | manufacturing | $3,800 | 3% | 1 | $0 | 0 |
| `civil_servant` | gov | $4,500 | 3% | 2 | $1,000 | 2 月 |
| `nurse` | healthcare | $5,200 | 4% | 2 | $4,000 | 3 月 |
| `financial_analyst` | finance | $5,500 | 5% | 3 | $5,000 | 3 月 |
| `software_engineer` | tech | $6,500 | 6% | 4 | $8,000 | 4 月 |
| `data_scientist` | tech | $6,800 | 6% | 4 | $8,000 | 4 月 |
| `investment_banker` | finance | $7,500 | 5% | 5 | $12,000 | 6 月 |

### 薪资公式

```
gross = base_monthly × (1 + skill_sensitivity × (skill_level − 3))
after_tax = gross × (1 − 0.15)
```

- 参考技能等级为 3：技能 = 3 时拿基础月薪
- 技能每高 1 级，薪资提升 `skill_sensitivity × 100%`
- 技能每低 1 级，薪资下降相同比例
- 统一 15% 所得税

### 技能转移

换职业时，技能按行业亲密度折算后保留：

| | tech | finance | healthcare | manufacturing | gov |
|---|---|---|---|---|---|
| **tech** | 100% | 50% | 20% | 10% | 10% |
| **finance** | 50% | 100% | 20% | 10% | 20% |
| **healthcare** | 20% | 20% | 100% | 20% | 30% |
| **manufacturing** | 10% | 10% | 20% | 100% | 20% |
| **gov** | 10% | 20% | 30% | 20% | 100% |

保留率 = `max(20%, 亲密度 × 80%)`，从失业状态找到第一份工作保持 100%。

---

## 架构总览

```
┌──────────────────────────────────────────────────────────┐
│  run_wealthsandbox.py          ← CLI runner + 重试循环    │
│                                                          │
│  ┌─────────────────────┐   ┌──────────────────────────┐  │
│  │  WealthSandBoxEnv    │   │  LLMAgent                │  │
│  │                     │   │                          │  │
│  │  MacroLayer ── 时间  │   │  system_prompt ── 规则   │  │
│  │  MicroLayer ── 状态  │   │  messages[]   ── 对话史  │  │
│  │  CareerSystem ─ 逻辑 │   │  tools[]      ── 工具定义│  │
│  │  Validator   ── 校验 │   │  client       ── API调用 │  │
│  │  history[]   ── 存档 │   │                          │  │
│  └─────────────────────┘   └──────────────────────────┘  │
│            ↑                         │                    │
│            │      Observation        │                    │
│            └─────────────────────────┘                    │
│                          │                               │
│                   Decision(tool_calls)                    │
│                          ↓                               │
│                   env.step()                              │
└──────────────────────────────────────────────────────────┘
```

### 各层职责

| 模块 | 职责 |
|---|---|
| **EnvConfig** | 所有超参数的单一声源（年龄、成本、税率……） |
| **MacroLayer** | 日历推进：年/月/总月数 |
| **MicroLayer** | `AgentState` 的容器和工厂，提供 `snapshot()` |
| **CareerSystem** | 职业注册表、收入计算、转职/培训/upskill/辞职逻辑，实现 `BaseSystem` 协议 |
| **ActionValidator** | 守卫层：每步动作在入队前经过纯函数校验，返回 `GuardResult` |
| **LLMAgent** | OpenAI function-calling 客户端：自然语言观察 → 推理 → 工具调用 |
| **Runner** | 连接 env 和 agent，处理重试、轨迹保存、终端渲染 |

---

## 月度循环详解

`env.step()` 分为三个阶段：

```
┌── Phase 1: TICK（每月仅执行一次）─────────────────────┐
│                                                        │
│  _tick_done 守卫确保重试不重复跑                         │
│                                                        │
│  CareerSystem.tick(state, macro):                      │
│    ├─ 有工作? → 自动发工资（不用 agent 调工具）           │
│    ├─ upskill_months_remaining -= 1                    │
│    │    └─ 到 0 → skill_level += 1                     │
│    ├─ training_months_remaining -= 1                   │
│    │    └─ 到 0 → 正式换到新职业（技能转移生效）          │
│    └─ 培训中? → energy -= 15% : energy += 2%            │
│                                                        │
├── Phase 2: VALIDATE + EXECUTE ────────────────────────┤
│                                                        │
│  action == NONE → 跳过                                  │
│                                                        │
│  action ≠ NONE:                                        │
│    validator.validate(action, state)                    │
│    ├─ 被拒 → return (obs, reward=0, done=False,         │
│    │                 {action_rejected: True})           │
│    │         ⚠️ 月份不推进，agent 当场用同一 obs 重试    │
│    │                                                   │
│    └─ 通过 → CareerSystem.handle_action()               │
│              扣钱 / 开始培训 / 辞职 / 开始 upskill       │
│                                                        │
├── Phase 3: FINALIZE（仅 Phase 2 成功或 NONE）──────────│
│                                                        │
│  macro.step()                    → 月份 +1             │
│  cash -= monthly_living_expense                        │
│  health -= 0.001                                       │
│  每 12 月 age += 1                                     │
│  _check_termination():                                 │
│    cash ≤ 0  → bankruptcy                              │
│    age ≥ 60  → age_limit                               │
│    health ≤ 0 → death                                  │
│  cash = max(0, cash)             → 清零保护             │
│  存档 history[]                                         │
│  _make_observation()             → 返回给 agent         │
│  _tick_done = False              → 下月重新 tick        │
└────────────────────────────────────────────────────────┘
```

关键设计决策：

- **工作是自动的** — agent 不需要每个月"决定上班"，只有在想改变现状时才调工具
- **拒绝不推进月份** — 被拒的动作当场反馈，agent 可以立即更换策略重试（最多 5 次）
- **破产检查在清零之前** — 先判断 `cash ≤ 0` 决定破产，再执行 `max(0, cash)` 美化显示
- **生活费守卫** — 动作开销必须 `cash ≥ cost + living_expense`，防止换完职业当场破产

---

## Agent 决策流程

```
Observation (自然语言)
    │
    ▼
LLMAgent.decide()
    │
    ├─ _format_observation(obs)     ← 转成结构化的纯文本
    │   ├─ 状���（age, cash, skill, health, energy, job）
    │   ├─ Legal Actions ✓/✗       ← 从 Validator 实时计算
    │   ├─ 宏观薪资水平
    │   ├─ 失业时显示完整职业表
    │   ├─ 上月事件列表
    │   └─ ⚠️ 被拒红字警告（如有）
    │
    ├─ messages.append(user_msg)
    │
    ├─ _call_llm()                 ← 单次 API 调用
    │   model="gpt-4.1-mini"
    │   tool_choice="auto"          ← 模型自己决定调不调工具
    │   temperature=0.7
    │   max_tokens=512
    │
    ├─ 矛盾检测                     ← 安全网
    │   "不调工具" + 却调了 → 丢弃工具调用
    │
    └─ return Decision(reasoning, tool_calls)
```

### 对话历史结构

```
messages = [
    {role: "system",    content: "<完整游戏规则 prompt>"},     ← 仅第 0 条
    {role: "user",      content: "<第 1 月观察>"},
    {role: "assistant", content: "<第 1 月推理>"},            ← 不含工具
    {role: "user",      content: "<第 2 月观察>"},
    {role: "assistant", content: "<第 2 月推理>"},
    ...
]
```

- 工具调用不在对话史中（API 协议兼容性）
- 全量保存，不做裁剪或摘要（当前 12 月游戏仅占 ~6,000 tokens / 128k）

---

## 三种可用工具

agent 不调工具 = 继续工作自动赚钱，这是最常见的默认选择。以下三种工具用于**主动改变现状**：

### 1. `switch_occupation(occupation_id)`

```
成本: switch_base_cost ($2,000) + 目标职业的 entry_cost
效果: 立即支付成本
      ├─ 目标职业 training_months > 0 → 进入培训期（留在当前岗位继续工作）
      │   培训结束自动转职 + 技能转移
      └─ training_months == 0 → 立即转职 + 技能转移
```

拒绝条件：

| guard | 原因 |
|---|---|
| 技能不足 | `skill_level < 职业最低要求` |
| 已在培训 | `training_months_remaining > 0` |
| 资金不足 | `cash < 总成本 + living_expense`（预留生活费） |
| 目标无效 | 职业 id 不存在 |

### 2. `upskill()`

```
成本: $5,000
耗时: 6 个月
效果: 6 个月后 skill_level += 1（上限 10）
期间: 继续正常工作赚工资
```

拒绝条件：

| guard | 原因 |
|---|---|
| 已达上限 | `skill_level >= 10` |
| 正在进行 | `upskill_months_remaining > 0` |
| 资金不足 | `cash < $7,000`（$5k 成本 + $2k 生活费） |

### 3. `quit_job()`

```
成本: 免费
效果: 立即变为 unemployed，职业清空，收入归零
注意: 生活费继续扣，需有足够储蓄
```

拒绝条件：

| guard | 原因 |
|---|---|
| 本来就没工作 | `job_status != employed` |

---

## 守卫系统

所有合法性判断集中在 `ActionValidator` 中。每个 guard 是纯函数：

```python
def guard_xxx(state: AgentState, career: CareerSystem) -> GuardResult:
    if 条件不满足:
        return GuardResult.reject(event_key, human_readable_message)
    return GuardResult.ok()
```

注册在一处，一目了然：

```python
class ActionValidator:
    def __init__(self, career):
        self.register(CareerMove.SWITCH_OCCUPATION, guard_switch_occupation)
        self.register(CareerMove.UPSKILL, guard_upskill)
        self.register(CareerMove.QUIT_JOB, guard_quit_job)
```

### 守卫链执行

```
validate(action, state)
    → 按注册顺序逐个运行 guard
    → 第一个 reject 立即返回（fail-fast）
    → 全部通过 → OK
```

### 实时可用动作

`available_actions(state)` 遍历所有已注册 guard，返回 `{action_name: bool}`。env 在构建 observation 时用此结果向 agent 展示 `✓` / `✗`，agent 可以据此避免无效调用。

---

## LLM Agent 集成

### System Prompt（动态生成）

prompt 由 `_build_system_prompt()` 函数动态生成，所有数字从 `EnvConfig` 注入：

```python
_build_system_prompt(
    start_age=30, end_age=31,        # → "you have only 12 months (1 years)"
    initial_cash=5000.0,
    living_expense=2000.0,           # → "living expense is $2,000/month"
    upskill_cost=5000.0,             # → "upskill costs $5,000"
    ...
)
```

prompt 核心内容：

- **目标声明** — 最大化最终净资产，RULE #1: 不要破产
- **时间约束** — 明确告诉 agent 还剩多少个月
- **工作自动化** — 强调不需要调 tool 来工作
- **完整职业表** — 7 种职业的薪资/门槛/成本/培训期
- **约束清单** — 6 种会导致拒绝的情况
- **推理一致性要求** — "如果推理说不调工具，就别调"

### Tool Definitions（动态生成）

工具描述中的金额同样从配置注入：

```python
build_tools(
    living_expense=2000.0,   # → "Living expenses ($2,000/month) still apply"
    upskill_cost=5000.0,     # → "Costs $5,000 and takes 6 months"
    switch_base_cost=2000.0, # → "Base switch cost is $2,000 plus..."
    occupations={...},       # → 职业列表中的数字全部动态
)
```

### 矛盾检测（安全网）

小模型偶尔会写出"我不会调工具"但同时又产生工具调用。`_reasoning_says_no_tool()` 是最后一道防线：

```python
# 检测 "call no tool" / "do nothing" / "continue working" 等短语
# 若同时存在 "I will switch" / "calling the..." 等覆盖短语 → 放行
# 否则 → 丢弃工具调用，让 agent 自动工作
```

---

## 配置 & 数据流

所有成本参数只有一个来源：`EnvConfig`。

```
EnvConfig
    │
    ├─→ CareerSystem(upskill_cost, switch_base_cost, living_expense, ...)
    │     ├─→ Validator 守卫校验（cash ≥ cost + living_expense）
    │     ├─→ env.step() 执行扣费
    │     └─→ _make_observation() 计算 available_actions
    │
    ├─→ _build_system_prompt(...)   → 规则 + 数字注入 prompt
    ├─→ build_tools(...)             → 工具描述中的金额
    └─→ build_agent()                → 串联上述所有
```

**没有任何地方硬编码 `$5,000` 或 `$2,000`** — 除了 `config.py` 中的默认常量，它们只作为 `EnvConfig` 的默认值使用，修改 `EnvConfig(...)` 即可全局生效。

---

## 轨迹文件格式

运行结束后生成 JSON 轨迹文件：

```json
{
  "config": {
    "start_age": 30, "end_age": 31, "seed": 42,
    "initial_cash": 5000.0,
    "monthly_living_expense": 2000.0,
    "upskill_cost": 5000.0,
    "upskill_months": 6,
    "max_skill_level": 10,
    "switch_occupation_base_cost": 2000.0
  },
  "initial_state": {
    "age": 30, "health": 1.0, "energy": 1.0, "cash": 5000.0,
    "occupation_id": "", "skill_level": 1, "job_status": "unemployed",
    "upskill_months_remaining": 0, "training_months_remaining": 0,
    "year": 2024, "month": 1
  },
  "system_prompt": "<完整 prompt 文本>",
  "total_steps": 12,
  "trajectory": [
    {
      "step": 1,
      "decision": {
        "reasoning": "I am unemployed...",
        "tool_calls": [
          {"tool_name": "switch_occupation", "parameters": {"occupation_id": "manufacturing_worker"}}
        ]
      },
      "state_after": {
        "month": 1, "year": 2024, "age": 30,
        "cash": 1000.0, "occupation_id": "manufacturing_worker",
        "skill_level": 1, "monthly_after_tax_income": 0.0,
        "health": 0.999, "energy": 1.0, "job_status": "employed",
        "training_months_remaining": 0,
        "events": ["no_occupation_no_income", "switched_to_manufacturing_worker_cost_2000", "living_expense_2000"]
      }
    }
    // ... 逐月记录
  ],
  "final_summary": {
    "months_played": 12, "total_reward": 36382.4,
    "final_cash": 7398.2, "final_occupation_id": "manufacturing_worker",
    "final_skill_level": 2, "final_health": 0.988,
    "final_job_status": "employed", "termination_reason": "max_steps_reached",
    "age": 31
  }
}
```

每个字段说明：

| 字段 | 说明 |
|---|---|
| `config` | 完整超参数，可复现运行 |
| `initial_state` | `env.reset()` 后、第一步前的状态 |
| `system_prompt` | LLM agent 收到的完整规则（非 LLM agent 不保存） |
| `trajectory[].decision` | agent 的推理文本 + 工具调用 |
| `trajectory[].state_after` | 该月结束后的状态快照 + 事件列表 |
| `final_summary` | 终局总结：资产、职业、技能、健康、奖励、终止原因 |

---

## 项目结构

```
sandbox/
├── run_wealthsandbox.py                # CLI 入口 + runner
├── README.md                           # 本文件
├── trajectories/                       # 轨迹输出
│   └── llm_YYYYMMDD_HHMMSS.json
│
└── wealthsandbox/
    ├── __init__.py                     # 导出 WealthSandBoxEnv, EnvConfig
    ├── config.py                       # 常量 + EnvConfig dataclass
    ├── types.py                        # Action, Observation, AgentState, CareerMove
    ├── macro_layer.py                  # 日历（年/月/总月数）
    ├── micro_layer.py                  # AgentState 容器
    ├── validator.py                    # 守卫层（纯函数校验）
    ├── env.py                          # 主环境：step/reset/observation
    │
    ├── systems/
    │   ├── __init__.py
    │   ├── base.py                     # BaseSystem 抽象类
    │   └── career.py                   # CareerSystem（职业/收入/转职/培训/upskill）
    │
    ├── agents/
    │   ├── __init__.py                 # 导出 LLMAgent, Decision, ToolCall
    │   ├── tools.py                    # 工具定义 + build_tools()
    │   └── llm_agent.py               # LLM Agent（prompt + tool calling）
    │
    └── tests/
        ├── __init__.py
        ├── test_env.py                 # 环境集成测试
        ├── test_career.py             # CareerSystem 单元测试
        ├── test_validator.py          # 守卫单元测试
        └── test_tools.py              # 工具解析测试
```

### 扩展点

- **新增职业**：在 `career.py` 的 `DEFAULT_OCCUPATIONS` 中添加 `Occupation(...)`
- **新增工具/动作**：在 `tools.py` 添加工具定义 + `types.py` 添加 `CareerMove` + `validator.py` 注册 guard + `career.py` 添加 `process_xxx()`
- **新增子系统**（如消费、家庭、投资）：实现 `BaseSystem` 协议 → `env.systems.append(...)`
- **调整难度**：修改 `EnvConfig` 参数即可（生活费、成本、税率、年龄跨度）
