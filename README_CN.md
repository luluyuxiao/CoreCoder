<div align="center">

# CoreCoder

**编程 agent 里的 nanoGPT。3.9k 行引擎、整包 9650 行纯 Python 全部一口气可读，读懂一个 coding agent 到底怎么运作，再 fork 出你自己的。**

*learn from it · fork it · ship something better*

中文 | [English](README.md) | [配套源码导读 · 八篇双语](article/)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/he-yufeng/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/CoreCoder/actions)
[![engine](https://img.shields.io/badge/engine-3918_LoC-blue)](article/)
[![源码导读](https://img.shields.io/badge/源码导读-8篇双语-orange)](article/)

</div>

- **读得完。** 一个下午读完整个引擎，没有一处藏着你看不懂的魔法。
- **改得动。** 每一行都能在你自己机器上下断点、改了再跑。它真能干活，所以这份参考是活的，不是示意图。
- **留白即起点。** 刻意只留最小核心，没做的那些不是半成品，是留给你 fork 出更好东西的地方。

## 和谁比

| | CoreCoder | Claude Code | aider | nanoGPT |
|---|---|---|---|---|
| 代码量 | 引擎约 3918 行 / 整包 9650 行 | 几十万行（闭源） | 数万行 Python | 约 600 行（两个文件） |
| 读完要多久 | 一个下午 | 读不了（闭源） | 得啃几天 | 一个下午 |
| 能不能下断点改了再跑 | 能，每一行 | 不能 | 能，但量大 | 能 |
| 定位 | 读懂并 fork 出你自己的 agent | 生产级编程助手 | 终端结对编程 | 教学用最小 GPT |

nanoGPT 那一列是拿来对照的：它最小、可读，但教的是训一个 GPT。CoreCoder 想干的是同一件事，只是把对象换成一个能真正改代码的 agent。和 Claude Code、aider 摆在一起，不是要跟它们抢用户，CoreCoder 是借它们来学、来起步的那块地基，根本不在一个赛道。

## 这是什么

我一直觉得 coding agent 被讲得太玄了。把 Claude Code、Cursor 这类工具扒到底，核心是一个 while 循环套着一个大模型，外加七八个让它能真正动手的工具。难的从来不是这个循环，而是循环跑进真实世界以后要兜的那些底。CoreCoder 就是把这个核心老老实实写出来的最小版本。

引擎部分（循环、模型接口、上下文、工具、会话）去掉空行和注释是 3918 行。把 Storage、Skills、Trace、Eval、最外层的 CLI、配置、打包一起算，整个包 38 个文件、物理 9650 行、净 8520 行，每个文件都短到能一口气读完。自 1161 行快照之后的增长都花在了看得见的功能上：plan mode、hooks、可持久化 checkpoint/后台任务、stdio/HTTP MCP、可随会话恢复的 Skill、bash/MCP Docker 隔离、Per-Tool Capability、带实时进度的资源感知调度、跨 provider fallback route、美元预算、按完整请求预算的上下文压缩、双层事务化 Session 存储、结构化任务 Memory、后台/worktree 子 agent，以及结构化 Trace/Eval，下文各有交代。

它真能跑：读写文件、执行 shell、按需加载项目工作流、把可压缩聊天历史和持久结构化任务 Memory 分开、派前台或后台子 agent、按需放进独立 Git worktree、分三层压上下文、自动保存可恢复会话，还能随时把这趟烧掉的 token 和美元数报给你。任何要动你磁盘、要跑命令的调用，都会先停下来等你点头，测试套件目前覆盖 297 个用例。但能跑不是为了劝你拿去日用，而是为了让这份「注释」不撒谎：一个解释 agent 怎么运作的范例，自己得真能运作。

代码来自一次公开拆解。公开的源码分析里，Claude Code 这类生产级 agent 暴露出不少关键架构，我挑出最核心的一层，用尽量少的代码诚实地复写了一遍。所以读 CoreCoder，约等于读一份基于公开源码分析的「可运行注释版」：讲的是这类 agent 的核心思路，而它本身只是最小复写，就摆在你机器上，随你拆、随你改。

<p align="center">
  <img src="assets/demo-plan-hooks.gif" width="760"
       alt="plan mode 实战：agent 先读 fib.py，写入被 plan mode 拦下，先给计划；批准之后才动手改文件、跑测试，Pre/PostToolUse hooks 在每次调用前后触发">
</p>

<p align="center"><sub><i>这一千行真能跑通一个完整回合：让它修 buggy.py，它自己读文件、改代码、跑一遍确认、再给结论。看完就回来读代码。</i></sub></p>

这份 README 也就按这条线铺开：上半带你**读懂**（代码地图、主循环、八篇导读），下半带你 **fork** 它、再指几个能往更好里做的方向。

## 先跑一次（读之前的五分钟）

读源码之前，先让它在你机器上活一次，建立点体感。它是个拿来 fork 的地基，所以推荐直接 clone 下来、可编辑安装，边读边改：

```bash
git clone https://github.com/he-yufeng/CoreCoder
cd CoreCoder
pip install -e .
```

只想先跑起来找找感觉，直接 `pip install corecoder` 也行。

给它一个模型加一把 key 就能动。默认走 OpenAI 兼容接口，换 provider 通常只是改两个环境变量：

| Provider | 环境变量示例 |
|---|---|
| OpenAI（默认 `gpt-5.5`） | `OPENAI_API_KEY=sk-...` |
| DeepSeek | `OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com CORECODER_MODEL=deepseek-chat` |
| OmniRoute | `OPENAI_API_KEY=your-key OPENAI_BASE_URL=http://localhost:20128/v1 CORECODER_MODEL=auto` |
| 本地 Ollama | `OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder` |

Kimi、Qwen 这些同样是改这两个变量；连 OpenAI 兼容接口都不给的 provider，装上可选的 LiteLLM 后端（`pip install "corecoder[litellm]"`）能路由一百多家。第三篇文章把这块讲得更细。key 可以直接 `export`，也可以在项目根目录扔个 `.env`，启动时自动加载。然后：

需要故障降级和客户端花费上限时，可以配置有顺序的 fallback 链与美元预算。`CORECODER_FALLBACK_MODELS` 是同一主端点内切模型的短写法；`CORECODER_FALLBACK_ROUTES` 可以为每一跳独立选择 OpenAI-compatible/LiteLLM 后端、端点、凭据环境变量、模型和可选价格：

```bash
export CORECODER_FALLBACK_MODELS=gpt-5.4-mini,gpt-4o-mini
export QWEN_API_KEY=sk-...
export CORECODER_FALLBACK_ROUTES='[{"name":"qwen","provider":"openai","model":"qwen3-plus","base_url":"https://dashscope.example/v1","api_key_env":"QWEN_API_KEY"}]'
export CORECODER_MAX_COST_USD=1.00
# CLI 可重复传 --fallback-model 和 --fallback-route JSON。
```

端到端真机冒烟过三家（读文件、改代码、跑一次确认、自己报告）：DeepSeek、Qwen3、Kimi K2，走同一个 OpenRouter 兼容端点，各自完整跑完全循环。写脚本用 one-shot 的留意：`-p` 默认拒绝一切改动类工具，要加 `--yes`，这是设计如此。

```bash
corecoder                                  # 交互式 REPL
corecoder -p "给 parse_config() 加错误处理"   # 一次性模式，干完就退
```

## 读懂它：代码地图

整个项目摊开就这么大，clone 之前扫一眼，心里就有数了。这也是它和 Claude Code 几十万行最实在的区别：你能把它当一本书的目录来读。建议从 `agent.py` 的主循环读起，那是整个 agent 的心脏。

```
corecoder/
├── agent.py        主循环 + 调度 + 稳定状态快照        1220 行   ← 从这里开始读
├── capabilities.py Per-Tool 权限能力策略              176 行
├── decisions.py    统一结构化工具闸门决策               97 行
├── resources.py    跨 Agent 资源读写锁                  97 行
├── llm.py          流式 + 重试 + provider route        871 行
├── context.py      按完整请求预算的上下文压缩          431 行
├── session.py      存储兼容会话门面                   161 行
├── storage.py      Transcript + Active Context 存储    790 行
├── memory.py       Goal + 约束 + Plan + 决策            299 行
├── permissions.py  改动类工具的用户授权               110 行
├── hooks.py        工具调用前后的用户 shell 钩子        99 行
├── protect_paths_hook.py  可选敏感路径写保护           118 行
├── mcp.py          stdio/HTTP MCP + 恢复/准入         707 行
├── skills.py       项目/用户 Skill 发现与解析         160 行
├── sandbox.py      local/Docker 命令隔离边界           342 行
├── trace.py        默认隐私安全的内存/JSONL 事件        160 行
├── eval.py         可重复用例、断言与指标               586 行
├── prompt.py       系统提示词                          41 行
├── cli.py          REPL + 斜杠命令 + 一次性模式       1023 行
├── config.py       环境变量配置                       109 行
├── checkpoints.py  可持久化 Agent 级 /undo 栈          125 行
├── demo.py         离线端到端演示                       100 行
└── tools/
    ├── bash.py       shell + 执行后端 + cd 追踪         179 行
    ├── edit.py       唯一匹配搜索替换 + diff           126 行
    ├── grep.py       内容搜索                          114 行
    ├── glob_tool.py  文件名匹配                         67 行
    ├── read.py       文件读取                           80 行
    ├── write.py      文件写入                           73 行
    ├── todo.py       agent 自维护的任务清单             91 行
    ├── memory.py     结构化任务状态更新                  83 行
    ├── agent.py      可持久化子 agent/后台任务          645 行
    ├── fetch.py      有大小上限的 HTTP(S) 文本抓取       46 行
    ├── now.py        当前本地时间                        21 行
    ├── skill.py      Skill 加载与持久化激活状态          119 行
    └── base.py       工具基类 + 资源元数据               88 行
.corecoder/skills/
└── corecoder-review/SKILL.md  针对本仓库的审查工作流
examples/
├── plan_hooks_demo.py  离线 plan mode + hooks 演示（免 API key）
└── eval_cases.json     Eval 清单起步示例
```

十三个内建工具：`bash`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`、`todo_write`、`memory_update`、`agent`、`agent_status`、`agent_resume`、`fetch_url` 和 `now`。发现 Skill 时还会加入一个只读的 `load_skill` 适配器；它只增加工作方法，不增加执行权限。存在 `~/.corecoder/mcp.json` 时，里面的 MCP 服务器会再以 `mcp__*` 工具的身份并进来。

## 一个 while 循环就是 agent 的本体

一个 agent 的本体，一句话就能讲清：把用户的话交给模型，模型想调工具就执行，把结果塞回上下文，再问模型，直到它不再要工具、给出回答。落到代码，也就十来行：

```python
# corecoder/agent.py · 主循环（精简骨架）
def chat(self, user_input):
    self.messages.append(user_input)

    for _ in range(self.max_rounds):                   # 循环有上限，跑不飞
        reply = self.llm.chat(self.messages, self.tools)   # 交给模型规划下一步
        if not reply.tool_calls:                       # 模型不再要工具
            return reply.text                          #   → 收工，把回答给用户
        results = schedule_by_resource(reply.tool_calls) # 资源不冲突才并发
        self.messages += results                       # 结果回灌，进入下一轮

    return "(已达轮次上限)"
```

就这么点。这个循环的核心骨架就二十来行，把并行执行和被 Ctrl+C 打断后的回填都算上，也才四十多行。CoreCoder 一千多行里剩下的，几乎全在收拾它真跑起来之后冒出来的岔子。`llm.py` 最后成了全项目最大的文件，不是因为调模型有多难，而是流式返回里一个工具调用的参数会被切成好几段先后送到、得按顺序拼回去，provider 偶尔吐半截 JSON 或把 usage 填成 null，限流（429）、超时、连接中断和 5xx 都得退避重试，其余 4xx 该直接抛就别硬试。这些不起眼的脏活，而不是那个循环，才是一个 agent 从能演示走到能交付真正吃工程功夫的地方；第三篇文章顺着它拆到每一行。

有三个决定值得单独看，因为它们是「先读懂别人怎么做」之后才做得出的取舍，也是你 fork 自己 agent 时可以直接抄走的判断。

**`edit_file` 用唯一匹配的搜索替换，不靠行号。** 行号这东西，模型只要数偏一行，就会悄悄改错地方；锚定一段唯一的原文：匹配不到，就把文件开头甩回去让模型照着重新锚定；匹配到多处，就让它多带几行上下文再来，而不是赌一个。改成功了，连一段 diff 一起返回。失败能复位、成功能复核，闭环都收在工具自己手里。

**上下文不是满了才一刀切，而是先按完整请求算预算，再分层退让。** CoreCoder 会先扣除动态 system prompt、Tool Schema、输出 token 预留和估算安全余量，再对剩余消息预算使用 50/70/90% 阈值：先机械截短陈旧工具输出，再总结旧对话，最后才硬折叠。最新 Tool Result 在被一次成功的 LLM 请求消费前不会参与普通截短；最终还有确定性的强制适配步骤，保证估算后的完整请求低于窗口上限，只有最新一批结果自身都放不下时才会有标记地压缩它。

**约束子 agent 能干什么，靠的是不给它那些能力，而不是写一堆规则求它听话。** 派出去的子 agent 有隔离的上下文和一套新的内建工具实例，但拿不到 `agent`、`agent_status` 与 `agent_resume`，所以不能递归派后代。它复用父 agent 的模型连接、花费账本与资源锁，结果超过 5000 字会截短，轮次上限是 20。`run_mode` 选择阻塞的前台执行或进程内后台执行；`isolation` 再独立选择当前 checkout 或从 `HEAD` 创建并保留的 Git worktree/分支。后台任务记录和文件 checkpoint 会进入 Session 快照；恢复时原本 `queued/running` 的任务会成为 `interrupted`，必须显式且通过 Permission 的 `agent_resume` 才会重启，避免悄悄重复副作用。

每一个「为什么」，下面的文章系列都拆到了具体代码行。

## 配套源码导读 · 八篇双语

我还写了一套双语源码导读，一篇导言加八篇正文，每篇都配英文镜像（`_EN.md`）。它对着 CoreCoder 的真实代码，讲 Claude Code 这类 agent 的内部构造。有一条给自己立的硬规矩：每一处行数、每一段代码都从仓库里现读现核，绝不凭印象编。前六篇带你读懂，第七篇带你 fork，第八篇讲怎么不动主循环地扩展它，哪篇先读都行。

- **[导言 · 用 CoreCoder 读懂 Claude Code，再造一个你自己的](article/00-index.md)**
- **[01 一个 agent 的本体，是一个 while 循环](article/01-the-loop.md)** — `agent.py` 的主循环、打断与轮次上限
- **[02 工具系统：让模型安全地动手](article/02-tools.md)** — `tools/` 内建工具、副作用元数据与 bash 安全闸
- **[03 接入任意大模型，顺便把账算清楚](article/03-llm-and-cost.md)** — `llm.py` 的 provider 包装、重试与成本统计
- **[04 用有限的窗口扛住一个长任务](article/04-context.md)** — `context.py` 的三层压缩与孤儿 tool 消息
- **[05 并行执行与子 agent](article/05-parallel-and-subagents.md)** — 资源感知并发与子 agent 隔离
- **[06 把它跑成一个真正的命令行工具](article/06-session-and-cli.md)** — CLI、事务化会话存储与崩溃恢复
- **[07 Fork CoreCoder，搭一个你自己的 coding agent](article/07-build-your-own.md)** — 从 fork 到加自定义工具到换模型
- **[08 不动主循环的三种加法：MCP、钩子与计划模式](article/08-extensibility.md)** — v0.6.0 扩展三件套，以及让它们成立的那条契约

## Fork 它，造个更好的

读懂之后，最自然的下一步就是 fork。起手不用伤筋动骨：

- **换个你常用的模型。** 就是上面那两个环境变量，`llm.py` 是 provider 适配、跨 provider fallback 和花费控制的入口。
- **加一件你自己的工具。** 照 `tools/base.py` 的工具契约写个新文件，声明副作用、Capability 和具体资源读写声明，跑测试、抓网页、调 LSP 都行，第二篇文章末尾手把手带你写第一个。
- **改系统提示词。** `prompt.py` 才 41 行，改一句就能看到 agent 的脾气变了，是门槛最低的「改一处就有反馈」。
- **直接当库 import。** 顶层导出了 `Agent`、`LLM`、`Config`，能嵌进你自己的程序：

```python
from corecoder import Agent, LLM

llm = LLM(model="deepseek-chat", api_key="sk-...", base_url="https://api.deepseek.com")
print(Agent(llm=llm).chat("找出项目里所有 TODO 注释并列出来"))
```

往深里做，方向也都摆在明处。Docker 沙箱现在已有一个可工作的最小基线；下面这些仍是你能接着往下做、把它推向生产级的入口：

- **继续加固沙箱。** `--sandbox docker` 已经给 `bash` 提供真正的容器边界，单个 MCP Server 也可以选择进入同一套加固后的 Docker 运行时；生产部署还可以补自定义 seccomp/AppArmor、按任务制作镜像、镜像签名/扫描，以及把 hooks 也隔离起来。
- **模型降级策略保持显式。** 瞬时故障先耗尽指数退避，再沿独立配置的 provider route 切换；成功后保持在备用路由。可选美元上限按 route 计价，在发送前预留下一次输入、压低最大输出，对未知价格或缺失 usage 直接拒绝。生产 fork 还可以继续做健康度加权路由和账户侧账单告警。
- **子 agent 模式已经显式化，但仍是进程内实现。** 当前已有前台/后台和共享目录/worktree 两组正交模式，任务记录可随 Session 恢复，但 Python 线程不会跨进程存活；未完成任务恢复为 `interrupted`，等待显式 `agent_resume`。生产 fork 还可以补持久化 worker、取消、事件流、自动 merge/cherry-pick 策略和跨进程/跨主机隔离。
- **Trace 和 Eval 是本地、刻意保持小型的积木。** JSONL 能精确复盘一次运行，Eval runner 把确定性用例变成成功率、延迟、token、工具和成本指标。生产 fork 还可以接 OpenTelemetry、Trace UI、语义/模型裁判、数据集管理和 CI 趋势存储。
- **不做 RAG，MCP 客户端也只讲工具这一小片。** 当前已支持 stdio/Streamable HTTP Tool、重连、熔断、刷新和启动前准入，但仍特意没实现 MCP resources/prompts；大仓检索与这两块协议面都是继续扩展的真实方向。

README 只给方向，每条的代码细节第七篇接着讲。挑一个动手，就是把它做得更好的开始。

## 命令

进了 REPL，`/help` 列全部，常用的这几个：

```
/model <名称>    切换模型
/compact         手动压缩上下文
/tokens          查看 token、当前 fallback、费用和剩余预算
/memory          查看可持久化的结构化任务记忆
/goal <内容>     设置任务目标（`clear` 清除）
/constraint <内容>  增加用户关键语义约束
/decision <内容>    记录持久化决策
/diff            查看本次会话改过的文件
/undo            撤销最近一次文件改动
/plan            开关计划模式（只读摸底，再交出待批准的计划）
/save            立即保存当前会话
/name <名称>     给当前会话命名，不改变稳定 Session ID
/session         查看当前会话和存储位置
/sessions        列出会话名称、ID 和首条消息预览
/skills          列出项目和用户工作流 Skill
/transcript      查看不受上下文压缩影响的完整原始历史
/delete-session  按 ID 删除非当前会话
/agents          列出后台子 agent
/mcp-refresh     必要时重连并刷新 MCP Tool Schema
quit / exit      退出（Ctrl+C 取消当前回合）
```

交互和一次性任务默认自动保存到 `~/.corecoder/sessions/sessions.db`。输入 `/name 修复登录流程` 可以增加便于人识别的名称，自动生成的 Session ID 仍是 `corecoder -r <id>` 使用的稳定唯一键，名称允许重复；`/sessions` 会同时显示名称、ID 和首条消息预览，不再需要靠记随机 ID 选择会话。存储分成两层：`events` 是 append-only Transcript，完整保留原始 user/assistant/tool call/tool result；`messages` 是真正用于 Resume 和下一轮模型请求的 Active Context，允许被 Context Manager 摘要和截短。压缩后的上下文快照、原始 Transcript 尾部和 Session 元数据在同一个事务里提交，因此模型可以只读较短上下文，`/transcript` 仍能回看压缩前原文。

SQLite WAL 支持并发读写。Agent 只在 provider 可接受的稳定边界发快照：用户消息落入后、完整 Tool Result 批次回填后，以及完成、失败或中断时。因此恢复出来的 Active Context 不会出现只有 assistant `tool_calls`、缺少 observation 的半截结构。压缩记录还会进入 `summaries` 表，包含动作、前后 token、生成模型和摘要消息。`--no-autosave` 可关闭自动写入但保留 `/save`；`--storage PATH` 或 `CORECODER_STORAGE_PATH` 可换数据库位置。旧 v1/v2/JSON Session 会在读取时迁移；因为过去被压掉的原文无法重建，迁移记录会明确标为 Transcript 不完整。

持久化内容包括完整 Transcript、压缩后的 Active Context、摘要记录、结构化任务 Memory、模型、workspace、状态、按模型/Provider Route 统计的 token/cost、plan mode、todo、已激活 Skill、文件 checkpoint、后台任务记录和尚未消费的 Tool Result ID；API Key 与 permission 授权刻意不落盘。Session ID 仍会先规整成安全的数据库 key/旧版文件名，SQLite 与 JSON 文件在系统允许时只授予当前用户访问权限。

数据库在本机，但没有加密。messages 和 Tool Result 自身可能包含源码、prompt、命令输出，甚至工具从 workspace 读到的秘密；敏感任务请加 `--no-autosave`，或者用 `--storage` 指向妥善保护的位置。

## 结构化任务 Memory

模型看到的状态不只是一串对话。`MemoryState` 把有界的 Goal、用户 Critical Constraints、Plan Steps、Decisions 和系统推导的 Files Modified 放在 `messages` 之外；`Agent._full_messages()` 每轮重新注入，Session 快照在重启后恢复。因此 Context 摘要可以压掉旧闲聊，却不会顺手丢掉当前目标或用户明确约束。内建 `memory_update` Tool 允许模型维护 Goal、Plan 和 Decision，但刻意不能把 Tool Output 自行提升为 Critical Constraint。成功的内建文件写入会根据真实 Tool 参数登记 Files Modified，而不是相信模型声称改过什么。

这一层的 Constraint 是语义指令，不冒充操作系统安全边界；可以用 `/constraint revoke <id>` 撤销，真正必须强制执行的规则仍应交给 Hook、Capability Policy、Permission 或 Sandbox。前台/后台子 Agent 会继承父 Agent 约束和决策的副本，但各自维护 Plan 与修改文件状态，不会通过共享引用污染父 Memory。

## Skills

Skill 是可复用的工作流指令，刻意与可执行 Tool 分开。CoreCoder 会发现 `~/.corecoder/skills/*/SKILL.md` 下的用户 Skill，以及 `<workspace>/.corecoder/skills/*/SKILL.md` 下的项目 Skill；同名时项目级覆盖用户级。启动时只把校验后的 `name` 和 `description` 暴露给模型。任务匹配时，模型才调用只读的 `load_skill`，完整指令作为普通 Tool Result 进入上下文，因此无需给 Agent Loop 再开一条特殊执行通道，Context、Transcript、Trace 和子 agent 工具共享也都照常工作。

加载成功后还会生成一条轻量、可随 Session 持久化的 `active_skills` 记录，包含名称、作用域、加载时间、状态和 SHA-256 内容哈希。Agent 每次请求 LLM 都会把哈希匹配的已激活指令重新注入 system message，因此 Context 压缩和 Resume 不会悄悄忘掉工作流；完整指令仍只保存在 `SKILL.md` 中，不会复制进 Session 元数据。如果文件发生变化或消失，CoreCoder 会 fail closed：旧指令不再注入，`/skills` 显示 `changed` 或 `unavailable`；需要再次调用 `load_skill` 才会明确接受新版本。`/reset` 会清除已激活 Skill。

仓库自带一个自然的示例 `corecoder-review`：审查改动时重点检查 Agent Loop 协议不变量、调度/Permission/Sandbox 边界、Context 保护以及双层 Session 规则。输入 `/skills` 可以查看发现结果，然后直接要求模型使用 `corecoder-review`。Skill 永远不能授予 Agent 原本没有的 Tool，也不能绕过 Permission 或 Sandbox；项目 Skill 属于仓库提供的指令，面对不可信 checkout 应先审阅内容。

## 工具执行进度

工具执行现在有独立的实时视图，不再只是把已发起的调用名逐行打印出来。模型一次返回多个调用时，只有资源感知调度器确认存在真正并发批次，标题才显示 `Parallel tools`；否则显示 `Tool batch`。Pure/Read 调用仍可重叠；显式 opt-in 的有状态 Tool 会返回读/写 `ResourceClaim`：不同文件写入、不同 MCP Server 可并行，同一文件/Server 必须串行；共享的 `ResourceLockManager` 还会协调父子 Agent。每项调用会按 `queued` → `running` → `done`、`error` 或 `blocked` 更新，带耗时和总体 `n/N` 计数。所有 permission 提问都先在主线程处理完，之后才启动进度 UI，因此交互授权不会和 Rich 的实时刷新抢终端。

把 CoreCoder 当库嵌入时，也可以向 `Agent.chat()` 传 `on_tool_progress`，用同一组结构化事件驱动自己的界面：`batch_started`、`tool_started`、`tool_completed`、`batch_completed`。原有的 `on_tool(name, arguments)` 仍保持兼容；进度回调只属于展示层，即使它自己报错，工具和 Agent Loop 也会继续执行。

## 权限

只读和进程内状态工具（`read_file`、`glob`、`grep`、`todo_write`、`memory_update`、`now`、`agent_status`、`load_skill`）模型一调就跑。会修改磁盘或访问外部资源的那些（`edit_file`、`write_file`、`bash`、`fetch_url`、MCP 工具、派生子 agent 与 `agent_resume`）先停下来等你点头，REPL 启动横幅里能看到当前是哪种模式。Hook、Capability、Plan Mode 和 Permission 现在统一返回结构化 `ToolDecision`，Trace 不再需要猜测混杂的字符串/tuple：

- REPL 里每次调用问一次：允许这一次、本工具本次会话都允许、或者拒绝。前台子 agent 继承这层交互授权；后台线程绝不会与 REPL 抢输入，只能使用 `--yes` 或已被标记为「总是允许」的工具，其余有状态调用 fail closed，让子 agent 自己绕开。
- 一次性模式（`-p`）没人可问，改动类调用当场被拒，拒绝理由作为普通工具结果回给模型：循环绝不会卡在等一个永远不会来的输入上。要全部预授权就加 `--yes`（脚本、CI 场景）。
- 判断本身是 `permissions.py` 里的纯逻辑，终端只是塞进来一个提问回调。不用 TTY 也能单测授权逻辑，或者直接搬进你自己的嵌入场景。

## Per-Tool Capability

每个 Tool 现在显式声明它可能使用的权限能力：`filesystem_read`、`filesystem_write`、`network`、`process`、`subagent`、`mcp` 或 `unknown`。可选的 `~/.corecoder/capabilities.json` 可以按精确工具名或 glob 限制这些能力。检查发生在 PreToolUse Hook 之后、Permission 之前，因此 Capability 被拒时既不会弹授权问题，也不会进入 `Tool.execute()`：

```json
{
  "default": "deny",
  "tools": {
    "read_file": {"allow": ["filesystem_read"]},
    "bash": {"allow": ["filesystem_read", "filesystem_write", "process"]},
    "fetch_url": {"allow": ["network"]},
    "mcp__weather__*": {"allow": ["mcp", "process", "network"]}
  }
}
```

可以使用 `--capability-policy PATH` 或 `CORECODER_CAPABILITY_POLICY`；`/capabilities` 会显示当前策略和所有 Tool 的能力声明。没有策略文件时保持向后兼容的全部允许。`default: deny` 下，不需要外部能力的 `now`、`agent_status`、内存 todo 和结构化 Memory 仍可运行，未知自定义 Tool 则 fail closed。MCP 会在启动进程或首次 HTTP 请求前先检查服务器通配名（例如 `mcp__weather__*`），拒绝的服务器根本不会启动，发现后的每次 Tool Call 还会再过一次策略。更完整的起点见 [examples/capabilities.json](examples/capabilities.json)。

```bash
corecoder --capability-policy examples/capabilities.json --sandbox docker
# 进入 REPL 后输入 /capabilities
```

网络策略只承诺真正能强制执行的边界。`fetch_url` 声明 `network`；本地 `bash` 因宿主进程可以联网也声明它，只有 Docker `bash` 在 `network=none` 真正切断网络后才移除该能力。因此规则不允许 `network` 时，本地或联网 bash 会整项拒绝，而不会假装能从 Shell 字符串猜出哪条命令要联网。当前不宣称支持域名白名单：任意 Shell/MCP 实现可以隐藏或重定向真实目标，需要域名级边界时应使用断网容器或外部白名单代理。

## Docker 沙箱

为了兼容旧用法，默认仍在本机执行。要给 `bash` 加上容器级边界，先构建项目镜像，再显式开启：

```bash
docker build -f Dockerfile.sandbox -t corecoder-sandbox:latest .
corecoder --sandbox docker
```

每次 bash 调用都会启动一个全新的一次性容器。只有启动 CoreCoder 时的当前目录会以 `/workspace` 挂进去；容器根文件系统只读，网络默认 `none`，全部 Linux capabilities 被丢弃，禁止提权，并设置 CPU、内存、PID 和 `/tmp` 上限。进程使用宿主机 UID/GID，镜像缺失时也不会偷偷联网拉取。内建文件工具同时拒绝解析后落在启动目录之外的路径，包括通常的 `..` 和符号链接逃逸。bash 命令请使用项目相对路径或 `/workspace/...`。

这些默认值可以通过 `CORECODER_SANDBOX*` 环境变量或对应 CLI 参数覆盖：`--sandbox-image`、`--sandbox-network`、`--sandbox-memory`、`--sandbox-cpus`、`--sandbox-pids`。例如 `--sandbox-network bridge` 会显式开放容器网络。自定义镜像只要带 `/bin/sh` 和项目需要的运行时即可。

边界范围需要说清：hooks、LLM 客户端和 `fetch_url` 仍在宿主机运行；MCP server 没有单独配置 `sandbox` 时也仍在宿主机。bash 的工作区必须可写，因为这是 coding agent。宿主机环境变量不会传进 bash 容器，但工作区内原本就有的所有文件——包括 `.env` 或凭据——工具仍然看得见；面对不可信任务，应使用干净 worktree，并把秘密放在工作区之外。Permission 决定「是否授权做」，Capability Policy 限制 Tool 可以申请哪类能力，Sandbox 限制授权后最多能伤到哪里。Docker daemon 或镜像不可用时，调用会作为普通工具错误返回，绝不会降级到宿主机执行。

## 计划模式

REPL 里 `/plan` 开关计划模式。开着的时候，提示符变成 `(plan)`，一切对外改动类调用（写文件、编辑、bash、MCP 工具、子 agent）当场被拒：拒绝理由作为普通工具结果回给模型，让它用只读工具继续摸底。模型可以通过 `memory_update` 维护结构化 Plan Steps，它们不会被 Context 压缩删除，并能随 Session Resume。计划看着没问题，敲 `approve`（或再敲一次 `/plan`）就把执行权交回去，agent 接着干活；执行限制本身仍只是 `Agent` 开关和授权闸前的一支拒绝分支。

## 钩子

在 `~/.corecoder` 下放一个 `hooks.json`，就能让你自己的 shell 命令在每次工具调用前后跑起来，思路和 Claude Code 的 hooks 一致：

```json
{
  "PreToolUse":  [{"matcher": "bash", "command": "cat >> ~/.corecoder/audit.jsonl"}],
  "PostToolUse": [{"matcher": "*",    "command": "cat >> ~/.corecoder/trace.jsonl"}]
}
```

每个钩子从 stdin 拿到这次调用的 JSON（`tool_name`、`tool_input`，post 钩子还带 `tool_response`）。matcher 是精确的工具名，留空或写 `*` 表示对所有工具生效。pre 钩子可以否决这次调用：退出码 2，它的 stderr 会作为理由回给模型，让它换条路走。post 钩子只观察，永远拦不住。钩子报错或超过十秒会被跳过并记一条警告：钩子是来帮忙的，没权力弄死循环。整个机制就是 `hooks.py` 一个文件，REPL 启动横幅会显示加载了几条。

CoreCoder 现在附带一个可选但实用的敏感路径保护 Hook。拉到这版源码后先更新
editable 安装；仅当你还没有 Hook 配置时复制示例，否则请把其中的
`PreToolUse` 项合并进现有文件：

```bash
python -m pip install -e .
mkdir -p ~/.corecoder
cp examples/hooks.protect-sensitive.json ~/.corecoder/hooks.json
```

示例使用 Hook Runtime 注入的 `CORECODER_PYTHON`，它始终指向运行 CoreCoder
的同一个 Python，不会误用另一个虚拟环境：

```json
{
  "PreToolUse": [{
    "matcher": "*",
    "command": "\"$CORECODER_PYTHON\" -m corecoder.protect_paths_hook"
  }]
}
```

它会拒绝 `write_file` / `edit_file` 修改常见 `.env` 文件、私钥扩展名、
`.git`、`secrets/` 以及 `production.yaml` / `production.yml`。匹配同时检查
原始路径、解析后的真实路径和 Workspace 相对路径，因此 `..` 与已有符号链接
不能伪装受保护目标。可以通过
`corecoder-protect-paths --pattern 'config/prod/*'` 增加项目规则（也可把这些
参数直接加到 JSON 里的 module 命令后），或者用
`--no-defaults` 只保留自己的规则。这个 Hook 刻意不猜测 `bash` 的副作用：
任意 shell 命令无法靠字符串可靠分类，那一层应交给 Capability Policy 与
Docker Sandbox。

## MCP 服务器

在 `~/.corecoder` 下放一个 `mcp.json`，任何 MCP 服务器的工具就能通过 stdio 或 Streamable HTTP 接进 agent：

```json
{
  "mcpServers": {
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
    "weather": {"url": "https://mcp.example.test", "headers": {"Authorization": "Bearer ..."}}
  }
}
```

每个通过启动准入的服务器会握手、列出工具；每件工具都注册成 `mcp__<服务器>__<工具>`，Hook、Capability Policy、Permission、资源调度与 Trace 对它和内建工具一视同仁。MCP 工具不在只读名单里。握手给十五秒，一次调用给六十秒。stdio 进程死亡后，下一次调用前会自动重连和重新握手；失败的那次 `tools/call` 本身绝不自动重放，因为远端副作用是否已经发生并不确定。连续传输失败会打开可配置熔断器（默认三次、三十秒）。`/mcp-refresh` 会按需重连、重新执行 `tools/list`，并原子替换 Agent Tool Registry 与 System Prompt 中的 Schema；刷新失败时保留旧 Tool 包装器。HTTP 会保存 `Mcp-Session-Id`，接受 JSON/SSE 响应，关闭时发送 `DELETE`。客户端仍只实现工具协议面（`initialize`、`ping`、`tools/list`、`tools/call`），没有实现 resources/prompts。没有 `mcp.json` 就没有 MCP，一切照旧。

`reconnect`、`circuit_failures`、`circuit_cooldown` 可以写在 `defaults` 或某个 Server 下。stdio Server 可配置 `sandbox`；远端 HTTP 不能塞进本地进程容器，其能力声明只有 `mcp` + `network`。同一个 Server 的调用因为共享会话/状态而串行，不同 Server 可以并行。

为了兼容已有配置，默认仍是宿主机进程；也可以让 Server 常驻在一个加固的 Docker stdio 容器中：

```json
{
  "defaults": {
    "sandbox": {
      "mode": "docker",
      "image": "my-mcp-runtime:latest",
      "network": "none",
      "workspace": "none",
      "memory": "512m",
      "cpus": 0.5,
      "pids": 64
    }
  },
  "mcpServers": {
    "weather": {
      "command": "weather-mcp-server",
      "env": {"WEATHER_API_KEY": "replace-me"},
      "sandbox": {"network": "bridge"}
    },
    "review": {
      "command": "review-mcp-server",
      "sandbox": {"workspace": "ro"}
    }
  }
}
```

指定的可执行文件必须存在于镜像内部。Docker MCP 和 Docker bash 使用相同的只读根文件系统、Capability 清空、禁止提权、非 root UID/GID、私有 `/tmp`、资源限制、`--pull never`、精确容器名清理和失败关闭机制。MCP 容器不继承宿主环境变量，只有配置里的 `env` 会进入；Workspace 默认完全不挂载，必须显式选择 `ro` 或 `rw`，挂载后位于 `/workspace`。网络默认 `none`；`bridge` 会给整个 Server（包括启动阶段）开放出站网络。一个 MCP Server 下的所有 Tool 共享同一进程边界，因此不同信任级别或网络要求的 Tool 应拆成不同 Server/镜像。

仓库提供了一个不依赖第三方包的端到端验证样例：把 [examples/mcp.sandbox.json](examples/mcp.sandbox.json) 复制到 `~/.corecoder/mcp.json`，从本仓库根目录启动 CoreCoder，再让模型调用 `mcp__sandbox_demo__echo`。它会在只读 Workspace、关闭网络的容器里运行 [examples/minimal_mcp_server.py](examples/minimal_mcp_server.py)。

## Trace 与 Eval

Trace 默认不开。传入路径（或设置 `CORECODER_TRACE`）后，CoreCoder 会按事件逐行追加 JSON：

```bash
corecoder --trace .corecoder/run.jsonl
# 默认不记载荷；这个显式开关可能记录源码和秘密：
corecoder --trace .corecoder/debug.jsonl --trace-content
```

时间线覆盖 Agent 运行与轮次、LLM 延迟/usage/重试/fallback、工具请求与耗时、Hook 和 Permission 决策、Context 压缩、子 Agent 生命周期。每条记录都有时间戳、序号和 Trace session ID；Agent 事件还会在适用时带上 `run_id`、`agent_id`、`parent_agent_id`、`subagent_task_id` 与当前轮次。不开内容记录，也足以回答「哪一轮走偏」「哪个工具最慢」「为什么触发 fallback」「这个子 Agent 花了多少 token」。开 `--trace-content` 后，Context 压缩事件还会带有受长度限制的压缩前后 messages，便于检查丢失了什么。JSONL 写入线程安全且 best-effort：Trace sink 坏了只告警，不会反过来弄死 Agent。

作为库使用时，同一接口可选择 `MemoryTrace`、`JsonlTrace` 或 `CompositeTrace`：

```python
from corecoder import Agent, MemoryTrace

trace = MemoryTrace()
agent = Agent(llm=llm, trace=trace)
agent.chat("检查这个仓库")
print(trace.events)
```

`corecoder-eval` 会让新建的 Agent 实例运行 JSON 清单。默认每个用例都复制到临时工作区，可以断言最终文本、正则、文件、文件内容、用过的工具，以及轮次/调用数/延迟/token/成本上限。重复运行会给出每个用例的通过率与 pass@k，以及整套汇总：

```bash
corecoder-eval examples/eval_cases.json --repeat 3 \
  --output reports/current.json --trace-dir reports/traces
corecoder-eval examples/eval_cases.json --repeat 3 \
  --baseline reports/current.json --output reports/candidate.json
```

第二条命令会对基线增加带正负号的单次运行指标差值，于是一次改动可以用成功率、延迟、token、工具数和成本讨论，而不是只说「演示看起来更好」。报告会保存最终回答、实际使用的模型、不含密钥的运行配置与清单哈希；CLI 的模型/fallback/预算参数和环境变量配置与 `corecoder` 一致。改动类工具默认拒绝，只有 `--yes` 才放行；源码工作区仍会复制，除非显式 `--in-place`，而 `--keep-workspaces DIR` 可保留副本供排查。复制目录不是 shell 安全边界：面对不可信 prompt 或允许修改的 Eval，请先构建沙箱镜像并加 `--sandbox docker`。

## 相关项目

如果你读 CoreCoder 读得还顺，下面几个我做的 agent / LLM 系统方向的工具也许用得上：

- **[RepoWiki](https://github.com/he-yufeng/RepoWiki)** — 被丢进一个陌生代码库？它给你一份带「从哪读起」路径的 wiki，一个可自托管的 DeepWiki 替代。
- **[FindJobs-Agent](https://github.com/he-yufeng/FindJobs-Agent)** — 别再手动刷招聘网站：它按你的简历给岗位排序，还能跑模拟面试。
- **[ContractGuard](https://github.com/he-yufeng/ContractGuard)** — 签字前先把有风险的条款挑出来：它读合同、标出危险点。
- **[GitSense](https://github.com/he-yufeng/GitSense)** — 想给开源做贡献？它帮你找到值得做的 issue，还能估你的 PR 多大概率被合。
- **[CodeABC](https://github.com/he-yufeng/CodeABC)** — 不会写代码也能看懂一个项目，专给小白做的。

## 贡献 / License

动手之前先跑一遍 `pytest tests/ -q`（297 个用例）、`ruff check` 和 `compileall`，绿了再提。MIT License，欢迎 fork 拿去造更好的东西，能在 README 里留一句出处就更好。

---

作者 [何宇峰](https://github.com/he-yufeng)，曾任职 Moonshot AI (Kimi)。早前写过一篇相当完整的 [Claude Code 源码分析](https://zhuanlan.zhihu.com/p/1898797658343862272)，这个项目是它的动手版：那篇带你读懂，这个带你重建。

> CoreCoder 原名 NanoCoder，为避免和 [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder) 混淆而改名，旧链接会自动跳到这里。
