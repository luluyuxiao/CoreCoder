<div align="center">

# CoreCoder

**编程 agent 里的 nanoGPT。2.6k 行引擎、整包 5357 行纯 Python 全部一口气可读，读懂一个 coding agent 到底怎么运作，再 fork 出你自己的。**

*learn from it · fork it · ship something better*

中文 | [English](README.md) | [配套源码导读 · 八篇双语](article/)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/he-yufeng/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/CoreCoder/actions)
[![engine](https://img.shields.io/badge/engine-2597_LoC-blue)](article/)
[![源码导读](https://img.shields.io/badge/源码导读-8篇双语-orange)](article/)

</div>

- **读得完。** 一个下午读完整个引擎，没有一处藏着你看不懂的魔法。
- **改得动。** 每一行都能在你自己机器上下断点、改了再跑。它真能干活，所以这份参考是活的，不是示意图。
- **留白即起点。** 刻意只留最小核心，没做的那些不是半成品，是留给你 fork 出更好东西的地方。

## 和谁比

| | CoreCoder | Claude Code | aider | nanoGPT |
|---|---|---|---|---|
| 代码量 | 引擎约 2597 行 / 整包 5357 行 | 几十万行（闭源） | 数万行 Python | 约 600 行（两个文件） |
| 读完要多久 | 一个下午 | 读不了（闭源） | 得啃几天 | 一个下午 |
| 能不能下断点改了再跑 | 能，每一行 | 不能 | 能，但量大 | 能 |
| 定位 | 读懂并 fork 出你自己的 agent | 生产级编程助手 | 终端结对编程 | 教学用最小 GPT |

nanoGPT 那一列是拿来对照的：它最小、可读，但教的是训一个 GPT。CoreCoder 想干的是同一件事，只是把对象换成一个能真正改代码的 agent。和 Claude Code、aider 摆在一起，不是要跟它们抢用户，CoreCoder 是借它们来学、来起步的那块地基，根本不在一个赛道。

## 这是什么

我一直觉得 coding agent 被讲得太玄了。把 Claude Code、Cursor 这类工具扒到底，核心是一个 while 循环套着一个大模型，外加七八个让它能真正动手的工具。难的从来不是这个循环，而是循环跑进真实世界以后要兜的那些底。CoreCoder 就是把这个核心老老实实写出来的最小版本。

引擎部分（循环、模型接口、上下文、工具、会话）去掉空行和注释是 2597 行。把 Trace、Eval、最外层的 CLI、配置、打包一起算，整个包 29 个文件、物理 5357 行、净 4594 行，每个文件都短到能一口气读完。自 1161 行快照之后的增长都花在了看得见的功能上：plan mode、hooks、checkpoints、MCP、可选的 Docker 沙箱、带实时进度的副作用感知工具调度、模型 fallback、美元预算、按完整请求预算的上下文压缩、后台/worktree 子 agent，以及结构化 Trace/Eval，下文各有交代。

它真能跑：读写文件、执行 shell、派前台或后台子 agent、按需放进独立 Git worktree、分三层压上下文，还能随时把这趟烧掉的 token 和美元数报给你。任何要动你磁盘、要跑命令的调用，都会先停下来等你点头，测试套件目前覆盖 228 个用例。但能跑不是为了劝你拿去日用，而是为了让这份「注释」不撒谎：一个解释 agent 怎么运作的范例，自己得真能运作。

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

需要故障降级和客户端花费上限时，可以配置有顺序的 fallback 链与美元预算。同一个 `LLM` 实例里的候选模型共用 backend、端点和显式凭据；凭据齐备时，LiteLLM 可以使用 `provider/model` 名称：

```bash
export CORECODER_FALLBACK_MODELS=gpt-5.4-mini,gpt-4o-mini
export CORECODER_MAX_COST_USD=1.00
# 等价 CLI：--fallback-model gpt-5.4-mini --fallback-model gpt-4o-mini --max-cost 1.00
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
├── agent.py        主循环 + 调度 + Trace 事件          701 行   ← 从这里开始读
├── llm.py          流式 + 重试 + fallback + 预算       567 行
├── context.py      按完整请求预算的上下文压缩          431 行
├── session.py      会话存盘 / 续聊 + 路径穿越防护      97 行
├── permissions.py  改动类工具的用户授权               108 行
├── hooks.py        工具调用前后的用户 shell 钩子        85 行
├── mcp.py          MCP stdio 客户端，接外部工具       210 行
├── sandbox.py      local/Docker 命令隔离边界           272 行
├── trace.py        默认隐私安全的内存/JSONL 事件        160 行
├── eval.py         可重复用例、断言与指标               552 行
├── prompt.py       系统提示词                          41 行
├── cli.py          REPL + 斜杠命令 + 一次性模式        634 行
├── config.py       环境变量配置                        88 行
├── checkpoints.py  /undo 快照与回滚                      44 行
├── demo.py         离线端到端演示                       100 行
└── tools/
    ├── bash.py       shell + 执行后端 + cd 追踪         169 行
    ├── edit.py       唯一匹配搜索替换 + diff           105 行
    ├── grep.py       内容搜索                          112 行
    ├── glob_tool.py  文件名匹配                         65 行
    ├── read.py       文件读取                           65 行
    ├── write.py      文件写入                           52 行
    ├── todo.py       agent 自维护的任务清单             80 行
    ├── agent.py      子 agent 模式 + 后台任务管理       432 行
    ├── fetch.py      有大小上限的 HTTP(S) 文本抓取       44 行
    ├── now.py        当前本地时间                        20 行
    └── base.py       工具基类 + 副作用元数据             56 行
examples/
├── plan_hooks_demo.py  离线 plan mode + hooks 演示（免 API key）
└── eval_cases.json     Eval 清单起步示例
```

十一个内建工具：`bash`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`、`todo_write`、`agent`、`agent_status`、`fetch_url` 和 `now`。其余都是包在引擎核心外面的 CLI 外壳、配置和打包。存在 `~/.corecoder/mcp.json` 时，里面的 MCP 服务器会以 `mcp__*` 工具的身份并进来，下面有专门一节讲。

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
        results = schedule_by_effect(reply.tool_calls) # 只读并发，副作用调用串行
        self.messages += results                       # 结果回灌，进入下一轮

    return "(已达轮次上限)"
```

就这么点。这个循环的核心骨架就二十来行，把并行执行和被 Ctrl+C 打断后的回填都算上，也才四十多行。CoreCoder 一千多行里剩下的，几乎全在收拾它真跑起来之后冒出来的岔子。`llm.py` 最后成了全项目最大的文件，不是因为调模型有多难，而是流式返回里一个工具调用的参数会被切成好几段先后送到、得按顺序拼回去，provider 偶尔吐半截 JSON 或把 usage 填成 null，限流（429）、超时、连接中断和 5xx 都得退避重试，其余 4xx 该直接抛就别硬试。这些不起眼的脏活，而不是那个循环，才是一个 agent 从能演示走到能交付真正吃工程功夫的地方；第三篇文章顺着它拆到每一行。

有三个决定值得单独看，因为它们是「先读懂别人怎么做」之后才做得出的取舍，也是你 fork 自己 agent 时可以直接抄走的判断。

**`edit_file` 用唯一匹配的搜索替换，不靠行号。** 行号这东西，模型只要数偏一行，就会悄悄改错地方；锚定一段唯一的原文：匹配不到，就把文件开头甩回去让模型照着重新锚定；匹配到多处，就让它多带几行上下文再来，而不是赌一个。改成功了，连一段 diff 一起返回。失败能复位、成功能复核，闭环都收在工具自己手里。

**上下文不是满了才一刀切，而是先按完整请求算预算，再分层退让。** CoreCoder 会先扣除动态 system prompt、Tool Schema、输出 token 预留和估算安全余量，再对剩余消息预算使用 50/70/90% 阈值：先机械截短陈旧工具输出，再总结旧对话，最后才硬折叠。最新 Tool Result 在被一次成功的 LLM 请求消费前不会参与普通截短；最终还有确定性的强制适配步骤，保证估算后的完整请求低于窗口上限，只有最新一批结果自身都放不下时才会有标记地压缩它。

**约束子 agent 能干什么，靠的是不给它那些能力，而不是写一堆规则求它听话。** 派出去的子 agent 有隔离的上下文和一套新的内建工具实例，但拿不到 `agent` 与 `agent_status`，所以不能递归派后代。它复用父 agent 的模型连接与花费账本，结果超过 5000 字会截短，轮次上限是 20。`run_mode` 选择阻塞的前台执行或进程内后台执行；`isolation` 再独立选择当前 checkout 或从 `HEAD` 创建并保留的 Git worktree/分支。

每一个「为什么」，下面的文章系列都拆到了具体代码行。

## 配套源码导读 · 八篇双语

我还写了一套双语源码导读，一篇导言加八篇正文，每篇都配英文镜像（`_EN.md`）。它对着 CoreCoder 的真实代码，讲 Claude Code 这类 agent 的内部构造。有一条给自己立的硬规矩：每一处行数、每一段代码都从仓库里现读现核，绝不凭印象编。前六篇带你读懂，第七篇带你 fork，第八篇讲怎么不动主循环地扩展它，哪篇先读都行。

- **[导言 · 用 CoreCoder 读懂 Claude Code，再造一个你自己的](article/00-index.md)**
- **[01 一个 agent 的本体，是一个 while 循环](article/01-the-loop.md)** — `agent.py` 的主循环、打断与轮次上限
- **[02 工具系统：让模型安全地动手](article/02-tools.md)** — `tools/` 十一个工具、副作用元数据与 bash 安全闸
- **[03 接入任意大模型，顺便把账算清楚](article/03-llm-and-cost.md)** — `llm.py` 的 provider 包装、重试与成本统计
- **[04 用有限的窗口扛住一个长任务](article/04-context.md)** — `context.py` 的三层压缩与孤儿 tool 消息
- **[05 并行执行与子 agent](article/05-parallel-and-subagents.md)** — 副作用感知的只读并发与子 agent 隔离
- **[06 把它跑成一个真正的命令行工具](article/06-session-and-cli.md)** — `session.py` 与路径穿越防护
- **[07 Fork CoreCoder，搭一个你自己的 coding agent](article/07-build-your-own.md)** — 从 fork 到加自定义工具到换模型
- **[08 不动主循环的三种加法：MCP、钩子与计划模式](article/08-extensibility.md)** — v0.6.0 扩展三件套，以及让它们成立的那条契约

## Fork 它，造个更好的

读懂之后，最自然的下一步就是 fork。起手不用伤筋动骨：

- **换个你常用的模型。** 就是上面那两个环境变量，`llm.py`（498 行）是 provider 适配、fallback 和花费控制的入口。
- **加一件你自己的工具。** 照 `tools/base.py`（47 行）的工具契约写个新文件、声明副作用类型，跑测试、抓网页、调 LSP 都行，第二篇文章末尾手把手带你写第一个。
- **改系统提示词。** `prompt.py` 才 41 行，改一句就能看到 agent 的脾气变了，是门槛最低的「改一处就有反馈」。
- **直接当库 import。** 顶层导出了 `Agent`、`LLM`、`Config`，能嵌进你自己的程序：

```python
from corecoder import Agent, LLM

llm = LLM(model="deepseek-chat", api_key="sk-...", base_url="https://api.deepseek.com")
print(Agent(llm=llm).chat("找出项目里所有 TODO 注释并列出来"))
```

往深里做，方向也都摆在明处。Docker 沙箱现在已有一个可工作的最小基线；下面这些仍是你能接着往下做、把它推向生产级的入口：

- **继续加固沙箱。** `--sandbox docker` 已经给 `bash` 提供真正的容器边界；生产部署还可以补自定义 seccomp/AppArmor、只读工作区、按任务制作镜像，以及把 hooks 和 MCP server 进程也隔离起来。
- **模型降级策略保持显式。** 瞬时故障先耗尽指数退避，再沿配置的 fallback 链切换；成功后保持在备用模型。可选美元上限会在发送前预留下一次输入、压低最大输出，对未知价格或缺失 usage 直接拒绝。生产 fork 还可以继续做健康度路由、provider 独立凭据和账户侧账单告警。
- **子 agent 模式已经显式化，但仍是进程内实现。** 当前已有前台/后台和共享目录/worktree 两组正交模式；生产 fork 还可以补持久化 worker、取消、事件流、自动 merge/cherry-pick 策略和跨进程/跨主机隔离。
- **Trace 和 Eval 是本地、刻意保持小型的积木。** JSONL 能精确复盘一次运行，Eval runner 把确定性用例变成成功率、延迟、token、工具和成本指标。生产 fork 还可以接 OpenTelemetry、Trace UI、语义/模型裁判、数据集管理和 CI 趋势存储。
- **不做 RAG，MCP 客户端也只讲工具这一小片。** 给大仓加检索式代码定位还空着，`mcp.py` 也特意没实现 resources 和 prompts。随便挑一个，都是从最小核心往你自己的更强 agent 扩的真实方向。

README 只给方向，每条的代码细节第七篇接着讲。挑一个动手，就是把它做得更好的开始。

## 命令

进了 REPL，`/help` 列全部，常用的这几个：

```
/model <名称>    切换模型
/compact         手动压缩上下文
/tokens          查看 token、当前 fallback、费用和剩余预算
/diff            查看本次会话改过的文件
/undo            撤销最近一次文件改动
/plan            开关计划模式（只读摸底，再交出待批准的计划）
/save  /sessions 保存 / 列出会话
/agents          列出后台子 agent
quit / exit      退出（Ctrl+C 取消当前回合）
```

会话 ID 会先清洗成安全字符再拿去当文件名，存档统统落在 `~/.corecoder/sessions` 里，恶意会话名穿越不出去。

## 工具执行进度

工具执行现在有独立的实时视图，不再只是把已发起的调用名逐行打印出来。模型一次返回多个调用时，只有副作用调度器确认其中存在真正并发的只读批次，标题才显示 `Parallel tools`；否则显示 `Tool batch`。每项调用会按 `queued` → `running` → `done`、`error` 或 `blocked` 更新，带耗时和总体 `n/N` 计数。所有 permission 提问都先在主线程处理完，之后才启动进度 UI，因此交互授权不会和 Rich 的实时刷新抢终端。

把 CoreCoder 当库嵌入时，也可以向 `Agent.chat()` 传 `on_tool_progress`，用同一组结构化事件驱动自己的界面：`batch_started`、`tool_started`、`tool_completed`、`batch_completed`。原有的 `on_tool(name, arguments)` 仍保持兼容；进度回调只属于展示层，即使它自己报错，工具和 Agent Loop 也会继续执行。

## 权限

只读工具（`read_file`、`glob`、`grep`、`todo_write`、`now`、`agent_status`）模型一调就跑。会修改状态或访问外部资源的那些（`edit_file`、`write_file`、`bash`、`fetch_url`、MCP 工具以及派生子 agent）先停下来等你点头，REPL 启动横幅里能看到当前是哪种模式：

- REPL 里每次调用问一次：允许这一次、本工具本次会话都允许、或者拒绝。前台子 agent 继承这层交互授权；后台线程绝不会与 REPL 抢输入，只能使用 `--yes` 或已被标记为「总是允许」的工具，其余有状态调用 fail closed，让子 agent 自己绕开。
- 一次性模式（`-p`）没人可问，改动类调用当场被拒，拒绝理由作为普通工具结果回给模型：循环绝不会卡在等一个永远不会来的输入上。要全部预授权就加 `--yes`（脚本、CI 场景）。
- 判断本身是 `permissions.py` 里的纯逻辑，终端只是塞进来一个提问回调。不用 TTY 也能单测授权逻辑，或者直接搬进你自己的嵌入场景。

## Docker 沙箱

为了兼容旧用法，默认仍在本机执行。要给 `bash` 加上容器级边界，先构建项目镜像，再显式开启：

```bash
docker build -f Dockerfile.sandbox -t corecoder-sandbox:latest .
corecoder --sandbox docker
```

每次 bash 调用都会启动一个全新的一次性容器。只有启动 CoreCoder 时的当前目录会以 `/workspace` 挂进去；容器根文件系统只读，网络默认 `none`，全部 Linux capabilities 被丢弃，禁止提权，并设置 CPU、内存、PID 和 `/tmp` 上限。进程使用宿主机 UID/GID，镜像缺失时也不会偷偷联网拉取。内建文件工具同时拒绝解析后落在启动目录之外的路径，包括通常的 `..` 和符号链接逃逸。bash 命令请使用项目相对路径或 `/workspace/...`。

这些默认值可以通过 `CORECODER_SANDBOX*` 环境变量或对应 CLI 参数覆盖：`--sandbox-image`、`--sandbox-network`、`--sandbox-memory`、`--sandbox-cpus`、`--sandbox-pids`。例如 `--sandbox-network bridge` 会显式开放容器网络。自定义镜像只要带 `/bin/sh` 和项目需要的运行时即可。

边界范围需要说清：hooks、MCP server 进程、LLM 客户端和 `fetch_url` 仍在宿主机运行；工作区也必须可写，因为这是 coding agent。宿主机环境变量不会传进容器，但工作区内原本就有的所有文件——包括 `.env` 或凭据——工具仍然看得见；面对不可信任务，应使用干净 worktree，并把秘密放在工作区之外。权限确认仍然有用——permission 决定「是否授权做」，sandbox 限制「即使授权了最多能伤到哪里」。Docker daemon 或镜像不可用时，调用会作为普通工具错误返回，绝不会降级到宿主机执行。

## 计划模式

REPL 里 `/plan` 开关计划模式。开着的时候，提示符变成 `(plan)`，一切改动类调用（写文件、编辑、bash、MCP 工具、子 agent）当场被拒：拒绝理由作为普通工具结果回给模型，让它用只读工具继续摸底，给出一份编号计划。计划看着没问题，敲 `approve`（或再敲一次 `/plan`）就把执行权交回去，agent 接着干活。机制上就是 `Agent` 身上的一个开关，加授权闸前面多出来的一支拒绝分支，授权闸本身一行没动；不落盘计划文件，会话之间也不记任何东西。

## 钩子

在 `~/.corecoder` 下放一个 `hooks.json`，就能让你自己的 shell 命令在每次工具调用前后跑起来，思路和 Claude Code 的 hooks 一致：

```json
{
  "PreToolUse":  [{"matcher": "bash", "command": "cat >> ~/.corecoder/audit.jsonl"}],
  "PostToolUse": [{"matcher": "*",    "command": "cat >> ~/.corecoder/trace.jsonl"}]
}
```

每个钩子从 stdin 拿到这次调用的 JSON（`tool_name`、`tool_input`，post 钩子还带 `tool_response`）。matcher 是精确的工具名，留空或写 `*` 表示对所有工具生效。pre 钩子可以否决这次调用：退出码 2，它的 stderr 会作为理由回给模型，让它换条路走。post 钩子只观察，永远拦不住。钩子报错或超过十秒会被跳过并记一条警告：钩子是来帮忙的，没权力弄死循环。整个机制就是 `hooks.py` 一个文件，REPL 启动横幅会显示加载了几条。

两个能直接抄走的（命令里用了 `jq`，装一下就有）：

```bash
# 1. lint 门：每次 edit/write 之后立刻对改动的文件跑快速 lint。
#    模型同一回合就能看到输出，自己把低级错误修了，不用等 CI 回来。
{
  "PostToolUse": [{
    "matcher": "edit",
    "command": "f=$(jq -r .tool_input.path); ruff check \"$f\" 2>&1 | head -20"
  }]
}

# 2. 写保护：指定路径一律不让 agent 碰。退出码 2 当场否决，
#    拒绝原因会送到模型那边。
{
  "PreToolUse": [{
    "matcher": "edit",
    "command": "case \"$(jq -r .tool_input.path)\" in .env*|*/secrets/*|*.pem) echo 'that path is off-limits' >&2; exit 2;; esac"
  }]
}
```

两个例子都是纯 shell；除了 JSON 形状和退出码 2 否决这两个约定，没有任何 CoreCoder 特有的语法。

## MCP 服务器

在 `~/.corecoder` 下放一个 `mcp.json`，任何 MCP 服务器的工具就能通过 stdio 接进 agent，配置形状和 Claude Code 的一样：

```json
{
  "mcpServers": {
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}
  }
}
```

每个配好的服务器在启动时拉起一个子进程，握手、列出工具；每件工具都注册成 `mcp__<服务器>__<工具>`，钩子匹配和授权闸对它和内建工具一视同仁。MCP 工具不在只读名单里，模型要调，得先问过你。握手给十五秒，一次调用给六十秒；服务器挂了或者迟迟不应，那一次调用就以普通工具结果的形式报错，循环照常往下走。客户端只实现协议里工具那一小片（initialize、tools/list、tools/call），别的一概不碰，所以整块实现收在 `mcp.py` 一个文件里，两百行出头。没有 `mcp.json` 就没有 MCP，一切照旧。

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

动手之前先跑一遍 `pytest tests/ -q`（228 个用例）、`ruff check` 和 `compileall`，绿了再提。MIT License，欢迎 fork 拿去造更好的东西，能在 README 里留一句出处就更好。

---

作者 [何宇峰](https://github.com/he-yufeng)，曾任职 Moonshot AI (Kimi)。早前写过一篇相当完整的 [Claude Code 源码分析](https://zhuanlan.zhihu.com/p/1898797658343862272)，这个项目是它的动手版：那篇带你读懂，这个带你重建。

> CoreCoder 原名 NanoCoder，为避免和 [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder) 混淆而改名，旧链接会自动跳到这里。
