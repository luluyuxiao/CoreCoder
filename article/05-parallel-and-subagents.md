# 并行执行与子 agent

第一篇讲主循环时，我在「执行工具」那一步留了个口子：模型一次只要一个工具，和一次要好几个工具，走的是两条不同的路。这一篇先把这个口子补上，再讲一个让 agent 能给自己「分身」的工具。两件事其实是同一个主题的两面：怎么让一个 agent 同时处理多件事，而不互相搞乱。

## 一次回来好几个工具调用

模型并不总是一次只要一个工具。你让它「看看这三个文件分别写了啥」，它很可能一口气返回三个 `read_file` 调用。这三个读操作彼此独立，串行跑就是干等；但如果同一批里还有 `write_file` 或 `bash`，把整批一股脑并发又会产生顺序和竞态问题。

所以 CoreCoder 不是简单地按「一个/多个」决定串行或并行，而是让每个 `Tool` 声明粗粒度副作用：

```python
class ToolEffect:
    PURE = "pure"
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

class Tool(ABC):
    effect = ToolEffect.UNKNOWN

    def is_concurrency_safe(self) -> bool:
        return self.effect in {ToolEffect.PURE, ToolEffect.READ}
```

`read_file`、`glob`、`grep`、`agent_status` 标成 `READ`，`now` 标成 `PURE`；`write_file`、`edit_file`、`todo_write` 是 `WRITE`；`bash`、`fetch_url`、`agent` 是 `EXTERNAL`。第三方自定义工具和 MCP Tool 如果没有可信的副作用声明，就沿用默认的 `UNKNOWN`。默认未知而不是默认可并发，是一个 fail-closed 选择：宁可少一点并行，也不赌外部工具没有共享状态或副作用。

主循环仍然在模型一次返回多个调用时进入 `_exec_tools_parallel`，但这个名字现在表达的是调度入口，不是「所有调用都并发」。把实际代码压成伪代码后是：

```python
results = [self._pre_hooks(tc) or self._permit(tc) for tc in tool_calls]
pending = [i for i, result in enumerate(results) if result is None]

cursor = 0
while cursor < len(pending):
    if current_tool.is_concurrency_safe():
        batch, cursor = take_consecutive_safe_calls(pending, cursor)
        run_in_thread_pool(batch, max_workers=8)
    else:
        run_exclusively(current_call)
        cursor += 1
```

调度器只把**连续的** `PURE/READ` 调用组成一个最多 8 个线程的批次。`WRITE/EXTERNAL/UNKNOWN` 都是独占屏障：它以及它的 post hook 完成之后，后面的调用才能开始。因此 `read(A), read(B), write(C), read(D)` 会按「A/B 并发 → C 串行 → D」执行，工具结果仍按模型原始顺序写回 conversation。

pre hooks 和 permission 仍先在主线程逐个处理，避免多个 worker 同时向一个终端提问；只有真正获准的调用进入调度器。被 hook 或 permission 拒绝的调用已经有一条文本结果，不会执行，也不会触发 post hook。

CLI 通过 `Agent.chat(..., on_tool_progress=...)` 观察这套调度。所有 permission 决策结束后先收到一次 `batch_started`，每个调用再发 `tool_started` 和 `tool_completed`（并发时可能来自 worker thread），最后是 `batch_completed`。这个顺序让 Rich 可以稳定显示 queued/running/done/error/blocked、单项耗时与总体 `n/N`，又不会和交互式授权提示抢终端。只有获准调用里确实存在会重叠执行的批次时，标题才叫 `Parallel tools`；只是数量多但实际串行时显示 `Tool batch`。回调只是展示层：CLI 自己保证线程安全，Agent 也会吞掉回调异常，因此进度 UI 坏了不会连带弄死工具调用。

线程而不是进程或协程，是因为这里允许并发的文件读取和搜索主要是 IO 密集型；线程池是标准库里足够轻量的实现。但这个实现仍然保守：它不知道两个写调用是否落在完全不同的文件，也不知道两个 MCP 工具是否都只读。后续若要继续提速，可以把 `effect` 扩成资源级 read/write set、按路径加锁或构建依赖 DAG。

## 这个简化，相对更完整的调度器差在哪

第一，CoreCoder 会等流式响应完整结束、tool calls 全部组装好后才执行，没有做到边生成边投机启动。第二，当前 `ToolEffect` 只描述粗粒度类别，不能证明「两个写操作互不冲突」，也没有利用 MCP annotations 做可信策略。它换来的好处是规则简单、顺序确定，而且未知工具不会被误并发。

## 并行不是免费的：共享可变状态会咬人

副作用分类解决的是 Agent 内一批 tool calls 的调度问题，工具自身仍要为可能的多线程使用负责。例如两个 Agent 可以共享同一个工具实例，测试或库调用者也可以直接从线程池调用工具。

`BashTool` 需要记住连续命令之间的 `cd`，因此它没有把 cwd 放进一个全局字符串，而是把 `threading.local()` 放在实例上：

```python
self._local = threading.local()
cwd = getattr(self._local, "cwd", None) or str(default_cwd)
# 成功执行 cd 后：
self._local.cwd = running
```

当前调度器把 `bash` 标成 `EXTERNAL`，同一批 bash 不会互相并发；实例级、线程级 cwd 隔离则是第二层防护，避免不同 Agent 或直接并发调用时共享一份目录状态。关键原则仍然是：**给 agent 加并发，就同时给所有带可变状态的工具加了并发正确性要求。** `ToolEffect` 决定调度器敢不敢并发，工具内部隔离或锁决定它在其他并发入口下是否仍然正确，两者不能互相替代。

## 子 agent：给自己开一个分身

`agent` 工具解决的是另一个问题。有些子任务很重，比如「把这个陌生代码库摸一遍，告诉我认证是怎么实现的」。这种活如果让主 agent 自己干，它得读一大堆文件、跑一堆搜索，这些中间过程全堆进主对话的窗口，等它摸清楚了，窗口也被探索垃圾塞得差不多了，真正的任务反而没空间了。

子 agent 的思路是：派一个有独立上下文的分身去干这件重活，它在自己的窗口里折腾，干完只把一个精简结论交回来。主 agent 的窗口始终干净，只多了一句「认证是这么实现的」。

当前 `agent` 把两个问题拆成了两个正交参数：

```python
agent(
    task="分析认证模块并报告",
    run_mode="foreground" | "background",
    isolation="shared" | "worktree",
)
```

`run_mode` 回答「父 Agent 要不要等」：`foreground` 保持原来的同步行为，`background` 在 daemon thread 里跑并立即返回 task ID。`isolation` 回答「在哪份代码上干活」：`shared` 使用当前 checkout，`worktree` 从父仓库当前 `HEAD` 建 `corecoder/subagent-<id>` 分支和独立 worktree。两者可以任意组合，所以后台任务也能待在独立 worktree 里。

无论哪种模式，主 agent 都新建一个 `Agent`，所以 child 有全新的 `messages` 与 `ContextManager`。它共享父 agent 的 LLM、permission 和 hooks，但内建工具会重新实例化：自己的 `BashTool` cwd、自己的 todo 列表；worktree 模式下，文件工具还会被强制绑定到新工作区。模型总账仍合在一起，为避免父子并发改坏 token/美元预算状态，模型调用共用一把锁；工具阶段仍然可以重叠。

子 agent 的轮次上限是 20，最终文本超过 5000 字符会被截断。失败也变成普通文本结果，不会沿调用栈炸掉父 agent。

## 后台任务怎样回来

后台 `agent` 调用不会假装自己已经完成，而是返回 task ID。新增的只读工具 `agent_status` 承担查询：不给 ID 就列出保留的任务；给 ID 可立即轮询，或者用 `wait_seconds` 最多等 60 秒。REPL 的 `/agents` 也走同一个状态源。

```python
started = agent(task="跑完整测试", run_mode="background")
# => Task ID: 8f31...
agent_status(task_id="8f31...", wait_seconds=30)
```

任务表有两道界限：最多同时 4 个后台 child，最多保留 32 条状态；后台线程是进程内 daemon，不是独立服务，CoreCoder 一退出它就不保证继续运行。这叫「后台执行」，不叫「持久化作业系统」。

后台还有一个容易忽略的终端竞态：worker 若沿用交互 permission，可能在用户正输入时突然抢走 stdin。因此后台 child 只继承 `--yes` 或已经做出的 `always allow`；其他改动类调用直接拒绝成普通 tool result。要临时逐次确认，就用 foreground。

## worktree 隔离到底隔离了什么

worktree 从 `HEAD` 创建，因此父 checkout 的未提交修改不会被暗中复制；结果明确返回 worktree 路径、分支名和 `git status --short`，目录会保留给用户检查、提交、merge 或 cherry-pick。它不是安全沙箱：local bash 仍能访问宿主机，hooks、MCP/custom tools 也没有通用的「重绑到 worktree」协议。真正的执行边界仍要靠 Docker sandbox；worktree 解决的是并行改代码时互不踩文件。

如果父 agent 使用 Docker executor，child 会复制相同的镜像、网络和资源限制，只把挂载根换成 worktree；绝不会为了 worktree 悄悄降级成本地 shell。无法重绑的自定义 command executor 会 fail closed。

## 为什么子 agent 不准生孙 agent

派生子 agent 时，工具集会同时过滤 `agent` 与 `agent_status`。于是 child 既不能继续开分身，也不会拿到一个指向父任务表的孤立 status 工具；它干不了的事只能自己处理，不能再往下派。

为什么要禁？因为递归的 agent 是一颗随时会失控的炸弹。设想不禁会怎样：主 agent 派了个子 agent，子 agent 觉得任务还是太大又派了孙 agent，孙 agent 再派……每一层都在烧 token、占线程、加延迟，而且模型对「这个子任务该不该再拆」的判断并不可靠，它完全可能陷进一个越拆越细、永远收不拢的无底洞。一刀切死递归，是最省心的安全策略：分身只能有一层，要么这个子 agent 自己搞定，要么它失败返回，没有第三种走向。

还记得第一篇强调过 `_tool_by_name` 是实例级的吗？正是因为每个 agent 认的工具是它自己的事，这里的能力裁剪才真正生效。child 即便在某段文字里看到过 `agent` 这个名字，去调也只会得到 unknown tool。

## 和 Claude Code 的对照

CoreCoder 现在覆盖了前台/后台和共享/worktree 这两个最能说明架构的维度，但仍不是生产级调度器：没有持久化 worker、跨进程恢复、取消、事件流、自动合并策略，也没有预设 agent 类型。刻意留下这些边界，比把一个 daemon thread 宣传成完整多 agent 平台更诚实。

「用独立上下文隔离重活、保护主窗口」仍是第一动机；background 才在此之上增加时间并行，worktree 再增加代码状态隔离。三种隔离——context、调度、checkout——不要混成一个词。

## 这一篇带走什么

- `Tool.effect` 把 permission 和并发安全分开：前者回答「准不准执行」，后者回答「获准后能不能同时执行」。
- 只有连续的 `PURE/READ` 调用进入线程池；`WRITE/EXTERNAL/UNKNOWN` 都是独占屏障，未知工具默认串行。
- 并行不是免费的：调度策略之外，带可变状态的工具仍要用实例隔离、线程局部状态或锁保证自身正确。
- 子 agent 的首要价值是上下文隔离，让重活在独立窗口里折腾、只把精简结论交回主对话，其次才是任务分解。
- `run_mode` 与 `isolation` 正交：前台/后台决定等待方式，共享/worktree 决定 checkout。
- 后台权限不碰 stdin，worktree 不冒充 sandbox，失败和超长结果都在父循环边界内收口。
- 子 agent 不准递归，是用「一刀切」换「绝不失控」。它靠实例级工具集落地。

下一篇，我们把这些零件装进一个真正能用的命令行工具：会话怎么存、断点怎么续、斜杠命令怎么接，以及一个藏在存盘里的安全细节。
