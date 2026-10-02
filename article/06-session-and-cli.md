# 把它跑成一个真正的命令行工具

前五篇拆的是 agent 的内脏：循环、工具、模型接口、上下文压缩、并发与子 agent。这些零件再漂亮，你也没法直接用，因为还缺一层皮，一个让人能坐下来跟它对话、能存盘、能续聊、能查状态的命令行界面。这一篇讲这层皮，对应 `cli.py`、`session.py` 和 `storage.py`。

它不光是「锦上添花的 UI」。这层皮里藏着一个很值得讲的安全细节，我们留到后面压轴。

## REPL：一个对话循环套着一个对话循环

`cli.py` 的主体是 `_repl`，一个 `while True`。注意它和第一篇那个 agent 主循环是两层不同的循环：这里这层是「读用户输入、交给 agent、打印回复、再读下一句」，是人机交互的循环；agent 内部那层是「问模型、跑工具」，是任务执行的循环。一次人类输入，往往触发 agent 内部转好几圈。

输入这块用了 `prompt_toolkit`，带历史记录，还自定义了一个挺贴心的键位：

```python
@kb.add("enter")
def _submit(event):
    event.current_buffer.validate_and_handle()

@kb.add("escape", "enter")
def _newline(event):
    event.current_buffer.insert_text("\n")
```

回车直接提交，Esc 加回车才是换行。这样你贴一段多行代码进去，不会刚贴到第二行就被提前提交了。一个小细节，但用过就回不去。

拿到用户输入后，真正调 agent 的是这几行，它把第一篇和第三篇的两个回调在这里落了地：

```python
def on_token(tok):
    streamed.append(tok)
    print(tok, end="", flush=True)

def on_tool(name, kwargs):
    console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

response = agent.chat(user_input, on_token=on_token, on_tool=on_tool)
```

`on_token` 让模型的文字一个个实时冒出来，就是第三篇流式那一层在界面上的样子。`on_tool` 在每次调工具前打一行灰字，告诉你「它正要去读哪个文件、跑哪条命令」。这两个回调是 agent 内核和界面之间唯一的耦合点，内核不关心你怎么显示，只在该出事件的时候喊一声，界面爱怎么呈现是界面的事。这种「内核出事件、外壳管呈现」的切分很干净，你想把 CoreCoder 嵌进别的程序（比如一个 web 服务），只要换掉这两个回调就行，内核一行不用动。第七篇会用到这个性质。

## 斜杠命令：在不打断对话的前提下管状态

REPL 里认一批斜杠命令，它们不发给模型，而是直接操作 agent 的状态：

```
/help      显示帮助
/reset     清空对话历史
/model     查看或切换模型
/tokens    显示 token 用量和估算花费
/compact   手动触发上下文压缩
/diff      列出本次会话改过的文件
/save      立即保存当前会话
/session   查看当前会话和存储位置
/sessions  列出已存的会话
/transcript [id]  查看不受上下文压缩影响的原始事件历史
/delete-session <id>  删除非当前会话
```

这些命令把前几篇讲的内核能力暴露成了用户能直接拨的开关。`/tokens` 调的是第三篇那个 `estimated_cost`，`/compact` 调的是第四篇那个 `maybe_compress`，`/diff` 读的是第二篇 `edit_file` 一直在维护的那个「改过的文件」集合。内核早就把这些能力准备好了，CLI 只是给每个能力配了一个顺手的入口。

这里有个不起眼但体现品味的判断。一句以 `/` 开头、却不在上面名单里的输入，该怎么办？

```python
# an unknown /command shouldn't be sent to the model as a prompt
if user_input.startswith("/"):
    console.print(f"[yellow]Unknown command: {user_input.split()[0]} (try /help)[/yellow]")
    continue
```

它不会把 `/qiut`（手滑拼错的 quit）当成一句话发给模型，而是提示「没这个命令」。要是没这道拦截，你打错一个斜杠命令，模型会一本正经地把它当任务来理解，浪费一轮调用还可能干出莫名其妙的事。把「用户显然是想敲命令但敲错了」和「用户真的在给任务」区分开，是交互设计里很小但很真实的体贴。

## 一次性模式：让它能被脚本调用

除了交互式 REPL，CLI 还有个 `-p` 一次性模式，跑一个 prompt 就退出，方便塞进脚本或者管道：

```python
def _run_once(agent, prompt):
    try:
        agent.chat(prompt, on_token=on_token, on_tool=on_tool)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
```

注意退出码。被 Ctrl+C 打断退 130（这是 Unix 下「被信号中断」的惯例退出码），出错退 1，正常退 0。一个想被脚本调用的命令行工具，必须把退出码当回事，因为调用方就靠这个数字判断你成没成。这又是那种「demo 不会管、产品必须管」的细节。

## 压轴：持久化不能留下半截 Agent 协议

现在讲 `session.py` 和 `storage.py` 里最值得记住的两件事：不信任 Session ID，以及只保存能合法恢复的消息结构。

早期版本把 `messages` 和模型名 dump 成 JSON，当前版本默认存进 SQLite，并保留原子 JSON 后端和旧存档迁移。无论落点是数据库 key 还是旧版文件名，Session ID 都可能来自用户：`corecoder -r <id>` 里的 `<id>` 是任意命令行输入。

设想最朴素的实现：`SESSIONS_DIR / f"{session_id}.json"`。如果用户（或者某个喂给它会话名的上游程序）把 id 设成 `../../etc/passwd` 会怎样？这个路径会解析到会话目录之外，你的「存会话」变成了「往任意位置写文件」，「读会话」变成了「读任意文件」。这是经典的路径穿越漏洞，无数真实系统栽在它上面。

CoreCoder 用两道关来防，纵深防御。第一道，把 id 规整成一个安全的纯文件名：

```python
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")

def normalize_session_id(session_id):
    if not session_id:
        return new_session_id()
    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    if len(name) > _MAX_SESSION_ID_LEN:
        name = name[:_MAX_SESSION_ID_LEN].strip(".-_")
    return name or new_session_id()
```

它先把反斜杠统一成正斜杠（堵住 Windows 的 `..\..\` 写法），再用 `split("/")[-1]` 只取最后一段，把所有目录部分全扔掉。于是 `../../etc/passwd` 取到 `passwd`，`/etc/shadow` 取到 `shadow`。然后把剩下的字符里凡不是字母数字和 `._-` 的，全替换成 `-`，再砍掉过长的部分。一个恶意路径进去，出来就是一个老老实实待在会话目录里的普通文件名。

兼容 JSON 后端还有第二道关。`JsonSessionStore._path` 把最终路径解析出来后，明确检查它的父目录就是会话目录本身：

```python
def _path(self, session_id):
    path = (self.directory / f"{normalize_session_id(session_id)}.json").resolve()
    if path.parent != self.directory:
        raise ValueError("Invalid session id")
    return path
```

`resolve()` 会把路径里所有的 `..` 和符号链接都摊平成真实绝对路径，然后一句 `root != path.parent` 卡死：只要最终落点不是直接躺在会话目录里，立刻拒绝。

为什么要两道关，第一道不是已经够了吗？因为安全这件事，单点防御是脆弱的。第一道是基于「净化输入」的，万一哪天有人改了那个正则、漏了一种攻击写法，第二道基于「校验输出落点」的关还能兜住，反过来也一样。两道关用的是完全不同的思路（一个管输入、一个管输出），所以它们不会一起失效。这就是纵深防御的精髓：不指望任何单一防线绝对可靠，而是叠几道原理不同的防线，让攻击者得同时骗过所有人。这套防御被一串测试盯死，路径穿越、绝对路径、Windows 反斜杠、超长名字，逐个验证攻击字符串进去都变成了乖乖的文件名。

顺带一提，`JsonSessionStore.load` 对坏文件也很克制：

```python
try:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SessionRecord(...)
except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
    return None
```

JSON 写入也不再直接覆盖目标文件，而是同目录临时文件 `flush + fsync` 后用 `os.replace` 原子替换。一个写到一半断电、内容截断的旧存档，不该让你下次 `-r` 续聊时直接崩在脸上；读取失败就返回 `None`。SQLite 是默认后端，使用 WAL、`busy_timeout`、外键和 `PRAGMA user_version`，为并发读写、事务与将来的 schema migration 留出空间。

更关键的是把历史审计和模型上下文分开。`events` 是 append-only Transcript：原始 user/assistant/tool call/tool result 一经写入就不因压缩改变；`messages` 是 Active Context，Context Manager 可以把它截短、总结或硬折叠，Resume 和下一轮 LLM 只加载这一份。每次稳定保存会在同一个事务里 upsert Session、替换 Active Context、追加尚未落盘的 Transcript Events，并记录新的摘要，因此上下文变短不会让原文永远消失，重复 autosave 也会靠 event id 去重。

事务边界仍然尊重 Provider 协议。带 `tool_calls` 的 assistant 消息如果没有完整的 Tool Result 配对，OpenAI-compatible provider 会直接拒绝。`Agent` 因此不把每次 `append` 都当可恢复点，而只在这些稳定边界触发 storage-neutral `state_callback`：用户消息进入后、整个工具批次全部回填后，以及最终完成、失败或中断时。任何一步失败都会回滚到上一个稳定 Active Context 和 Transcript 尾部。

Active Context 快照除了 messages 和 model，还保存 workspace、status、token/cost、fallback usage、plan mode、todo 和尚未被下一轮 LLM 消费的 Tool Result IDs。恢复时只加载这份精简上下文和运行状态，而不是把完整 Transcript 重新塞回模型；因此 `/tokens` 不会归零，Todo 不会消失，新鲜 observation 仍受 Context Manager 保护。`summaries` 另外记录压缩动作、覆盖到的事件序号、前后 token、生成模型与摘要消息。API Key 和 `always allow` 权限故意不保存：前者是秘密，后者跨进程继承会把一次会话内授权升级成永久授权。

CLI 默认自动保存到 `~/.corecoder/sessions/sessions.db`，`--no-autosave` 只关闭自动写入、不影响 `/save`；`--storage PATH` 与 `CORECODER_STORAGE_PATH` 可以改位置。`/transcript [id]` 可以检查完整事件历史。旧 v1/JSON 会在首次读取时迁移进 SQLite；旧数据已经丢失的压缩前内容无法恢复，所以 `transcript_complete` 会保持 false。作为库使用时，`SessionStore` Protocol 让调用者换掉后端，而 Agent 只认识快照回调，不依赖 `sqlite3`。

## 和 Claude Code 的对照

Claude Code 的会话与查询引擎（公开拆解里上千行）远比这复杂，它还要处理远程同步、分支会话和更多终端状态。但 CoreCoder 已经把本地 Agent 最关键的持久化不变量摆在明面上：可插拔后端、事务快照、稳定消息边界、旧格式迁移、并发写入和敏感状态取舍。

## 这一篇带走什么

- CLI 是内核之上的一层壳。内核出事件（`on_token`、`on_tool`），外壳管呈现，两者只通过回调耦合，换壳不用动内核。
- 斜杠命令把内核能力暴露成用户能直接拨的开关；拦住敲错的斜杠命令，别把它当任务发给模型。
- 想被脚本调用，就得认真对待退出码：中断 130、出错 1、正常 0。
- 会话 id 会从用户来，路径穿越是真实威胁。用纵深防御应对：一道净化输入，一道校验落点，两道思路不同所以不会一起失效。
- Transcript 面向审计，Active Context 面向模型预算；两者生命周期不同，不能共享一份会被原地压缩的 messages。
- Agent 存储的事务边界必须尊重协议边界：一组 `tool_calls + tool results` 要完整提交，不能留下半截可恢复状态。
- JSON 原子替换解决单文件损坏；SQLite WAL、事务和 schema version 解决查询、并发与演进，两者解决的层次不同。
- 持久化要做安全取舍：状态和 usage 可以恢复，密钥与跨进程授权不该默认落盘。

下一篇是收尾，也是最实操的一篇：把这六篇拆开看过的零件重新装起来，带你 fork CoreCoder，改出一个真正属于你自己的 coding agent。
