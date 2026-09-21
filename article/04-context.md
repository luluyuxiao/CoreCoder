# 用有限的窗口，扛住一个长任务

agent 有一个绕不过去的物理约束：上下文窗口就那么大。

而编码任务偏偏极其能产 token。模型读一个一千行的文件，那一千行连同行号全进了历史；跑一次测试，几百行输出全进了历史；grep 一下，几十个匹配全进了历史。一个稍微像样的任务转上十几轮，几万 token 就没了。窗口一旦塞满，要么 API 报错，要么你得砍历史，而砍历史砍不好，agent 就会「忘事」，前面读过的文件转头又读一遍，刚做过的决定又推翻重来。

所以怎么在有限窗口里装下一个长任务，是 agent 工程里最硬核的子问题之一。这一篇看 `corecoder/context.py`（431 行）怎么解。

先把一个常见误区说清：上下文窗口限制的不是 `messages` 一项，而是整个请求。动态 system prompt、Tool Schema、消息协议开销都占输入 token，模型还必须留出生成答案的空间。只看聊天历史，会在工具多、system prompt 长或者输出上限高的时候严重高估可用空间。

## 分层，从轻到重

Claude Code 的策略公开拆解里是四层，从最廉价的处理逐级升到最激进的。CoreCoder 蒸馏成三层，思路一致：能用便宜手段省出来的空间，绝不动用贵手段。但阈值不是直接乘整个模型窗口，而是先算出真正能留给 conversation 的预算：

```python
message_budget = (
    max_tokens
    - fixed_tokens       # system prompt + Tool Schema + 请求协议开销
    - output_reserve     # 给本轮 completion 留出的空间
    - safety_margin      # 估算器误差余量
)
snip_at = int(message_budget * 0.50)
summarize_at = int(message_budget * 0.70)
collapse_at = int(message_budget * 0.90)
```

`Agent.compress_context()` 每一轮在请求 LLM 前重算 system prompt 和工具 schemas，通过 `estimate_request_tokens()` 得到固定开销，再把剩余预算交给 `ContextManager.maybe_compress()`。这样 plan mode、todo 状态、工具集或输出上限变化后，预算也跟着变。

`maybe_compress` 从轻到重施加三层，每层之后重新估算；最后的 `_fit_to_budget` 是硬后置条件，而不只是启发式建议：

```python
def maybe_compress(
    self, messages, llm=None, *,
    fixed_tokens=0, output_reserve=0,
    protected_tool_call_ids=None,
) -> bool:
    budget = self.available_message_tokens(fixed_tokens, output_reserve)
    current = estimate_tokens(messages)

    if current > int(budget * 0.50):
        self._snip_tool_outputs(messages, protected_tool_call_ids)
        current = estimate_tokens(messages)

    if current > int(budget * 0.70) and len(messages) > 10:
        self._summarize_old(messages, llm, keep_recent=8)
        current = estimate_tokens(messages)

    if current > int(budget * 0.90) and len(messages) > 4:
        self._hard_collapse(messages, llm)

    if estimate_tokens(messages) > budget:
        self._fit_to_budget(messages, budget, protected_tool_call_ids)

    if estimate_tokens(messages) > budget:
        raise ContextOverflowError(...)
```

压缩只放在每次 LLM 请求之前。工具执行完不会立刻截它的结果；下一轮开始时，Agent 把这些 `tool_call_id` 标记为「尚未消费」，普通 snip 会跳过它们。只有一次 LLM 请求成功返回后，这批保护才解除。于是最新 Tool Result 至少有一次机会完整进入模型，而不是刚执行完就被上下文管理器截掉。如果最新结果自身大到整个批次无法放进窗口，最终适配步骤才会把它压成带明确标记的有界 observation。

token 估算仍然保持零依赖，但不再是统一的「字符数除以 3」。`_approx_tokens` 分别处理 CJK、符号密集的代码和普通文本；`estimate_tokens` 还计算每条消息、`tool_calls` 和 `tool_call_id` 的协议开销，`estimate_tool_schema_tokens` 则单独计算 schemas：

```python
def _approx_tokens(text: str) -> int:
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    dense = rest > 0 and len(_SYMBOL_RE.findall(text)) / len(text) > 0.25
    return math.ceil(cjk / 1.5) + int(rest / (2.8 if dense else 3.4))
```

它依然不是 provider tokenizer 的精确计数，所以额外保留窗口的 5%（小窗口至少 64 token）作为安全余量。这里的取舍是：不引入 provider 专属 tokenizer，但把最容易低估的中文、代码、Tool Schema 和协议开销显式算进去，再用硬适配保证估算值不会越界。

## 第一层：旧的工具输出，是有保质期的

第一层 `_snip_tool_outputs` 是最便宜的，不调模型，纯文本处理。它把超过 1500 字符的旧工具结果压成有界的头尾片段；按字符而不是按行判断，因此一整行的超大 JSON 也能被处理：

```python
content = message.get("content", "")
if len(content) <= 1500:
    continue
if message.get("tool_call_id") in protected_tool_call_ids:
    continue
message["content"] = _truncate_text(content, 1500, "snipped to save context")
```

这一层背后有个我觉得很漂亮的洞察：工具输出是有保质期的。

二十轮之前那次 grep 吐出来的两百行匹配，在当时很有用，模型靠它定位了代码。但到了现在，模型早就用完那个结果、做完了相应的修改，那两百行就成了纯粹的占位垃圾，留着它只是白占窗口。把它截成头尾几行，既保留了「这里曾经查过一次、大致是这些文件」的线索，又把绝大部分死重量扔掉了。新鲜的信息值钱，陈旧的信息廉价，压缩就该优先压陈旧的。公开拆解里管这层叫 HISTORY_SNIP，干的是同一件事。

为什么截头尾而不是只截开头？因为命令输出最有用的信息常常在两端：开头是它在干什么，结尾是结果和报错。中间那一大坨过程往往可以丢。这个「留头尾、弃中间」的选择，和上一篇 bash 输出截断的逻辑是一脉相承的。

## 第二层：让模型给旧对话写个摘要

第一层只压工具输出，压不动对话本身。当窗口涨到 70%，第二层 `_summarize_old` 上场：把旧的对话整段交给模型写一个摘要，只留最近 8 条消息原样不动。

```python
split = self._safe_split(messages, keep_recent)
old = messages[:split]
tail = messages[split:]

summary = self._get_summary(old, llm)

messages.clear()
messages.append({
    "role": "user",
    "content": f"[Context compressed - conversation summary]\n{summary}",
})
messages.append({
    "role": "assistant",
    "content": "Got it, I have the context from our earlier conversation.",
})
messages.extend(tail)
```

旧对话被替换成一条「这是之前对话的摘要」的 user 消息，外加一条模型「我记下了」的 assistant 回应，然后接上原封不动的近期消息。摘要本身由 `_get_summary` 生成，它给模型的指令很聚焦：保留改过的文件路径、做过的关键决定、遇到的错误、当前任务状态；丢掉啰嗦的命令输出、代码清单、来回的废话。这正是一个长任务里真正需要被记住的东西。

如果没有可用的模型（或者摘要调用本身失败了），它退化成 `_extract_key_info`，用正则把文件路径和带 error 的行抽出来拼一个粗摘要。又是优雅降级：宁可给个糙摘要，也不让压缩这一步把整个会话拖垮。

## 那个一定会咬你的坑：孤儿 tool 消息

现在讲这一篇的重头戏，也是我在打磨这个项目时实打实踩过的坑。

回想第一篇那条铁律：一条带 `tool_calls` 的 assistant 消息，后面必须跟着配对的 `tool` 回复，少一个 API 都会拒。压缩这件事的本质，是在历史的某个位置切一刀，前面的压掉、后面的留下。问题来了：如果这一刀正好切在一组工具调用的中间呢？

设想历史是这样一段：assistant 发起了工具调用、紧跟着是对应的 tool 回复。如果「保留最近 N 条」这个边界恰好落在 tool 回复上，那么被保留的尾巴就会以一条 tool 消息开头，而产生它的那条 assistant 消息被切到前面、压进摘要里没了。这条 tool 回复成了孤儿，它前面找不到对应的 tool_calls。下一次请求带着这个孤儿发出去，API 当场拒绝。你的压缩逻辑本来是来救场的，结果亲手把会话搞死了。

`_safe_split` 就是来防这个的。它在切分前，把边界往前挪，直到边界那条消息不再是 tool：

```python
@staticmethod
def _safe_split(messages: list[dict], keep_recent: int) -> int:
    """Index where the kept tail should start.

    Walk the boundary back so a 'tool' result is never separated from the
    assistant message whose tool_calls produced it - an orphaned tool
    message has no preceding tool_calls and OpenAI-compatible APIs reject it.
    """
    split = max(0, len(messages) - keep_recent)
    while split > 0 and messages[split].get("role") == "tool":
        split -= 1
    return split
```

就这么一个 `while` 循环往回退。逻辑短得不能再短，但少了它，压缩就是个定时炸弹，平时不响，偏在长对话、窗口吃紧、最不该出事的时候炸。第二层和第三层切分时都走 `_safe_split`，不直接用 `len - keep_recent`。

这个坑值得你记住，因为它有一切「隐蔽 bug」的特征：它依赖一个跨越多条消息的不变式（tool 必须紧跟 tool_calls），这个不变式没写在任何一处显眼的地方，平时也不触发，只在「切分点恰好落在工具调用中间」这个特定时机才暴露。这种 bug 你很难靠盯着单个函数看出来，得在脑子里同时装着「压缩逻辑」和「API 的配对约束」两件事，才能意识到它们会在某个角落打架。CoreCoder 用两个测试把这个不变式钉死了，`test_safe_split_never_orphans_a_tool_message` 验切分点不落在 tool 上，`test_compress_never_leaves_an_orphan_tool_reply` 验整轮压缩后每条 tool 回复都还紧跟着它的 tool_calls。写这种测试，本质上是把一条「藏在脑子里的不变式」固化成代码，让它以后别再被人不小心破坏。

## 第三层：最后手段

消息预算涨到 90%，说明前两层都没压够，第三层 `_hard_collapse` 是急刹车，只保留最后一组完整消息加一个摘要，其余全部折叠掉。它同样走 `_safe_split` 保证不留孤儿。

三层是质量策略，`_fit_to_budget` 则是正确性边界。如果 hard collapse 后仍超预算，它会依次把陈旧 Tool Result 压到 500 字符、压缩 assistant `tool_calls` 里的超大参数、必要时把尚未消费的结果压到 700 字符，最后才用一条包含最新用户请求、可恢复状态和最新 observations 的 emergency state 替换不可约历史。若连固定的 system prompt、Tool Schema、输出预留和安全余量都已经塞满窗口，则直接抛出 `ContextOverflowError`，而不是发送一个注定会被 provider 拒绝的请求。

## 和 Claude Code 的对照

四层和三层的差别，主要在 Claude Code 多了一层带缓存的微压缩（microcompact）和周期性的后台自动压缩，工程上更精细。但「分层、从轻到重、惰性触发、优先压陈旧信息」这套核心思路，两者完全一致。CoreCoder 把它压成三层，刚好够你看清每一层在解决什么、代价是什么，而不至于淹没在缓存和调度的细节里。

这一篇其实也回答了一个开头的问题：为什么 agent 偶尔会「忘事」。因为它真的会忘，压缩就是有损的，被摘要掉的细节就是丢了。好的压缩策略不是不丢，而是丢得聪明，优先丢那些已经没有保质期的东西。

## 这一篇带走什么

- 上下文窗口是 agent 最硬的物理约束，编码任务又极其能产 token，撞墙是迟早的事。
- 压缩要分层、从轻到重、惰性触发：能用纯文本截断省出来的，就别动用 LLM 摘要。
- 预算要覆盖完整请求，而不只是 conversation：system prompt、Tool Schema、协议开销和输出预留都必须计入。
- 工具输出有保质期，陈旧的输出是优先压缩对象。新鲜信息值钱，陈旧信息廉价。
- 最新 Tool Result 应至少被模型完整消费一次；启发式压缩之后还需要一个保证请求能放进窗口的硬后置条件。
- 孤儿 tool 消息是个典型的隐蔽 bug：它依赖一条跨消息的不变式，平时不发作，只在切分点落在工具调用中间时炸。把这种不变式写成测试钉死，是对抗这类 bug 的正道。
- 压缩是有损的，agent「忘事」是它的固有代价。好策略不是不丢，是丢得聪明。

下一篇，我们回到第一篇里被跳过的另一处：当模型一次返回好几个工具调用，怎么并发地跑，以及它什么时候能开一个子 agent 替自己分担。
