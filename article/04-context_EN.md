# Surviving a long task in a finite window

An agent has one physical constraint it can't get around: the context window is only so big.

And coding tasks happen to be prolific token producers. The model reads a thousand-line file, and those thousand lines, line numbers and all, go into the history; it runs a test, and several hundred lines of output go into the history; it greps once, and dozens of matches go into the history. A halfway-decent task running a dozen-odd rounds burns tens of thousands of tokens. Once the window fills, either the API errors or you have to cut the history, and cut it badly and the agent starts "forgetting": a file it read earlier it reads again, a decision it just made it overturns.

So fitting a long task into a finite window is one of the most hardcore subproblems in agent engineering. This piece looks at how `corecoder/context.py` (431 lines) solves it.

First, clear up a common misconception: the context-window limit does not apply to `messages` alone. The dynamic system prompt, tool schemas, and protocol framing all consume input tokens, and the model still needs room to generate its answer. Looking only at chat history badly overstates the usable space when the tool set or system prompt is large.

## Layered, lightest to heaviest

Claude Code's strategy is four layers in public teardowns, escalating from the cheapest handling to the most aggressive. CoreCoder distills it to three, same idea: space you can save with a cheap means, never spend an expensive means on. The thresholds apply to the budget actually left for conversation history:

```python
message_budget = (
    max_tokens
    - fixed_tokens       # system prompt + tool schemas + request framing
    - output_reserve     # room for this round's completion
    - safety_margin      # estimator slack
)
snip_at = int(message_budget * 0.50)
summarize_at = int(message_budget * 0.70)
collapse_at = int(message_budget * 0.90)
```

`Agent.compress_context()` recomputes the system prompt and tool schemas before each LLM round, uses `estimate_request_tokens()` for fixed overhead, and gives the remaining budget to `ContextManager.maybe_compress()`. Plan mode, todo state, the tool set, and output limits can therefore change the budget dynamically.

`maybe_compress` applies the three quality-preserving layers from lightest to heaviest. Its final `_fit_to_budget` is a hard postcondition, not just a heuristic:

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

Compression runs immediately before each LLM request. Tool execution does not instantly snip its own result: at the start of the next round, the Agent marks those `tool_call_id`s as unconsumed, and ordinary snipping skips them. Protection is cleared only after one LLM request returns successfully. Fresh Tool Results therefore get at least one chance to reach the model intact. If the fresh batch itself cannot fit, the final fitting step bounds it as an explicitly marked observation rather than sending an impossible request.

Token estimation remains dependency-free, but it is no longer one flat characters-per-token ratio. `_approx_tokens` treats CJK, symbol-dense code, and ordinary prose separately; `estimate_tokens` includes per-message framing, `tool_calls`, and `tool_call_id`, while `estimate_tool_schema_tokens` covers schemas:

```python
def _approx_tokens(text: str) -> int:
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    dense = rest > 0 and len(_SYMBOL_RE.findall(text)) / len(text) > 0.25
    return math.ceil(cjk / 1.5) + int(rest / (2.8 if dense else 3.4))
```

This still is not a provider tokenizer, so CoreCoder reserves 5% of the window (at least 64 tokens for small test windows) as estimator slack. The tradeoff is deliberate: avoid provider-specific tokenizers, explicitly account for the common underestimation traps, and use deterministic fitting as the final safety boundary.

## Layer one: old tool outputs have a shelf life

The first layer, `_snip_tool_outputs`, is the cheapest, calling no model, pure text processing. It bounds stale tool results over 1500 characters to useful head and tail text. The check is character-based, so one giant line of JSON is handled too:

```python
content = message.get("content", "")
if len(content) <= 1500:
    continue
if message.get("tool_call_id") in protected_tool_call_ids:
    continue
message["content"] = _truncate_text(content, 1500, "snipped to save context")
```

Behind this layer is an insight I find rather beautiful: tool output has a shelf life.

Those two hundred lines of matches a grep spat out twenty rounds ago were very useful at the time; the model used them to locate the code. But by now the model long ago finished with that result and made the corresponding edit, and those two hundred lines have become pure placeholder garbage, kept only to hog window. Snipping them to a few head and tail lines preserves the clue "a search happened here, roughly these files," while throwing away the vast majority of the dead weight. Fresh information is valuable, stale information is cheap, and compression should compress the stale first. Public teardowns call this layer HISTORY_SNIP, doing the same thing.

Why snip head and tail rather than just the head? Because a command's most useful information is often at both ends: the head is what it's doing, the tail is the result and the error. The big middle of the process can usually be dropped. This "keep head and tail, discard the middle" choice is of a piece with the bash output truncation from the last piece.

## Layer two: have the model write a summary of the old conversation

Layer one only compresses tool output; it can't budge the conversation itself. When the window climbs to 70%, layer two `_summarize_old` steps in: hand the whole old conversation to the model to write a summary, keeping only the most recent 8 messages untouched.

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

The old conversation gets replaced by a single user message saying "this is a summary of the earlier conversation," plus an assistant response of "got it," and then the untouched recent messages are appended. The summary itself is generated by `_get_summary`, whose instruction to the model is tightly focused: keep the file paths that were changed, the key decisions made, the errors encountered, and the current task state; drop the verbose command output, the code listings, and the back-and-forth chatter. This is exactly what genuinely needs to be remembered in a long task.

If there's no model available (or the summary call itself fails), it degrades to `_extract_key_info`, using regex to pull out file paths and lines containing "error" to stitch a crude summary. Graceful degradation again: better a crude summary than letting this compression step drag the whole session down.

## The trap that's bound to bite you: orphaned tool messages

Now the centerpiece of this piece, also a trap I genuinely stepped on while polishing this project.

Recall the iron rule from piece one: an assistant message carrying `tool_calls` must be followed by paired `tool` replies, and the API rejects it if even one is missing. The essence of compression is cutting once at some position in the history, compressing what's before and keeping what's after. Here's the question: what if that cut lands right in the middle of a group of tool calls?

Picture the history as this stretch: assistant initiates tool calls, immediately followed by the corresponding tool replies. If the "keep the most recent N" boundary happens to fall on a tool reply, then the kept tail begins with a tool message while the assistant message that produced it got cut to the front and compressed into the summary, gone. This tool reply is orphaned; there's no matching tool_calls before it. Send that orphan out on the next request and the API rejects it on the spot. Your compression logic, meant to save the day, has killed the session with its own hands.

`_safe_split` exists to prevent this. Before cutting, it walks the boundary backward until the message at the boundary is no longer a tool:

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

Just one `while` loop walking back. The logic couldn't be shorter, but without it compression is a time bomb that stays quiet normally and goes off exactly during a long conversation, with the window tight, at the moment it least should. Both layer two and layer three go through `_safe_split` when cutting, never `len - keep_recent` directly.

This trap is worth remembering, because it has every feature of a "hidden bug": it depends on an invariant spanning multiple messages (a tool must immediately follow its tool_calls), that invariant is written nowhere conspicuous, it doesn't trigger normally, and it only surfaces at the specific moment when "the cut point happens to land in the middle of a tool call." This kind of bug is hard to spot by staring at a single function; you have to hold both "the compression logic" and "the API's pairing constraint" in your head at once to realize they'll clash in some corner. CoreCoder nails this invariant down with two tests: `test_safe_split_never_orphans_a_tool_message` checks the cut point doesn't land on a tool, and `test_compress_never_leaves_an_orphan_tool_reply` checks that after a full round of compression every tool reply still immediately follows its tool_calls. Writing this kind of test is essentially solidifying an invariant hidden in your head into code, so nobody breaks it carelessly later.

## Layer three: the last resort

When the message budget climbs to 90%, layer three `_hard_collapse` is the emergency brake: keep one complete recent group plus a summary and collapse everything else. It likewise goes through `_safe_split` to guarantee no orphans.

Those three layers are a quality strategy; `_fit_to_budget` is the correctness boundary. If history remains too large, it bounds stale Tool Results to 500 characters, compacts oversized assistant tool-call arguments, bounds unconsumed results to 700 characters only if necessary, and finally replaces irreducible history with one emergency state carrying the latest user request, recoverable state, and latest observations. If the fixed system prompt, schemas, output reserve, and safety margin already fill the entire window, `ContextOverflowError` is raised instead of sending a request the provider must reject.

## Compared with Claude Code

The difference between four layers and three is mainly that Claude Code adds a cache-backed micro-compression (microcompact) and a periodic background auto-compression, more refined as engineering. But the core idea, "layered, lightest to heaviest, lazily triggered, compress the stale first," is identical in both. CoreCoder compresses it into three layers, just enough for you to see clearly what each layer solves and what it costs, without drowning in cache and scheduling details.

This piece also answers a question from the opening: why does an agent occasionally "forget"? Because it really does forget; compression is lossy, and the details summarized away are gone. A good compression strategy isn't about losing nothing, it's about losing smart, dropping the things that have run out of shelf life first.

## What this piece leaves you with

- The context window is the agent's hardest physical constraint, and coding tasks are prolific token producers, so hitting the wall is only a matter of time.
- Compression should be layered, lightest to heaviest, lazily triggered: what you can save with pure-text truncation, don't spend an LLM summary on.
- Budget the complete request, not just the conversation: system prompt, tool schemas, protocol framing, and output reserve all count.
- Tool output has a shelf life, and stale output is the priority compression target. Fresh information is valuable, stale information is cheap.
- A fresh Tool Result should reach the model intact at least once; heuristic compression still needs a hard postcondition that the request fits.
- The orphaned tool message is a textbook hidden bug: it depends on a cross-message invariant, doesn't act up normally, and only blows up when the cut point lands in the middle of a tool call. Nailing this kind of invariant down as a test is the right way to fight this class of bug.
- Compression is lossy, and an agent "forgetting" is its inherent cost. A good strategy isn't losing nothing, it's losing smart.

Next piece, we return to another spot skipped in piece one: when the model returns several tool calls at once, how to run them concurrently, and when it can open a sub-agent to share the load.
