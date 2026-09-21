# 接入任意大模型，顺便把钱算清楚

前两篇讲的是循环和工具，也就是 agent 的手脚。这一篇讲大脑接口：模型怎么接进来，流式输出怎么处理，provider 抽风了怎么扛，以及一个被很多教程跳过、但你上线后第一天就会关心的问题，这一轮到底花了多少钱。

对应的文件是 `corecoder/llm.py`，498 行，是整个项目最大的单文件。它大，是因为它替你扛下了和真实 API 打交道时所有不优雅的部分。

## 一个赌注：大家都长得像 OpenAI

`llm.py` 开头的注释把整个设计的赌注讲明白了：

> 既然大多数 provider（DeepSeek、Qwen、Kimi、GLM、Ollama 等）都暴露了 OpenAI 兼容的接口，我们就直接用 openai SDK。换 provider 只需要改 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，没了。

这个赌注在 2026 年基本是稳赢的。OpenAI 的 `/v1/chat/completions` 接口形状已经成了事实标准，国内外绝大多数模型服务要么原生兼容，要么提供一个兼容端点。所以 CoreCoder 的主力 `LLM` 类，本质上就是 openai 官方 SDK 外面薄薄一层包装。你想从 OpenAI 换到 DeepSeek，不改一行代码，改两个环境变量：

```bash
export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com CORECODER_MODEL=deepseek-chat
```

这个「一套接口接住所有人」的选择，是 CoreCoder 能这么小的重要原因之一。它没有为每家 provider 写一个 adapter，因为它赌大家都会向 OpenAI 的形状靠拢。

那不兼容的怎么办？比如 AWS Bedrock、Google Vertex 这些。`llm.py` 末尾有一个 `LiteLLM` 子类兜底，后面会讲。先看主路径。

## 流式输出，比你想的麻烦一点

`LLM.chat()` 把消息发出去，开了 `stream=True`，然后一块一块地收。文本好办，来一块拼一块。真正麻烦的是工具调用，因为工具调用的参数也是流式吐出来的，一个调用的 JSON 参数会被切成好几个碎片，分散在多个 chunk 里到达，你得自己把它们按调用的次序缝回去。

CoreCoder 用一个以 index 为键的字典来缝：

```python
tc_map: dict[int, dict] = {}  # index -> {id, name, arguments_str}

for chunk in stream:
    # ...
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            if idx not in tc_map:
                tc_map[idx] = {"id": "", "name": "", "args": ""}
            if tc_delta.id:
                tc_map[idx]["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    tc_map[idx]["name"] = tc_delta.function.name
                if tc_delta.function.arguments:
                    tc_map[idx]["args"] += tc_delta.function.arguments
```

注意那个 `+=`。参数字符串是累加上去的，因为它一次只到一截。等流收完，再把每个调用累积的参数字符串 `json.loads` 成真正的字典：

```python
for idx in sorted(tc_map):
    raw = tc_map[idx]
    try:
        args = json.loads(raw["args"])
    except (json.JSONDecodeError, KeyError):
        args = {}
    parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))
```

这里有个防御：如果累积出来的参数串不是合法 JSON（模型偶尔会吐出半截或者格式坏掉的参数），不是让整个流程崩掉，而是退化成一个空字典 `{}`，让这次调用带着空参数往下走，再由上一篇讲的参数校验去给模型一句「参数不对」的反馈。流式解析里，「坏数据要能优雅降级」是个反复出现的主题，因为流随时可能给你残缺的东西。

文本部分还支持一个 `on_token` 回调，每收到一截文本就喊一声，CLI 拿这个回调实现「打字机」效果，让你看着字一个个冒出来，而不是干等几十秒蹦出一整段。

## token 是怎么数出来的

要算钱，先得知道每次调用花了多少 token。CoreCoder 不自己估，它问 provider 要准数。办法是在请求里加一个 `stream_options`：

```python
params["stream_options"] = {"include_usage": True}
```

加了这个，provider 会在流的最后一个 chunk 里带回 `usage`，里头有 `prompt_tokens` 和 `completion_tokens`。代码在循环里接住它：

```python
if chunk.usage:
    # some providers send usage with null fields; coerce to 0 so the
    # running totals below don't blow up on int + None
    prompt_tok = chunk.usage.prompt_tokens or 0
    completion_tok = chunk.usage.completion_tokens or 0
```

那个 `or 0` 不是多余的。有些 provider 会把 `usage` 发回来但字段是 `null`，要是直接拿去和累计值相加，`int + None` 会抛异常，把整个会话搞挂。一句 `or 0` 把这种脏数据压平。这又是一个「真实 API 不会按文档那么干净」的例子，写包装层的人得替上层把这些坑都垫平。

数出来的 token 累加进 `total_prompt_tokens` 和 `total_completion_tokens`，CLI 的 `/tokens` 命令随时能查。

## provider 抽风了，自己扛

这是上一篇结尾埋的伏笔。主循环 `agent.py` 里没有任何重试逻辑，因为重试被下放到了这一层。`_call_with_retry` 负责扛住瞬时故障：

```python
def _call_with_retry(self, params: dict, max_retries: int = 3):
    """Retry on transient errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return self.client.chat.completions.create(**params)
        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            time.sleep(wait)
        except APIError as e:
            # retry 5xx server errors but not 4xx; base APIError has no
            # status_code so read it defensively
            status_code = getattr(e, "status_code", None)
            if status_code and status_code >= 500 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
```

逻辑是经典的指数退避：限流、超时、连接错误这类一看就是瞬时的故障，第一次失败等 1 秒、第二次失败等 2 秒，包含初次请求在内最多尝试三次，仍不行就抛出去。服务端 5xx 也重试，但客户端 4xx（参数错、鉴权失败这种）绝不重试，因为重试一百次结果都一样，只是白等。

这段重试逻辑值得专门点出来，因为「这么小的项目应该没空管重试吧」是个很自然的误会，而事实正相反。重试不但在，还考虑得相当周到，连基类 `APIError` 可能没有 `status_code` 属性都用 `getattr` 防住了（不同版本的 SDK 异常层级不一样，硬取属性会炸）。

## 重试耗尽以后，才轮到 fallback

当前源码已经补上显式 fallback 链。它没有猜「哪个模型能替哪个」，而是把策略交给配置：`CORECODER_FALLBACK_MODELS=gpt-5.4-mini,gpt-4o-mini`，或者重复传 `--fallback-model`。`LLM.chat()` 每轮先用当前模型；只有限流、超时、连接错误或 5xx 在 `_call_with_retry` 内重试耗尽后，才尝试链里的下一个模型。参数错误、鉴权失败等 4xx 不 fallback，因为换个模型掩盖不了坏配置。

```python
candidates = [self.model, *self.fallback_models]
for candidate in candidates:
    try:
        stream = self._call_with_retry({**params, "model": candidate})
        self.model = candidate
        break
    except Exception as e:
        if not self._is_fallback_error(e):
            raise
```

切换成功是 sticky 的：后续轮次直接从备用模型继续，不会每次都重新等主模型失败。`fallback_history` 留下切换记录，`/tokens` 会显示当前 active model。stream 建立后如果在第一个 chunk 之前断掉，也可以安全 fallback；一旦已有 token 交给 `on_token`，就不再自动重放，否则用户会看到重复文本，工具参数也可能被执行两次。同一个 `LLM` 实例里的候选模型共享 backend、`base_url` 和显式凭据，因此 OpenAI-compatible 路径适合一个端点能路由多个模型的 provider；LiteLLM 可以表达 `provider/model` 名称，但 provider 独立凭据仍需要由环境或后续路由层提供。

还有个容易被忽略的协调细节。`stream_options` 是 OpenAI 的扩展，有些 provider 不认，会回一个 400。CoreCoder 的处理是：

```python
try:
    stream = self._call_with_retry(params)
except BadRequestError:
    params.pop("stream_options", None)
    stream = self._call_with_retry(params)
```

未配置美元预算时，捕获 `BadRequestError`（400）后仍可以把 `stream_options` 去掉再试一次。配置预算后则不能这么降级：没有 `include_usage`，客户端就无法知道真实消耗，此时会 fail closed，抛出 `BudgetExceededError`。两种重试仍然分开：参数方言适配解决兼容性，指数退避解决瞬时故障，模型 fallback 只接住后者最终仍失败的结果。

## 不兼容 OpenAI 的，交给 LiteLLM

`LiteLLM` 子类是给那些不走 OpenAI 兼容接口的 provider 准备的逃生通道。它继承 `LLM`，但绕开了父类构造函数里创建 openai client 的那步，转而把请求交给 `litellm` 这个库去路由，litellm 支持上百家 provider：

```python
class LiteLLM(LLM):
    def _call_with_retry(self, params, max_retries=3):
        import litellm
        params["drop_params"] = True   # 不支持的参数自动丢掉，别报错
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url
        # ...同样的指数退避...
```

设了 `CORECODER_PROVIDER=litellm` 之后，你就能用 litellm 的模型串，比如 `anthropic/claude-3-haiku`、`bedrock/anthropic.claude-v2`、`vertex_ai/gemini-pro`。`drop_params=True` 是个贴心的开关，provider 不支持某个参数时自动丢掉而不是报错，省得你为每家去裁参数。绝大多数人用主路径就够了，LiteLLM 是那条「万一你的 provider 太特立独行」的后路。

## 把钱算出来

知道了 token，算钱就是查表乘一乘。`llm.py` 里有一张按百万 token 计价的表（输入价、输出价）：

```python
_PRICING = {
    "gpt-5.5": (5, 30),           # CoreCoder 默认就用它
    "deepseek-chat": (0.27, 1.10),
    "claude-sonnet-4-6": (3, 15),
    "kimi-k2.5": (0.6, 3),
    # ...
}
```

`estimated_cost` 会按实际响应模型分别累计 `usage_by_model`，再乘各自单价。这样从主模型切到便宜的 fallback 后，不会错拿当前模型价格计算此前所有 token：

```python
@property
def estimated_cost(self) -> float | None:
    total = 0.0
    for model, tokens in self.usage_by_model.items():
        pricing = _pricing_for_model(model)
        if not pricing:
            return None
        total += cost(tokens, pricing)
    return total
```

注意返回类型是 `float | None`。表里没有的模型，它不瞎猜，老老实实返回 `None`，CLI 那边看到 `None` 就不显示价格，而不是编一个数字骗你。这是个小但重要的诚实：宁可说「我不知道」，也不给你一个看着精确、实则瞎编的成本。CLI 的 `/tokens` 命令把它显示出来：

```
Tokens: 12043 prompt + 3201 completion = 15244 total  (~$0.0621)
```

配置 `CORECODER_MAX_COST_USD=1.00` 或 `--max-cost 1.00` 后，这个估算从展示信息变成执行闸。每次请求前，`LLM` 先扣掉已经报告的成本，再用序列化请求的 UTF-8 字节数加协议余量，保守预留输入 token 成本；剩下的钱换算成最多能生成多少输出 token，并压低 `max_tokens`。钱连下一次输入都覆盖不了，就在发请求前停止。

预算模式是 fail-closed 的：模型不在 `_PRICING`、provider 不支持 usage、或者响应没有返回 token 数，都拒绝继续花钱。这个「硬」指客户端基于当前价目表和 provider usage 给出的执行上限，不等于云账单担保：价格表可能过期，缓存折扣没算，失败或半截 stream 也可能被 provider 计费却没有 usage 返回。生产部署仍应叠加 provider 账户侧额度告警。

## 配置从哪来

最后串一下 `config.py`。它从环境变量读配置，带一个合理的优先级：

```python
api_key = (
    os.getenv("CORECODER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or ""
)
```

专属变量优先，然后退到通用的 `OPENAI_API_KEY`，再退到 `DEEPSEEK_API_KEY`。它还会从当前目录往上一直找到家目录，加载 `.env` 文件（`override=False`，不覆盖你已经设好的真环境变量）。这样你在项目目录放一个 `.env`，进来就能用，不用每次 export。小事，但顺手。

fallback 和预算同样能放进 `.env`：

```bash
CORECODER_FALLBACK_MODELS=gpt-5.4-mini,gpt-4o-mini
CORECODER_MAX_COST_USD=1.00
```

## 这一篇带走什么

- 「大多数 provider 都兼容 OpenAI 接口」这个赌注，让整个 provider 层薄到只是 openai SDK 的一层包装，换 provider 就是换两个环境变量。
- 流式输出里，工具调用的参数是分片到达的，得按 index 累加再 `json.loads`，还要对坏数据优雅降级。
- 重试该放在 provider 层，不该塞进主循环。指数退避只重试瞬时故障和 5xx，绝不重试 4xx。
- fallback 只接住重试耗尽后的瞬时故障和 5xx，不接住 4xx；切换成功后保持在备用模型，避免每轮重复等待。
- 成本按模型分别累计；美元预算在请求前预留输入成本并限制最大输出，对未知价格或缺失 usage 采取 fail-closed。

下一篇，我们面对 agent 最硬的那道物理约束：上下文窗口就这么大，一个长任务怎么塞得下。
