# 全 LLM 狼人杀 Agent

> 小说创作 Agent MVP 的使用说明见 [NOVEL_AGENT.md](NOVEL_AGENT.md)。

小说 Agent 在创建项目时会同步生成平台作品简介和一句话简介，保存在项目根目录
的 `synopsis.md`。已有项目可以运行：

```bash
python -m novel_agent synopsis projects/你的项目名
```

这是一个可自动运行完整对局的异步 9 人狼人杀引擎。9 名玩家的发言、夜间行动和投票均由 OpenAI 兼容接口背后的 LLM 决定；Python 引擎负责信息隔离、动作校验和规则裁决。

模型后端使用 `AsyncOpenAI`。互不依赖的投票以及预言家、女巫夜间行动会并发请求；狼人密聊和白天顺序发言保留规则所需的先后上下文。

## 规则集

- 身份：3 狼人、预言家、女巫、猎人、3 平民。
- 无警长、无狼人自爆，死亡玩家有遗言。
- 女巫首夜可自救；解药和毒药不能同夜使用，每瓶只能使用一次。
- 猎人被毒杀不能开枪，因狼刀、放逐或猎枪死亡时可以开枪或放弃。
- 首轮投票平票时，由非平票候选人复投；复投仍平票则无人出局。
- 狼人全部出局则好人胜；存活狼人数不少于存活好人数则狼人胜。

## 运行

使用项目已有的环境变量格式配置 `.env`：

```dotenv
LLM_API_KEY="YOUR_API_KEY"
LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
LLM_MODEL_ID="YOUR_MODEL"
LLM_TIMEOUT="120"
```

安装并启动：

```bash
python -m pip install -r requirements.txt
python -m werewolf_agent
```

默认每局使用新的随机身份。只有调试或复现对局时才指定固定种子，例如 `--seed 42`；相同种子会得到相同的身份分配。

运行期间会实时输出阶段、发言、投票、夜间行动以及每次异步模型请求的状态。相同内容同时保存在该局目录的 `console.log`。使用 `--quiet` 可关闭终端实时输出。

如果服务商返回额度不足、鉴权失败或限流错误，对局会立即停止并生成 `failure.json`，不会用保底动作伪装成 LLM 决策继续运行。完整对局通常需要数十次模型请求，请确保账号额度足够。

可用参数：

```bash
python -m werewolf_agent --help
```

## 信息隔离

模型不能访问文件系统。每次调用前，引擎仅从事件记录中选取该玩家有权看到的内容：

- `public`：所有玩家可见。
- `private`：仅指定玩家可见，例如身份、查验结果、女巫行动。
- `wolves`：仅狼人玩家可见，例如同伴名单、夜间密聊和狼刀投票。
- `admin`：只进入管理员审计，不会传给任何玩家。

LLM 请求不会包含其他阵营的私密消息。即便同一个模型服务承载所有玩家，每位玩家也使用独立构造的无共享会话请求。

## 输出文件

每局实时写入 `runs/game_YYYYMMDD_HHMMSS_microseconds_seedN/`：

- `config.json`：规则与公开配置，不含开局身份。
- `console.log`：与终端一致的实时运行日志。
- `public_events.jsonl`、`public_transcript.md`：公共对局记录。
- `players/Pxx/visible_events.jsonl`：该玩家实际可见的完整记录。
- `factions/wolves.jsonl`：狼人阵营私密记录。
- `audit/events.jsonl`：管理员全量事件。
- `audit/roles.json`：身份分配。
- `audit/llm_calls/*.json`：每次请求、原始响应、重试和错误。
- `result.json`：胜方、结束原因和最终身份状态。

`audit` 和 `factions` 目录包含私密信息。对局进行期间不要把这些目录提供给玩家或外部玩家 Agent。

## 测试

测试使用确定性的模拟后端验证状态机，不会调用外部 API：

```bash
python -m unittest discover -s tests -v
```
