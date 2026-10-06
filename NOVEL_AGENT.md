# 自动写小说 Agent（MVP）

这是一个独立于 `werewolf_agent` 的 Python 异步小说创作 Agent。它把长篇创作拆成几个可恢复的阶段：

1. 根据题材和创意生成世界观、人物、主线大纲、伏笔账本和平台作品简介。
2. 为下一章生成场景节拍计划。
3. 生成正文。
4. 检查人物、时间线、世界规则和伏笔状态。
5. 必要时自动改写。
6. 提取人物状态、事件摘要和伏笔变化，然后原子保存项目。

## 配置

先安装仓库依赖：

```bash
python -m pip install -r requirements.txt
```

如果要启用本地 RAG，还需要安装 ChromaDB 和 Sentence Transformers：

```bash
python -m pip install -r requirements-rag.txt
```

RAG 默认关闭，因此不安装这些可选依赖也能正常使用原有的规划、写作和
`status` 命令。GPU 用户可以先根据本机 CUDA 版本安装匹配的 PyTorch；默认
配置也可以直接在 CPU 上运行。

配置 OpenAI 兼容接口（可复制 `.env.template` 为 `.env`）：

```dotenv
LLM_API_KEY="YOUR_API_KEY"
LLM_BASE_URL="https://your-endpoint/v1"
LLM_MODEL_ID="YOUR_MODEL"
LLM_TIMEOUT="120"

# 可选：本地 ChromaDB + BGE 检索
NOVEL_RAG_ENABLED="false"
NOVEL_RAG_EMBEDDING_MODEL_ID="BAAI/bge-small-zh-v1.5"
NOVEL_RAG_DEVICE="auto"
NOVEL_RAG_TOP_K="6"
NOVEL_RAG_MAX_CHARS="6000"
NOVEL_RAG_REBUILD="false"
NOVEL_RAG_OFFLINE="false"
NOVEL_DEAI_REVIEW="true"
```

## 使用

创建项目并自动生成故事圣经：

```bash
python -m novel_agent new \
  --title "雾港来信" \
  --premise "失忆的邮差追查一封来自未来的信" \
  --genre "悬疑奇幻" \
  --chapters 20 \
  --words-per-chapter 2500 \
  --theme "记忆,选择" \
  --output-dir projects
```

创建项目时也会同时生成长版作品简介和一句话简介，并保存为项目根目录的
`synopsis.md`。简介不会被注入章节正文提示词，因此营销文案不会污染故事连续性。

创建时即可为项目启用本地 RAG 配置：

```bash
python -m novel_agent new \
  --title "雾港来信" \
  --premise "失忆的邮差追查一封来自未来的信" \
  --chapters 20 \
  --rag \
  --rag-model "BAAI/bge-small-zh-v1.5" \
  --rag-device cpu
```

继续写一章或多章：

```bash
python -m novel_agent write projects/雾港来信 --count 1
```

也可以在已有项目上首次开启 RAG。引擎会为已完成章节建立本地 Chroma
索引，并在后续章节写作时检索相关旧片段：

```bash
python -m novel_agent write projects/雾港来信 \
  --count 1 \
  --rag \
  --rag-top-k 6 \
  --rag-max-chars 6000
```

章节写作默认会进行本地“去模板化”文风审稿：检查重复比喻、强刺激转折、
情绪套语、解释腔和过度均匀的节奏；这些只是提供给模型审稿人的证据，只有审稿人
判断可以安全修复时才会交给现有改写阶段。它不是平台 AI 检测器，也不会声称能绕过平台审核；目标是让
语言更自然，同时保持剧情、人物、时间线和伏笔不变。需要调试原始生成时可用：

```bash
python -m novel_agent write projects/雾港来信 --count 1 --no-deai-review
```

本地预检不调用额外模型；它只把小型、可解释的报告附加到原有审稿提示中。
只有模型审稿决定需要修改时，才会增加原本就存在的改写调用。

也可以完全脱离章节生成，单独检查已经保存的正文。只检查、不修改：

```bash
python -m novel_agent polish projects/雾港来信 \
  --chapter 1 \
  --check-only
```

检查并应用安全的自然化润色：

```bash
python -m novel_agent polish projects/雾港来信 --chapter 1
```

从第 1 章开始连续处理 5 章：

```bash
python -m novel_agent polish projects/雾港来信 --chapter 1 --count 5
```

`polish` 不会创建场景计划、生成新章节或重新计算人物/伏笔记忆；它只修改正文表层表达，
并保护原有数字、主要角色、事件边界和元数据。应用成功后会同步保存章节文件、
`project.json`，把原稿备份到 `audit/polish/`，并在项目启用 RAG 时刷新本地向量索引。`--check-only` 每章使用一次
审稿调用；实际润色时如需要修改，会再使用一次改写调用。

RAG 参数如下：

- `--rag` / `--no-rag`：本次运行启用或禁用 RAG。
- `--rag-model`：本地 embedding 模型，默认
  `BAAI/bge-small-zh-v1.5`。
- `--rag-device`：运行设备，例如 `auto`、`cpu`、`cuda`、`cuda:0` 或
  `mps`。
- `--rag-top-k`：每章最多取回的历史片段数，默认 6。
- `--rag-max-chars`：注入提示词的检索文本字符上限，默认 6000。
- `--rag-rebuild`：本次运行强制重建项目索引，适合更换模型或修复索引。
- `--rag-offline`：只使用本地缓存的 BGE 文件，不允许联网下载模型。
- `--deai-review` / `--no-deai-review`：启用或禁用本地去模板化文风审稿，默认启用。

命令行参数优先于 `NOVEL_RAG_*` 环境变量；环境变量优先于项目中已经保存的
RAG 配置。没有显式配置时保持关闭，旧项目无需修改即可继续按原方式写作。

首次使用某个 BGE 模型时，Sentence Transformers 通常需要从 Hugging Face
下载模型文件。下载完成后可设置标准的 `HF_HOME` 缓存目录，并结合
`--rag-offline` 完全离线运行。ChromaDB 和 BGE 都在本机工作，不产生
embedding API 调用费用，但会占用本地磁盘、内存和 CPU/GPU 计算资源。

查看状态（不调用模型）：

```bash
python -m novel_agent status projects/雾港来信
python -m novel_agent status projects/雾港来信 --json
```

给已经创建、但没有简介的旧项目生成或重写简介：

```bash
python -m novel_agent synopsis projects/雾港来信
python -m novel_agent synopsis projects/雾港来信 --json
```

`synopsis` 命令只读取故事圣经和已完成章节摘要，不会初始化向量数据库，也不会生成正文。

项目目录中会保存 `project.json`、`synopsis.md`、`chapters/001.md` 等正文文件，以及
`audit/llm_calls/` 下的请求审计记录。启用 RAG 后，本地 Chroma 数据和索引
清单保存在项目的 `rag/` 目录；它们是可从已完成章节重新生成的缓存，不会
改变 `project.json` 的核心小说结构。项目状态在每章完成后原子写入，进程
中断后可以再次执行 `write` 继续。

终端会在规划、正文、审稿、改写和记忆更新阶段显示进度。相同标题在未显式指定 `--project-name` 时会自动使用新的目录名；显式指定的已有目录会直接报错，避免覆盖旧项目。

## 离线测试

测试使用脚本化 fake backend，不需要 API key：

```bash
python -m unittest discover -s tests -v
```

当前版本是 MVP：模型负责创作和审稿，领域模型负责 ID、章节顺序、伏笔状态和存档约束。`metadata.usage` 记录的是请求次数和字符量估算，不是服务商返回的精确 token 用量。默认按单个项目顺序写作，暂不支持多个进程同时写同一项目。

真正上线前仍建议增加人工抽检、题材专用评测集、并发任务队列、精确 token
usage 和 Web 界面。
