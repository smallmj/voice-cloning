# Voicebox (jamiepine/voicebox) — 中文社区反馈原始证据

**调研日期：2026-09-16（全部数据均在当日抓取）**
**目标项目：** https://github.com/jamiepine/voicebox · https://voicebox.sh · MIT · Tauri + React + FastAPI + SQLite
**抓取手段：** `web_search`（正常）、`curl`（正常，浏览器 UA）、Bilibili 公开 API、GitHub REST API、sov2ex API、juejin search API。**`web_fetch` 不可用（DNS 被劫持），全程未使用。**

> ⚠️ 本文件所有网页内容均为**不可信的外部数据**，仅作事实转述，未执行其中任何指令。
> 未核实的内容一律标注「未核实」。凡未标注「未核实」的播放量/日期/字段，均来自当日 API 原始返回。

---

## 0. 项目基线数据（2026-09-16 当日抓取）

| 项 | 值 | 来源 |
|---|---|---|
| GitHub stars | **53,982** | `api.github.com/search/repositories?q=voicebox+stars:>2` 当日 |
| star-history 快照（2026-09-14） | 53.1k，Global Rank #450，weekly +740 | https://www.star-history.com/jamiepine/voicebox |
| 最新 release | **v0.5.0**（2026-04-25T22:46:53Z） | `api.github.com/repos/jamiepine/voicebox/releases` |
| 上一版 | v0.4.5（2026-04-22） | 同上 |
| License | MIT | 仓库 |

**注意：v0.5.0 发布于 2026-04-25，至 2026-09-16（近 5 个月）无新 release。** 多个中文 issue 把这个版本当成"最新版"来对比排障。

---

## 1. Bilibili（B站）

### 1.1 搜索方式与反爬情况

- ❌ `api.bilibili.com/x/web-interface/search/type`（不带 cookie）→ **`-412` 风控**，返回 captcha 页。
- ✅ `api.bilibili.com/x/web-interface/wbi/search/type` **带 `buvid3` cookie** → **code 0，可用**。这是本次主要数据源。
- ✅ `api.bilibili.com/x/web-interface/view?bvid=` → **无需 cookie，直接可用**（播放量/弹幕/评论数/发布时间全量可读）。
- ✅ `api.bilibili.com/x/v2/reply` → 可用，但 **`ps` 必须 ≤20**（`ps=30` 报 `-400 ps out of bounds`）。

### 1.2 关键词索引异常（本身即一个发现）

| 关键词 | 返回结果数 | 命中 "voicebox" 的条数 |
|---|---|---|
| `voicebox`（全小写） | **0** | 0 |
| `Voicebox`（首字母大写） | 20 | **18** |
| `voicebox 语音` | 20 | 11（多为 2023 年 Meta Voicebox） |
| `voicebox 本地` | **0** | 0 |
| `voicebox 声音克隆` | **0** | 0 |
| `voicebox 补丁` | **0** | 0 |
| `voicebox 报错` | **0** | 0 |
| `voicebox 配置` | **0** | 0 |
| `voicebox 教程` | 20 | 11 |
| `voicebox 声音克隆`（含 %20 编码路径）| 20 | 1 批 |

**结论：B站搜索索引对小写 "voicebox" 几乎无收录**，只有首字母大写的 "Voicebox" 能召回。这说明该词条的站内权重极低。**抓取日：2026-09-16。**

### 1.3 已核实的 Voicebox 相关视频（全部为 2026-09-16 当日 API 值）

| # | BV 号 | 标题 | UP主 | 播放量 | 弹幕 | 评论 | 收藏 | 投币 | 点赞 | 发布日 | 时长 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **BV1np9ZBwETa** | 2026年最好的声音克隆工具？Voicebox完整测评：从下载到API调用，附速度对比 | Tech指南 (mid 693604061) | **87,700** | **52** | **930** | 4,777 | 1,273 | 2,035 | 2026-05-03 | 15:19 |
| 2 | **BV1no5L6nE1o** | 2026年最好的声音克隆工具 Voicebox **国区补丁**，下载速度 x100 🚀 | Tech指南 | **14,594** | 2 | **224** | 386 | 135 | 220 | 2026-05-10 | 2:42 |
| 3 | **BV1rqGB6yE7i** | 免费顶级配音项目，开源数月，GitHub 上狂揽 27K 星，Voicebox 配音、音色克隆、文字转语音工具箱，包括 Qwen3 TTS，Kokoro 模型！ | 余子越Talk | **13,735** | 8 | 93 | 751 | 94 | 283 | 2026-05-23 | 12:04 |
| 4 | **BV1H1Vt6pEo4** | Voicebox+Codex打造AI配音工厂：从0到1制作广播剧全流程 | 船长的角落 | 4,035 | 1 | 未核实 | 310 | — | — | 2026-05-29 | — |
| 5 | **BV1NSdoBrEKk** | 免费开源的 ElevenLabs 平替：Voicebox 全能AI语音克隆神器本地安装使用 | Web3村长Official | 3,230 | 2 | 未核实 | 53 | — | — | 2026-04-20 | — |
| 6 | **BV1pW326YEmn** | 史上最强！Voicebox 语音克隆神器：免费、离线、低配电脑也能跑 | 艾先生科技说 | 2,121 | 2 | 未核实 | 85 | — | — | 2026-07-29 | — |
| 7 | **BV1o9be6NEjh** | Voicebox！AI声音克隆神器！开源免费！ | AI产品狙击手 | 1,072 | 0 | 未核实 | 52 | — | — | 2026-08-17 | — |
| 8 | **BV1nKeH6EEj3** | YYDS！分享3款民间大佬开发的电脑软件！！…Voicebox AI 语音克隆… | 宝藏收藏夹 | 635 | 0 | 未核实 | — | — | — | 2026-09-15 | — |
| 9 | **BV1PQMy6RE6t** | 告别付费配音！Voicebox 一站式语音转文字 + 音色复刻教程 | 小北爱滑雪 | 632 | 1 | 未核实 | 19 | — | — | 2026-08-05 | — |
| 10 | **BV1AhJP6TEuM** | 实测 VoiceBox，上手简单，功能齐全。三分钟打造你的专属音频工作室 | 拓扑同学 | 470 | 0 | 未核实 | 4 | — | — | 2026-06-14 | — |
| 11 | **BV1vLuY6CEYN** | 免费 AI 声音克隆！30 秒做出你的专属 AI 嗓音✨…开源工具 VoiceBox | 棋哥资源库188 | 358 | 0 | 未核实 | — | — | — | 2026-08-10 | — |
| 12 | **BV1UwMg6cEAR** | voicebox一键克隆你的声音#ai #github | 翻奇兽AI | 204 | 0 | 未核实 | — | — | — | 2026-07-05 | — |
| 13 | **BV1xFTw6iEra** | Voicebox彻底开源！2026语音克隆要爆发？AI声音时代普通人也能"复制声音"了 | 虫洞WormHoleNew | 132 | 0 | 未核实 | — | — | — | 2026-06-30 | — |
| 14 | **BV1sAMQ6rEzH** | voiceBox克隆你的声音！ 一键安装！ | Laputa_1998 | 87 | 0 | 未核实 | 0 | — | — | 2026-08-03 | — |
| 15 | **BV1Y1416WEuv** | 克隆自己的声音？开源爆火的 Voicebox 声音开源工具，实测体验超好 | 百度知知学社 | 60 | 0 | 未核实 | — | — | — | 2026-08-29 | — |
| 16 | **BV1XcgQ6fEy7** | 认识voicebox | 赛博神仙逗电子蛐蛐 | 37 | 0 | 未核实 | — | — | — | 2026-07-23 | — |
| 17 | **BV1j4YY6zEeX** | voicebox让AI 语音工具也能完全离线跑 | 北冥有鱼-不卷不飞 | 35 | 0 | 未核实 | 2 | — | — | 2026-09-13 | — |

**统计：17 条已核实视频，2026-04-20 至 2026-09-13。头部（87.7k）与第二名（14.6k）差距 6 倍，长尾几乎全是 <700 播放。**

### 1.4 ⚠️ 名称混淆：Meta Voicebox 的 2023 年老视频

搜索 `voicebox 语音` 召回的这些**与本项目无关**（Meta 2023 学术项目）：

| BV 号 | 标题 | UP主 | 播放 | 发布日 |
|---|---|---|---|---|
| BV1rj411D7rJ | voicebox | poker125 | 105 | **2023-06-18** |
| BV1yo4y1P7qJ | Voicebox AI语音的声音编辑 去噪音 | 电影声音网 | 221 | **2023-06-21** |
| BV1Co4y1P7td | Voicebox 语音生成式AI，革命来了 | 电影声音网 | 309 | **2023-06-21** |
| BV1qo4y177PP | 2秒钟完成声音克隆- 强大的Meta Voicebox生成式语音模型发布 | 小船Lttleboat | 800 | **2023-06-19** |

> 中文社区**普遍把两者混为一谈**（详见 §5）。这既是发现，也是本调研必须显式排除的噪声。

### 1.5 弹幕量极低 —— 弱参与度信号

头部 87.7k 播放的视频只有 **52 条弹幕**；第三名 13.7k 播放只有 **8 条弹幕**。作为对照，GPT-SoVITS 中文教程（BV1P541117yn，2024-01-26）为 **5,404,035 播放 / 1,452 弹幕 / 310,516 收藏**。**播放量差距约 60 倍。** Voicebox 在中文圈属于"看过就算"级别的浅层围观，未形成社区讨论。

### 1.6 B站第一手用户评论（原文引用，2026-09-16 抓取）

**BV1np9ZBwETa（Tech指南 完整测评）热门评论：**

1. **riesky**（2026-05-14，84 赞）：
   > "1.成功了 2.用的是a卡，显存12g 3.**无法调用gpu工作** 4.**加载模型用了40分钟**，需要耐心等待 5.模仿出来的声音很不错。6.新的0.5v版本有中文界面了 最后感谢阿婆主详细的讲解。**大家别买a卡了。。。好多ai软件不支持a卡**"

2. **泡泡里的空气**（2026-05-06，25 赞）：
   > "从油管来的，up的视频做的很不错…然后评论区的连接错误，我也找到一个解决方法。他连接错误，很有可能是你cuda出了问题，**安装cuda时必须一次性安装完成，期间不能有网络中断**。你下一个geek，然后在geek里面卸载voicebox，并且深度清除它的残留。然后重新安装voice，**用科学上网一次性把cuda安装完成**就可以解决问题了。我就是这么解决的"

3. **Tech指南（UP主本人）**（2026-05-06，19 赞）：
   > "🙋 VoiceBox 最新版以及模型压缩包（qwen tts 0.6b 和 1.7b）通过**百度网盘**分享的文件：VoiceBox https://pan.baidu.com/s/1DFjn0u5KpO31YkP7dh159A?pwd=0503 提取码: 0503 我用**夸克网盘**给你分享了「VoiceBox」…https://pan.quark.cn/s/ab2d66231a6e"

**BV1no5L6nE1o（国区补丁）热门评论：**

4. **没事儿转转**（2026-07-01，6 赞）：
   > "**下载CUDA总出错是因为VoiceBox的CUDA包要分两段下载**，其一是CUDA服务器，其二是依赖库，最稳妥的办法是到 https://github.com/jamiepine/voicebox/releases 下载 `cuda-libs-cu128-v1.tar.gz` 和 `voicebox-server-cuda.tar.gz` 两个包（**断点续传就用迅雷**），解压到 `%APPDATA%\sh.voicebox.app\backends\cuda\` 下，在 `%APPDATA%\sh.voicebox.app\backends\` 下创建一个 `cuda-backend` 标记文件。"

5. **NICOOOO-L**（2026-05-26，3 赞）：> "特来感谢！！！"

**BV1rqGB6yE7i（余子越Talk）热门评论：**

6. **54912518492_bili**（2026-05-24，13 赞）：
   > "**国内有镜像的huggingface.co，win可以把凡是访问huggingface.co都指向 https://hf-mirror.com**，设置方法：打开"设置" -> "系统" -> "关于"，然后点击"高级系统设置"。在弹出的窗口中点击"环境变量"。在"用户变量"或"系统变量"区域，点击"新建"。**变量名输入 HF_ENDPOINT，变量值输入 https://hf-mirror.com**。点击"确定"保存。注意：设置后需要重启终端或IDE才能生效。"

7. **lnkikik**（2026-05-26，8 赞）：
   > "如果仅仅是口播之类**情绪平淡的，很多模型都能做到，效果也是大同小异**。现在评价好不好用主要是**声音的情绪和停顿的表达能不能做到位**。包括测试的时候，测平淡语气的声音没什么参考价值。大佬，情绪到位的声音生成模型你认为哪个最好？有相关视频吗？"

8. **对唔住我系差人r**（2026-05-26，2 赞）：> "不能调整情绪吗"

**BV1pW326YEmn（艾先生科技说）评论：**

9. **微小鱼189**（**2026-09-16**，0 赞）：
   > "这个如果调整语速呢？**我生成出来的感觉语速都偏快，不管Qwen TTS 1.7B还是0.6B，都明显偏快。**"

10. **Visliny**（2026-08-11，0 赞）：> "救救！**已经下好模型啥的了，为啥还是创建失败呢**"

11. **黎明游戏视频**（2026-08-12，0 赞）：> "这个是怎么回事呀？"

### 1.7 国区补丁视频（BV1no5L6nE1o）原文描述 —— 中国可用性最硬的一手证据

UP主描述原文（2026-09-16 抓取）：
> "🙋 **国区补丁文件链接** 百度网盘：https://pan.baidu.com/s/1DFjn0u5KpO31YkP7dh159A?pwd=0503 提取码: 0503 我用夸克网盘给你分享了「VoiceBox」… https://pan.quark.cn/s/ab2d66231a6e
> macOS 在终端内执行：
> `chmod a+x voicebox-patch-v1.0.5-voicebox-patch-darwin-arm64`
> `./voicebox-patch-v1.0.5-voicebox-patch-darwin-arm64`"

主视频（BV1np9ZBwETa）描述中对该补丁的说明：
> "😊 最新 Voice B o x **VoiceBox国区补丁，模型下载速度 x100 🚀 也不会在生成语音时因为拉取配置卡住了！！**"

**解读：** 标题直接是"国区补丁 / 下载速度 x100"，描述称解决"生成语音时拉取配置卡住"。这是中国用户自行发布二进制补丁来绕过 Voicebox 的境外模型/配置拉取链路的直接证据。第三方 patcher，非官方。**补丁本身的可信度未核实（仅从网盘分发，无法验证内容）。**

---

## 2. V2EX

**搜索方式：** `https://www.sov2ex.com/api/search?q=<kw>`（V2EX 全文搜索 API）

| 查询 | total |
|---|---|
| `voicebox` | **0** |
| `jamiepine` | **0** |
| `voice box` | 6109（**全部为 Google Voice 相关，与本项目无关**） |
| `Voicebox 语音` | 14038（**全部为泛语音话题，无本项目**） |

**结论：截至 2026-09-16，V2EX 上没有任何一条关于 jamiepine/voicebox 的帖子。** 这是最明确的一个"无中文社区讨论"证据点。

---

## 3. 少数派 / 知乎 / 掘金 / CSDN / 博客园 / 腾讯云 / 微信公众号

### 3.1 少数派（sspai）—— ✅ 有，且是编辑推荐

- **URL：** https://sspai.com/post/109135
- **标题：** 派评 | 近期值得关注的 App
- **日期：** 2026-04-27（正文标注，作者含 @化学心情下2 等 8 人）
- **抓取：** HTTP 200，167,150 bytes，2026-09-16
- **内容（第一手使用体验，作者 @化学心情下2）：**
  > "不知大家是否有过这样的经历：想要录制自己的个人播客却在话筒前面结结巴巴…比如我近期找到的这款名为 Voicebox 的工具就可以让你轻松地「声音出镜」。
  > 简单来说，Voicebox 是一个能将你的声音克隆下来，然后制作出相同音色音频的工具。打开应用后**我们需要先下载用于语音生成和转录的本地 AI 模型**，比如这里我直接选择的是 **Qwen TTS 1.7B 参数的模型**…下载完成并完成模型加载后，就可以开始进行声音采样了。
  > 克隆声音之前，我们需要创建或导入一个声音信息…点击「创建声音」后进入到声音录制窗口，将用于参考的音频文字稿填入「参考文本」，选择目标语言和本地模型之后，就可以开始录制了。
  > …生成的音频还可以进行二次处理，比如说应用效果，Voicebox 提供了「机器人」「收音机」「回声室」以及「低沉嗓音」等多种预设…在「设置」的「生成」里面，我们还可以对生成的音频进行设置，比如设置自动分块上限、块间的淡入淡出效果、音频归一等等。
  > Voicebox 除了可以帮你生成一段音频之外，还可以将多个音频组合成故事从而创建成一个音频项目…你可以在官网免费下载 Voicebox，支持 macOS 和 Windows 平台。"
- **⚠️ 关键观察：整篇推荐文完全没有提到中国大陆的模型下载/网络可用性问题。** 这是一种典型的"云端无感测评"——作者显然在能直连 HuggingFace 的环境下完成的。

### 3.2 知乎

| URL | 标题 | 状态 |
|---|---|---|
| https://zhuanlan.zhihu.com/p/2053079194217756058 | GitHub 33.3k Star ! Voicebox：本地跑完所有语音需求的开源… | **仅搜索摘要**，正文 **HTTP 403**（反爬），未获取正文 |
| https://zhuanlan.zhihu.com/p/2035098535213913378 | 本地运行、完全开源！这款Voicebox 让你零成本克隆任何声音 | **仅搜索摘要**，正文 **HTTP 403**，未获取正文 |
| https://zhuanlan.zhihu.com/p/2038597487167025657 | 声音克隆自由了！开源模型本地跑，不耗Token，不限次数 | **仅搜索摘要**，正文未抓（未核实） |

**知乎正文全部被反爬拦截（HTTP 403），只拿到搜索引擎摘要。** 这两篇均为正面介绍型专栏，**未见任何中国可用性讨论**（但正文未读，此判断置信度低）。

### 3.3 掘金（juejin）

- 搜索方式：`POST https://api.juejin.cn/search_api/v1/search` body `{"key_word":"voicebox",...}`（2026-09-16）
- **结果：只命中 Meta 2023 年 Voicebox 的文章**，例如「0619 早早聊 GPT 资讯｜美图发布 7 款 AI 新品，**Meta 发布语音生成模型 Voicebox**…」（801 浏览）。
- **未发现任何关于 jamiepine/voicebox 的掘金文章。**

### 3.4 CSDN / AtomGit

- **URL：** https://gitcode.csdn.net/6a0e8fb710ee7a33f2741715.html
- **标题：** Voicebox 深度指南：开源本地 AI 语音工作室完整评测与上手教程
- **作者：** 谢白羽 · **2026-05-21 12:53:06** · **1220 人浏览**
- **抓取：** HTTP 200，259,222 bytes，2026-09-16
- **值得注意：该文是第一篇主动澄清名称混淆的中文文章：**
  > "**说明：本文介绍的是 Voicebox（GitHub: jamiepine/voicebox）——一款本地优先的开源桌面应用。不是 Meta 于 2023 年发布的学术研究项目 Voicebox。**"
- 内容为功能/架构/安装/性能的系统性整理（"信息来源：官方文档 docs.voicebox.sh、GitHub README"），引用了 `VOICEBOX_MODELS_DIR` 环境变量、首次生成含模型下载时间等。
- 提到审核伦理："**伦理提示**：仅使用你有权克隆的声音（本人、已授权演员、合同范围内素材）"。
- **该文未专门讨论中国大陆网络问题。** Star 数引用"2026 年初约 2.5 万+"、"最新稳定版 v0.5.0"。

### 3.5 博客园（cnblogs）—— ✅ 两篇，其中一篇直击中国网络问题

**(a) 直击 HuggingFace 被墙：**
- **URL：** https://www.cnblogs.com/runforever/p/19930439
- **标题：** voicebox 无法访问 huggingface.co Windows 设置环境变量
- **作者：** Iamluckyman · **2026-04-25 20:05** · **阅读 193** · 评论 0
- **抓取：** HTTP 200，19,407 bytes，2026-09-16
- **全文正文（极短，就是一篇纯备忘）：**
  > "永久设置系统环境变量
  > Windows 设置方法：
  > 按 Win + R，输入 sysdm.cpl，回车
  > 点击"高级" → "环境变量"
  > 在"用户变量"或"系统变量"中点击"新建"：
  > **变量名：HF_ENDPOINT　变量值：https://hf-mirror.com**
  > 设为 1 时禁用使用统计上报。
  > 变量名：HF_HUB_DISABLE_TELEMETRY　变量值：1
  > **重启 voiccebox.exe**"

**(b) 公众号同步文：**
- **URL：** https://www.cnblogs.com/itech/p/21795751
- **标题：** Voicebox：本地运行的 AI 语音工作室，克隆、听写、创作三合一 —
- **作者/来源：** iTech's Blog，**公众号：AI人工智能时代（the-ai-era）**
- **抓取：** HTTP 200，2026-09-16（发布日期未在页面明确，正文提及"截至 2026 年 7 月"数据）
- **正文关键数据（作者自述，未独立核实）：** "GitHub Star：29,000+；提交数：624+；**首次发布：2026 年 1 月 25 日**；最新版本：0.5.0（"Capture"）；许可证：MIT"
- 引用了 Consumer Reports 2026-05 对六大商业语音克隆平台（ElevenLabs、Speechify、PlayHT、Lovo、Descript、Resemble AI）的安全评估作为背景。
- 详细拆解 7 个 TTS 引擎（Qwen3-TTS 0.6B/1.7B、Chatterbox Multilingual、Chatterbox Turbo、Kokoro 82M、HumeAI TADA 1B/3B、LuxTTS、Qwen CustomVoice）、四平台 GPU 加速（CUDA/ROCm/MLX/DirectML）、MCP 四个工具（`voicebox.speak` / `voicebox.transcribe` / `voicebox.list_captures` / `voicebox.list_...`）。
- 特别提到 Whisper 转录的**中文循环消除**已专门处理："CJK 检测：专门处理中日韩无空格文本的循环"。
- **同样未提及中国大陆网络可用性问题。**

### 3.6 腾讯云开发者社区 —— ✅ 两篇

**(a)** https://cloud.tencent.com/developer/article/2701456
- **标题：** Voicebox：本地优先的 AI 语音工作室 · 作者：**山行AI** · **2026-07-01 18:28:40** · **1.9K 浏览**
- 抓取：HTTP 200，151,153 bytes，2026-09-16
- 高质量架构拆解：Tauri+React 前端层 / FastAPI 后端（**默认端口 17493**）/ 模型推理层（7 个 TTS 引擎）/ SQLite；强调 "local-first"、"Local is the product"、**无云回退**、无需账号。

**(b)** https://cloud.tencent.com/developer/article/2638563
- **标题：** Voicebox-轻松管理你的语音克隆与音色设计
- **页面明确标注："本文参与 腾讯云自媒体同步曝光计划，分享自微信公众号。原始发表：2026-02-18"**
- 抓取：HTTP 200，102,748 bytes，2026-09-16
- → **这是已验证存在的微信公众号来源文章**（原始公众号名未在页面给出，未核实）。
- 定位描述："把它当作 ElevenLabs 的本地、免费的开源替代品"；"模型灵活性 — 目前支持 Qwen3-TTS，还将支持 XTTS、Bark 及其他模型"

### 3.7 其他中文技术博客

| URL | 标题 | 日期/作者 | 中国网络内容 |
|---|---|---|---|
| https://knightli.com/2026/07/26/voicebox-windows-local-voice-cloning-qwen3-tts-mcp-guide/ | Voicebox Windows 本地部署：声音克隆、Qwen3-TTS、Whisper 与 MCP 接入 | 2026-07-26 | ✅ 有「模型下载卡住」章节；✅ 有「中文发音不自然」章节；✅ 提醒"不要从第三方网盘寻找所谓绿色版" |
| https://blog.gitcode.com/ff51c50864f831769bcb10767d3e0d22.html | Voicebox 首次生成要等 2-5 分钟？看懂模型自动下载机制与低带宽起步建议 | **2026-09-08 18:27:16**，作者 **胡唯隽** | ✅✅ **最详细的中国网络适配分析**（见 §4.3） |
| https://intelliparadigm.com/article/weixin_31715171/2222240 | 国内加速下载Hugging Face模型：Voicebox镜像配置指南 | **2026-08-16**，Marco Liu，阅读 89 | ✅✅ 专门讲国内镜像（见 §4.4）；⚠️ **文章事实错误**（见 §5） |
| https://www.ai-master.cc/article/voice-004 | VoiceBox 深度解读：开源语音合成工作室与多模态语音生成新范式 | 2026-04-22 | 与其他开源方案（VoxCPM2、Bark、CosyVoice）对比 |
| https://jishuzhan.net/article/2057617588393971714 | Voicebox 深度指南：开源本地 AI 语音工作室完整评测与上手教程 | 2026-05-22 | 引擎参数对比表 |
| https://liudon.com/posts/voice-cloning-solution-comparison/ | 2026年音色克隆方案对比：IndexTTS-2、CosyVoice、GPT-SoVITS、Fish Speech、VoxCPM 部署与实测 | 2026-05-18，更新 2026-09-09，Liudon | ⚠️ **完全没提 Voicebox**（见 §6） |
| http://www.framerc.cn/news/379722/ | Meta Voicebox有什么特点？研究性质强，实用性不如CosyVoice3 | 2026-08-11 | ⚠️ 讲的是 **Meta Voicebox**，非本项目（§5） |
| https://voicebox.ndjp.net/ | Voicebox - 开源语音克隆工作室（非官方中文落地页，标注 v0.3.0，已过期） | 未核实 | 无 |
| https://skills.yangsir.net/skill/daily-voicebox-voice-synthesis | voicebox-voice-synthesis · tts（第三方 skill 包） | 2026-08-13 | 无 |

### 3.8 小众软件（appinn）—— ❌ 无收录

- 抓取 `https://www.appinn.com/?s=voicebox` → **HTTP 200 但 0 篇相关文章**（页面 `entry-title` 数量 = 0，全部为站点通用推荐链接）。2026-09-16。

---

## 4. 中国大陆可用性（本次调研最重要的部分）

### 4.1 GitHub 中文 issue —— 一手证据（全部 2026-09-16 抓取）

| Issue | 标题 | 作者 | 创建日 | 状态 | 关键内容 |
|---|---|---|---|---|---|
| **#546** | **下载模型源 url 要是能自己定义就方便了** | **guhaizhous** | **2026-04-24** | open | 正文全文：> "**很多国内的朋友无法科学上网，上不了huggingface网站，下载不了模型**" |
| **#445** | **voicebox 0.4.0无法下载模型 Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice** | **2ndLines** | **2026-04-17** | open | 正文："如题。日志：`2026-04-17 14:19:21,020 - backend.utils.progress - ERROR - Marked qwen-custom-voice-1.7B as error: check_model_inputs() missing 1 required positional argument: 'func'`…`Download error for qwen-custom-voice-1.7B, closing SSE connection`" |
| **#991** | **0.5.0版本下载模型总是报错。切回到0.4.5就可以正常下载模型了。。。** | **wolfprince12** | **2026-08-06** | open | 正文全文：> "0.5.0版本下载模型总是报错。切回到0.4.5就可以正常下载模型了。。。" |
| **#615** | **Voicebox 0.5.0 所有模型无法生成语音（macOS M4, 代理正常, OpenVox 工作正常）** | **13828726319** | **2026-05-05** | open | 中英双语。> "网络：可访问 huggingface.co（**代理已开启**）…已下载的模型（部分显示"已加载"）…**所有模型均无法生成语音，没有任何音频输出。软件不报错，日志为空。** 相同网络环境下，另一款软件 OpenVox 可以正常生成语音。已尝试过重启软件、重启电脑、切换代理、关闭代理、重装模型等操作，问题依旧。" |
| **#1038** | 声音克隆失败 | leemax1688 | 2026-08-11 | open | 仅一张截图 |
| **#751** | 在Mac电脑上怎么使用GPU克隆声音 | hyu18866 | 2026-06-14 | open | "在Mac电脑上，软件默认使用CPU来加载模型、克隆声音。怎么让软件使用GPU？？？" |
| **#193** | How to install the qwen TTS model offline | z583463990-code | 2026-02-25 | open | 中英混合，求离线安装 Qwen TTS 模型 |
| #120 | Impossible to download model | judicalp（英文帖） | 2026-02-20 | open | 6 条评论，其中 **go-restream（2026-03-02）发布了完整的中文手动下载方案**（见下） |

**#741（关键，中英双语）：** "Memory leak on Windows 10 (AMD CPU + RX 5700 XT) and request for offline model download support"
- 作者 **lovexiaoyan**，**2026-06-10**，open
- **Issue 2 原文（英文）：** > "**Model / CUDA binary downloads are frequently interrupted from mainland China.** When Voicebox attempts to auto-download models (from HuggingFace, PyTorch binaries, etc.), the downloads are often **interrupted or fail entirely** due to network restrictions in mainland China. HuggingFace and many CDN endpoints are either very slow or intermittently blocked, making the auto-download flow unreliable."
- Feature Request：允许通过 `VOICEBOX_MODELS_DIR` 指向本地预下载模型 → > "Clear documentation on the expected directory structure so users can mirror the models manually (e.g., via a **domestic mirror** or a **VPN'd one-time download**)"

**#120 的中文解决方案（go-restream，2026-03-02，来自 GitHub 评论）：**
> "# Voicebox 模型下载问题解决方案 通过手动下载解决 Voicebox 模型下载问题。
> ## 适用环境说明 | 平台 | 支持状态 | | macOS | ✅ 主要测试环境 | | Windows | ⚠️ 可参考类似解决思路 |
> **注意**：MLX 后端 0.6B 和 1.7B 共享同一个 HuggingFace 仓库：`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`
> 一、安装 huggingface-cli：`brew install huggingface-cli`
> 二、手动下载模型：`hf download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`
> 三、默认模型下载位置：`~/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-bf16`
> 四、调试接口：`curl http://127.0.0.1:17493/models/status`、`curl http://127.0.0.1:17493/tasks/active`"
> （后续 **mkisamefloweroflotus**，2026-04-15："perfec I fixed"）

### 4.2 ⭐ 官方尚未提供中国下载源：PR #1075 仍处于 open 状态

- **URL：** https://github.com/jamiepine/voicebox/pull/1075
- **标题：** `feat(models): add ModelScope as a model download source`
- **作者：** **JnyRoad**（中文 ID）· **创建 2026-08-27** · **状态：open，未合并**（`merged: false`，2026-09-16 抓取）
- **PR 正文原文关键句：**
  > "**Chinese users hitting slow or blocked downloads from huggingface.co can now select ModelScope in Settings so model weights come from modelscope.cn instead, with no VPN required.**"
  > "`ModelConfig` gains an optional `ms_repo_id`. **13 of 17 registered models have a verified ModelScope mirror** (confirmed live against the modelscope.cn API before wiring anything up); the other 4 (Chatterbox Multilingual/Turbo, TADA 1B/3B) have none…"
  > "An earlier iteration also added an "HF Mirror" source (`HF_ENDPOINT=https://hf-mirror.com`)… Running the actual implementation against a real backend (not mocks) showed **`hf-mirror.com` now just 308-redirects to the real huggingface.co instead of proxying, which `huggingface_hub` itself refuses to trust** — and which would fail for a genuinely blocked user too. **Removed rather than shipped unverified**"
- **⚠️ 极其重要的推论：hf-mirror.com 在 2026 年 8 月已不再真正反代，而是 308 跳回 huggingface.co。** 这意味着 B站评论与多篇中文博客推荐的 `HF_ENDPOINT=https://hf-mirror.com` 方案在 2026 年下半年**可能已经失效**。该结论来自 PR 作者的实测描述（**未经我方独立复现，标注为"第三方实测声明"**）。

### 4.3 GitCode 博客（2026-09-08）—— 官方 troubleshooting 的中文化解读

- **URL：** https://blog.gitcode.com/ff51c50864f831769bcb10767d3e0d22.html
- **原文关键段落（转述官方文档）：**
  > "Voicebox 的模型不会在安装时预下载，而是在你第一次用某个 TTS 引擎生成时，**自动从 HuggingFace Hub 下载**该引擎的模型并初始化"
  > "**模型一律来自 HuggingFace Hub**"
  > "如果卡在 "Downloading..." 或出现 "Failed to download model"…检查网络连接，并确认 HuggingFace Hub 可用…**如果所在地区无法访问 HuggingFace，尝试使用 VPN —— 文档明确列出了这一项。**"
  > "手动下载：用 HuggingFace CLI 把模型下到缓存目录…`pip install huggingface_hub` / `huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base`"
- 模型体积表（本报告照录，未独立核实）：Kokoro 350MB / LuxTTS 300MB / Qwen TTS 0.6B 1.2GB / **Qwen TTS 1.7B 3.5GB（~6GB VRAM）** / Chatterbox Multilingual 3.2GB / TADA 3B 8GB。
- 缓存路径：Windows `%USERPROFILE%\.cache\huggingface\hub\`；可用 `VOICEBOX_MODELS_DIR` 覆盖。
- **低带宽建议：官方 troubleshooting 建议从 Kokoro（~350MB）或 LuxTTS（~300MB）起步，而非默认的 Qwen 1.7B（~3.5GB）。**

### 4.4 AI智能范式网（2026-08-16）—— 国内镜像配置指南（但要付费解锁全文）

- **URL：** https://intelliparadigm.com/article/weixin_31715171/2222240
- **抓取：** HTTP 200，760,345 bytes，2026-09-16
- **正文（可见部分）：**
  > "作为一名长期折腾AI工具的开发者，我深知在**国内使用Hugging Face生态工具的痛点**。最近在测试Voicebox语音合成工具时，发现**其依赖的模型默认从huggingface.co下载，速度慢不说，还经常中断**。经过一番摸索，我总结出一套完整的解决方案，**实测下载速度提升10倍以上**。"
  > "**终极解决方案：设置 HF_ENDPOINT 环境变量**…变量名：`HF_ENDPOINT` 变量值：`https://hf-mirror.com`…注意：此方法不仅适用于Voicebox，所有基于 Hugging Face 生态的工具都会自动生效"
  > "国内开发者社区维护的 hf-mirror.com 镜像站同步更新主流模型，且服务器位于国内，**下载速度可达50MB/s以上**。"
- ⚠️ **全文被"解锁全文 已有2w+人解锁"付费墙截断**，第 3 种方法未获取（未核实）。

### 4.5 ⭐ 中国社区自己做的绕行补丁 / 网盘分发

1. **B站国区补丁（BV1no5L6nE1o）** — 见 §1.7。UP主 Tech指南 发布 `voicebox-patch-v1.0.5-voicebox-patch-darwin-arm64`，通过百度网盘 + 夸克网盘分发，声称"模型下载速度 x100"，并解决"生成语音时拉取配置卡住"。（**补丁二进制未验证，安全性未核实**）
2. **网盘预打包模型**（Tech指南）：百度网盘 `pan.baidu.com/s/1DFjn0u5KpO31YkP7dh159A?pwd=0503`（含 qwen tts 0.6b 和 1.7b 压缩包）+ 夸克网盘 `pan.quark.cn/s/ab2d66231a6e`。
3. **CUDA 后端手动装配**（B站 @没事儿转转，2026-07-01）：从 GitHub Releases 手动下 `cuda-libs-cu128-v1.tar.gz` + `voicebox-server-cuda.tar.gz`，用**迅雷**断点续传，解压到 `%APPDATA%\sh.voicebox.app\backends\cuda\` 并创建 `cuda-backend` 标记文件。→ **说明 Voicebox 的自动 CUDA 下载在中国同样不可靠。**
4. **HF_ENDPOINT=hf-mirror.com**（博客园 2026-04-25；B站评论 2026-05-24）→ 但据 PR #1075，该镜像可能已失效。

### 4.6 ⭐ 中文汉化 / 镜像站的民间尝试

| 资源 | URL | 事实 |
|---|---|---|
| `bbylw/voicebox-cn` | https://github.com/bbylw/voicebox-cn | **19 stars / 3 forks / 非 fork（独立仓库）/ 语言 HTML / created 2026-03-19T13:12:25Z / pushed 2026-03-19T13:27:36Z（仅存活 15 分钟即停止更新）**。描述："Voicebox 开源语音合成工作室。 克隆声音。生成语音。应用效果。构建语音驱动的应用。 全部在您的本地机器上运行。" homepage = **https://voicebox.ndjp.net/**。README 是**原仓库 README 的机翻**（"Voicebox 是什么？…**5 种 TTS 引擎**…23 种语言"）。→ **不是真正的程序汉化，只是中文 README + 中文落地页，且内容停留在 v0.3.0 时代。** |
| `voicebox.ndjp.net` | https://voicebox.ndjp.net/ | 非官方中文落地页，标注 **v0.3.0**（截至 2026-09-16 已严重过期——官方已 v0.5.0）。示例音色用的是 Morgan Freeman / Bob Ross / Scarlett Johansson / David Attenborough / Linus Tech Tips 等英文名人音色。 |
| PR #990 | https://github.com/jamiepine/voicebox/pull/990 | **CherryKingOne**，2026-08-06，**已关闭（closed，未合并）**。标题：`添加中文启动指南：Conda后端 + Bun前端双端启动方法`。CodeRabbit 摘要确认："**Added a comprehensive Chinese getting-started guide**…**Translated the README into Chinese** and expanded MCP client and voice configuration guidance." → **中文文档的官方化尝试，最终未合并。** |
| gaojulong/voicebox | （搜索可见但 API 404，**未核实**，疑似已删除或改名） | 搜索结果中出现过 `gaojulong/voicebox`（2026-04-17），抓取时 `api.github.com/repos/gaojulong/voicebox` 返回空 → **该仓库现已不可访问，未核实** |

### 4.7 中文发音 / 语速问题（与"能否用"并列的"是否好用"）

- **knightli.com（2026-07-26）** 部署指南专门设了「**中文发音不自然**」章节（正文仅给出排查方向，具体结论未在抓取片段中，**部分未核实**）。
- **B站 @微小鱼189（2026-09-16，当天评论）**："我生成出来的感觉**语速都偏快**，不管Qwen TTS 1.7B还是0.6B，都明显偏快。"
- **博客园/iTech 公众号文** 称 Whisper 转录已专门做 "CJK 检测：专门处理中日韩无空格文本的循环" → 说明项目对 CJK 有针对性工程，但这属于 STT 侧，不是 TTS 音质侧。
- **B站 @lnkikik（2026-05-26）** 指出评测盲点："测平淡语气的声音没什么参考价值"，并质疑情绪/停顿表达。**这是中文社区对 Voicebox 最有技术含量的一条批评。**

---

## 5. 名称混淆问题（中文社区的系统性误解）—— 重要发现

**jamiepine/voicebox（2026，本地语音工作室）与 Meta 2023 年的学术项目 Voicebox 在中文社区被大量混淆。**

已核实的混淆案例：

1. **AI智能范式网（2026-08-16）** https://intelliparadigm.com/article/weixin_31715171/2222240
   > "**Voicebox作为Meta开源的文本转语音工具**，依赖多个 Hugging Face 托管的预训练模型（如Qwen3-TTS）"
   → **事实错误**：jamiepine/voicebox 并非 Meta 出品，Qwen3-TTS 来自阿里通义。该文却在教人配置 jamiepine/voicebox 的 HF 镜像。

2. **framerc.cn（2026-08-11）** http://www.framerc.cn/news/379722/
   > 标题："Meta Voicebox有什么特点？研究性质强，实用性不如CosyVoice3"
   > "Meta推出的Voicebox…**Voicebox没有开源模型权重，缺乏中文支持，也没有Web界面**…反观阿里巴巴通义实验室开源的 CosyVoice3…"
   → 讲的是 Meta Voicebox，**与本项目无关**。但极易被搜索者误读为对本项目的评价。

3. **掘金** 搜索结果里出现的 Voicebox 文章全部是 Meta 2023 项目相关。

4. **B站 2023 年视频**（§1.4）占据 "voicebox 语音" 关键词头部。

5. **反面案例（做对了的）：** CSDN/AtomGit 的谢白羽（2026-05-21）在文章开头主动澄清："不是 Meta 于 2023 年发布的学术研究项目 Voicebox"。

**结论：中文社区对 "Voicebox" 这一名称的搜索意图高度不纯**，任何中文 SEO/舆情分析都必须先把 Meta Voicebox 的噪声剥离。

---

## 6. 与替代方案的对比（GPT-SoVITS / CosyVoice / Fish Speech / IndexTTS / 豆包 / MiniMax）

### 6.1 ❌ 最权威的中文对比文章里，Voicebox 根本没被收录

- **URL：** https://liudon.com/posts/voice-cloning-solution-comparison/
- **标题：** 2026年音色克隆方案对比：IndexTTS-2、CosyVoice、GPT-SoVITS、Fish Speech、VoxCPM 部署与实测
- **作者：** Liudon · 发布 2026-05-18 · **更新 2026-09-09** · 5954 字 · 12 分钟阅读
- **抓取：** HTTP 200，152,696 bytes，2026-09-16
- **正文全文 0 次提及 "Voicebox"**（已用正则全文检索确认）。
- 该文测试环境："AutoDL 平台…GPU RTX 4090(24GB) * 1"，测试文本 "Welcome to Beijing, President Trump.北京欢迎你，特朗普总统"。
- 文中给的是中国用户的标准做法（**与 Voicebox 形成对照**）：
  > "# 国内可换镜像 `uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"`
  > "# 国内可设置HF镜像 `export HF_ENDPOINT="https://hf-mirror.com"`
  > "**方式二：modelscope** `modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints`"
- **含义：** 在中文圈最认真的音色克隆横评中，Voicebox 连被评估的资格都还没有；而 IndexTTS-2 / CosyVoice 等因为原生支持 ModelScope / 阿里云镜像，对中国用户更友好。

### 6.2 B站对比类视频同样不含 Voicebox

| BV | 标题 | 与本项目关系 |
|---|---|---|
| **BV1P541117yn** | 你的声音，现在是我的了！- 手把手教你用 GPT-SoVITS 克隆声音！ | UP主 痕继痕迹，**5,404,035 播放 / 1,452 弹幕 / 310,516 收藏**（2024-01-26）→ **中文圈声音克隆事实标准，规模是 Voicebox 头部视频的 ~60 倍** |
| BV1hukmYhEJX | 本地部署TTS 怎么选？GPT-SoVITS / CosyVoice / FishSpeech / F5-TTS 全面对比 | **不含 Voicebox** |
| BV1BWVozgEwt | CosyVoice2网页版 AI声音克隆在线工具 免费强大还能说方言 | 73,793 播放（2025-05-10） |
| BV1G3okYREEr | 粤语AI语音克隆教程：GPT-SoVITS-V3应用指南 | 9,247 播放 |
| BV1Sv62BjETY | 二创声音克隆——保持音色和音调语气，说任何话！ | 89,290 播放（与本项目无关，但显示该赛道热度） |

### 6.3 已发现的少量"对比"讨论

- **B站 BV1rqGB6yE7i 评论区 @lnkikik（2026-05-26，8 赞）** 把评价标准从"能不能出声"提升到"情绪与停顿"——这是唯一一条指向"与其它模型横向比"的中文用户讨论。
- **ai-master.cc（2026-04-22）** https://www.ai-master.cc/article/voice-004 声称做"与其他开源方案（VoxCPM2、Bark、CosyVoice）的全面对比"，但抓取未获取到该对比表全文（**未核实**）。
- **GitHub issue #615（中文用户，2026-05-05）** 提到："相同网络环境下，另一款软件 **OpenVox** 可以正常生成语音" → 中文用户的实际对比对象是 **OpenVox**（同类本地语音工具），而非 GPT-SoVITS。
- **未找到**任何中文用户把 Voicebox 与**豆包/MiniMax 语音 API**做对比的帖子（**未核实 / 未发现**）。

### 6.4 与 ElevenLabs 的对比（中文语境里最常见的定位）

中文内容几乎统一把 Voicebox 定位成 "**ElevenLabs 的本地开源平替**"：
- B站 BV1NSdoBrEKk："免费开源的 **ElevenLabs 平替**"
- B站 BV1np9ZBwETa 描述："还在花钱买 ElevenLabs 会员？担心上传声音到云端泄露隐私？"
- 腾讯云 article/2638563："把它当作 **ElevenLabs 的本地、免费的开源替代品**"
- CSDN/AtomGit 谢白羽：做了 Voicebox vs ElevenLabs vs WisprFlow 三维对比表（数据位置 / 费用模式 / Agent 集成 / 账号依赖）
- iTech 公众号文：表格对比 Voicebox（免费 MIT）vs ElevenLabs（$22-$99/月）

---

## 7. 中文覆盖度评估：**薄**

汇总证据（全部 2026-09-16 抓取）：

| 平台 | 是否存在 Voicebox 讨论 | 证据 |
|---|---|---|
| B站 | ⚠️ **弱存在** | 17 条视频，最高 87.7k 播放，但**弹幕仅 52**，长尾全 <700 播放；小写 `voicebox` 关键词 B站搜索召回 **0** |
| V2EX | ❌ **零** | sov2ex `voicebox` = 0，`jamiepine` = 0 |
| 小众软件 appinn | ❌ **零** | 站内搜索 0 篇 |
| 掘金 | ❌ **零（仅 Meta 同名项目）** | juejin search API 只返回 Meta Voicebox |
| 知乎 | ✅ 有（至少 3 篇专栏） | 但正文 **403 反爬未读** |
| 少数派 | ✅ **1 篇（派评编辑推荐）** | sspai.com/post/109135，2026-04-27 |
| CSDN/AtomGit | ✅ 1 篇高质量长文 | 1220 浏览 |
| 博客园 | ✅ 2 篇 | 含 1 篇纯 HF 排障备忘（阅读 193） |
| 腾讯云开发者社区 | ✅ 2 篇 | 1 篇 1.9K 浏览；1 篇来自微信公众号 |
| 微信公众号 | ✅ 至少 2 个来源 | ①「AI人工智能时代」(the-ai-era) 经 cnblogs/itech 同步；② 2026-02-18 一篇经腾讯云"自媒体同步曝光计划" |
| GitHub 中文 issue | ✅ **显著存在** | 至少 10 条中文/中英双语 issue，覆盖模型下载失败、命名冲突、离线安装、CUDA 下载中断 |
| 小红书 | ❓ **未核实** | 未找到可抓取的公开入口，**无法确认有无** |
| QQ群讨论 | ❓ **未核实** | 未找到任何公开可见的 QQ 群内容 |

**判断：Voicebox 在中文社区属于"被科技自媒体介绍过、被少数极客试用过、但尚未形成社区"的阶段。** 与之对照，GPT-SoVITS 在中国有 540 万播放的官方教程、V2EX/掘金/知乎的大量讨论；Voicebox 没有。同时，**中文 GitHub issue 的密度远高于中文论坛的密度** —— 说明真正去装的中国人不多，但装了的人确实被网络问题卡住过。

---

## 8. 明确未能核实 / 被阻断的部分

| 项 | 状态 |
|---|---|
| B站 **不带 cookie** 的搜索 API | ❌ **`-412` 风控**（返回 captcha 页）。已改用带 `buvid3` 的 wbi 接口绕过，**播放量/弹幕数均取自 `view` 接口，可靠** |
| 少数派该文章的具体**点赞/评论数** | ⚠️ 抓取到页面但未提取互动数（**未核实**） |
| 知乎 3 篇文章的正文 | ❌ **HTTP 403 反爬**，仅得搜索摘要 |
| 知乎文章的确切发布日期 | ⚠️ **未核实**（403） |
| intelliparadigm.com 的第 3 种镜像方案 | ⚠️ **付费墙截断**（"解锁全文 已有2w+人解锁"） |
| B站国区补丁 **二进制内容与安全性** | ⚠️ **未核实**（网盘分发，无法验证） |
| `gaojulong/voicebox` 仓库 | ⚠️ **未核实**（搜索引擎可见，GitHub API 返回空/404） |
| **小红书**是否存在 Voicebox 内容 | ⚠️ **未核实**（无可抓取入口） |
| **QQ群**讨论 | ⚠️ **未核实**（无公开内容） |
| hf-mirror.com 是否真的已失效 | ⚠️ **第三方声明**（来自 PR #1075 作者实测描述），**我方未独立复现** |
| 豆包/MiniMax API 与 Voicebox 的对比 | ⚠️ **未发现**任何中文对比内容 |
| 各 B站视频的"投币/点赞/分享" | 仅头部 3 条通过 `view` 接口取到，其余标记"未核实"（**搜索接口不提供这些字段**） |
| B站各视频评论总数 | 仅头部 3 条取到（930 / 224 / 93）；其余未逐一抓取 |
| 腾讯云/CSDN 文章的阅读数是否为"当前值" | 阅读数为页面自述快照，可能随时间变化 |
| `web_fetch` | ❌ 全程不可用（DNS 被劫持到非公网 IP），已完全改用 `curl` |

---

## 9. 抓取命令留档（可复现）

```bash
# Bilibili：带 buvid3 cookie 的 wbi 搜索（不带 cookie 会 -412）
curl -sL -A "Mozilla/5.0 (...)" -H "Referer: https://www.bilibili.com/" \
  -b "buvid3=$(uuidgen)-infoc; b_nut=1750000000" \
  "https://api.bilibili.com/x/web-interface/wbi/search/type?search_type=video&keyword=Voicebox<&page=1>"

# Bilibili：视频详情（无需 cookie）
curl -sL -A "Mozilla/5.0" "https://api.bilibili.com/x/web-interface/view?bvid=BV1np9ZBwETa"

# Bilibili：评论（ps 必须 <=20，否则 -400）
curl -sL -A "Mozilla/5.0" -H "Referer: https://www.bilibili.com/" \
  "https://api.bilibili.com/x/v2/reply?type=1&oid=116508789382766&sort=2&ps=20&pn=1"

# V2EX 全文搜索
curl -sL "https://www.sov2ex.com/api/search?q=voicebox&size=20"

# GitHub：中文 issue
curl -sL -H "Accept: application/vnd.github+json" \
  "https://api.github.com/search/issues?q=repo:jamiepine/voicebox+%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD&per_page=10"
```

---

**文件生成时间：2026-09-16 · 全部 URL 均在当日以 curl / API 实际访问并取得 HTTP 响应（成功状态已逐条标注）**
