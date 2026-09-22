# cmudict-mini.dict.gz — 精简 CMUDict 词典（离线打包）

发音标注英文路由（issue #56）使用的 ARPAbet 词典。

- 来源：CMUdict 0.7b，经 PyPI 包 `cmudict==1.1.3` 附带的数据文件
  （`cmudict/data/cmudict.dict`，BSD 风格许可，见同目录 `cmudict-mini-LICENSE`，
  原始 README 见 `cmudict-mini-README`）。
- 精简口径（一次性离线生成，运行时零联网）：
  1. 丢弃注释行（`;;;` 开头）与变体词（词形含数字，如 `record(1)`）；
  2. 只保留词形匹配 `[a-z][a-z'-]*` 的词（纯小写字母 + 连字符/撇号，专名与非
     ASCII 词形丢弃）；
  3. 每个词只保留第一个发音（标准读音）；
  4. 音素必须匹配 `[A-Z]+[0-2]?`（全大写 + 0/1/2 重音数字）。
- 规模：125,907 词，gzip 后约 864 KB。
- 加载：`voiceclone_sidecar.pronunciation.load_cmudict()` 惰性、进程内缓存，
  仅在用户使用英文发音标注时读取，纯本地文件。

重新生成脚本（保持口径一致）：从 `cmudict==1.1.3` wheel 取
`cmudict/data/cmudict.dict`，按上述 4 条过滤后逐词
`word + " " + " ".join(phones)` 排序输出，gzip -9。
