# astrbot_plugin_jargon 群黑话自动学习系统

让机器人像人一样，偷偷学会群里的黑话。

## 设计思路

本插件的整体设计移植自 **[MaiBot](https://github.com/Mai-with-u/MaiBot)** 的黑话（jargon）模块，特此说明并致谢。

### 学习侧（低频高质，LLM 驱动）

1. **候选提取**：每个会话攒够 `learn_frequency` 条消息后，把聊天记录丢给 LLM，
   让其按规则挑出"可能是黑话"的短词（拼音首字母缩写 / 英文缩写 / 中文缩略 /
   圈内高频短语），并要求返回 `source_id` 以便回溯证据上下文。
2. **入库计次**：候选词按 `content` 去重入库，维护 `count`、`chat_scopes`
   （会话作用域）、`evidence`（证据片段快照，最多 6 段）。
3. **阶梯阈值推断**：count 跨过 **4 / 8 / 25 / 100** 时各触发一次含义推断，
   用 `last_inference_count` 防止重复触发，100 封顶标记 `is_complete`。
4. **双路对照推断**（MaiBot 最核心的设计）：
   - 带证据上下文推断一次含义（信息不足可答 `no_info` 搁置）；
   - 只给裸词条再推断一次；
   - 对比两次结果——**若相似，说明是通用词，判定不是黑话；有差异，才说明含义依赖圈内语境，是真黑话**。
   本质上是"用含义是否依赖上下文"来定义黑话。

### 使用侧（高频低耗，零 LLM）

- 每次 LLM 请求前，对当前 prompt 和最近的用户上下文做**归一化子串机械匹配**；
- 命中的词条按 count 加权排序取前 `inject_max` 条，
  以"黑话参考"块注入 `system_prompt`，仅作理解语境的参考。

## 指令

| 指令 | 说明 | 权限 |
| --- | --- | --- |
| `/jargon list [页码]` | 查看已确认的黑话 | 所有人 |
| `/jargon add 词 含义` | 手动录入（不会被 AI 覆盖） | 管理员 |
| `/jargon del 词` | 删除词条 | 管理员 |
| `/jargon global 词` | 切换全局/本群作用域 | 管理员 |
| `/jargon stat` | 统计信息 | 所有人 |
| `/jargon block 群号` | 把群拉进黑名单（不再学习该群） | 管理员 |
| `/jargon unblock 群号` | 把群移出黑名单 | 管理员 |
| `/jargon blocklist` | 查看黑名单 | 所有人 |

## 配置

见 `_conf_schema.json`：总开关、提取频率、提取/注入上限、是否学习私聊，以及群聊黑名单（`blacklist_groups`，逗号分隔的群号，命中后该群不收集、不学习、不注入）。

## 数据

存储于 `data/plugin_data/astrbot_plugin_jargon/jargon.db`（SQLite），
更新/重装插件不会丢失。

## 许可与致谢

- 设计灵感与核心思路来自 [Mai-with-u/MaiBot](https://github.com/Mai-with-u/MaiBot)。
