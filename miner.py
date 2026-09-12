"""学习侧：黑话提取 + 含义推断管线。

移植自 MaiBot 的 jargon_miner / learn_jargon 思路：
1. 攒够一批群聊消息后，让 LLM 提取"可能是黑话"的候选词（要求带 source_id 以便回溯证据）
2. 候选词入库计次，count 跨过阶梯阈值（4/8/25/100）时触发含义推断
3. 含义推断走三次 LLM 调用（MaiBot 的"对照实验"设计）：
   - 带上下文推断：结合证据对话片段猜含义（信息不足可回答 no_info 搁置）
   - 裸词条推断：只给词本身再猜一次
   - 对比两次结果：若一致说明是通用词，判定不是黑话；有差异才说明依赖圈内语境
"""

import asyncio
import json
import re

from astrbot.api import logger
from astrbot.api.star import Context

from .store import JargonStore, INFERENCE_THRESHOLDS

# ---------- Prompt 模板（移植自 MaiBot prompts/zh-CN/learn_jargon 等） ----------

EXTRACT_PROMPT = """请从下面这段群聊记录中提取"可能是黑话"的候选项（黑话/俚语/网络缩写/口头禅）。

提取规则：
- 必须为对话中真实出现过的短词或短语
- 必须是你无法理解含义、或者需要当前聊天圈内语境才能理解的词语
- 不要选择含义清晰的普通词语
- 排除：人名、@、表情包/图片中的内容、纯标点、常规功能词（如的、了、呢、啊等）
- 每个词条长度建议 2-8 个字符（不强制），尽量短小
- 请尽量提取所有可能的黑话，最多 {extract_max} 个

黑话必须为以下几种类型：
- 由字母构成的，汉语拼音首字母的简写词，例如：nb、yyds、xswl
- 英文词语的缩写，用英文字母概括一个词汇或含义，例如：CPU、GPU、API
- 中文词语的缩写，用几个汉字概括一个词汇或含义，例如：社死、内卷
- 群聊内部反复使用、但脱离上下文不容易理解的短词或短语

输出要求：
请仅输出 JSON 数组，不要输出重复内容。每个元素为一个对象，字段名如下：
[
  {"content": "词条", "source_id": "12"},
  {"content": "词条2", "source_id": "5"}
]
- content：黑话候选词条的原文
- source_id：该黑话对应的来源编号，即聊天记录中 <message source_id="3"> 的数字

聊天记录：
{chat_str}

输出 JSON："""

INFER_WITH_CONTEXT_PROMPT = """你是一个网络流行语专家。请根据以下群聊上下文，推断词条「{content}」在该聊天圈语境下的含义。

{raw_content_list}
{previous_meaning_section}
要求：
- 只依据上下文进行推断，不要臆测
- 如果上下文不足以推断含义，请将 no_info 设为 true
{previous_meaning_instruction}
请仅输出 JSON：{"meaning": "推断出的含义", "no_info": false}"""

INFER_CONTENT_ONLY_PROMPT = """请仅根据词条本身推断「{content}」的可能含义。
从网络流行语、拼音/英文缩写、中文缩略的角度猜测，不要依赖任何聊天上下文。
请仅输出 JSON：{"meaning": "推断出的含义"}"""

COMPARE_PROMPT = """以下是对同一个网络词语的两次含义推断：

推断A（结合聊天上下文）：
{inference1}

推断B（仅凭词语本身）：
{inference2}

请判断两次推断的含义是否基本一致。
- 如果基本一致，说明该词是通用网络用语，而非小圈子黑话
- 如果有明显差异，说明该词的含义依赖特定语境，是圈子黑话

请仅输出 JSON：{"is_similar": true}"""

# ---------- 工具函数 ----------

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


def _parse_json(text: str):
    """宽松解析 LLM 输出中的 JSON（去代码围栏 + 截取首个括号区间）。"""
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text).strip()
    for start_ch, end_ch in (("[", "]"), ("{", "}")):
        start = cleaned.find(start_ch)
        end = cleaned.rfind(end_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _is_valid_term(content: str) -> bool:
    if not content:
        return False
    if not (2 <= len(content) <= 12):
        return False
    if "@" in content or "http" in content.lower():
        return False
    return True


class JargonMiner:
    def __init__(self, context: Context, store: JargonStore, config: dict):
        self.context = context
        self.store = store
        self.config = config
        self.extract_max = int(config.get("extract_max", 30))
        self._learn_locks: dict[str, asyncio.Lock] = {}
        self._infer_sem = asyncio.Semaphore(2)  # 限制并发推断，避免突发打爆 LLM

    # ---------- LLM 调用 ----------

    def _get_provider(self, umo: str | None = None):
        # 优先用配置的专用模型（分析黑话没必要动用主模型）
        pid = str(self.config.get("provider_id") or "").strip()
        if pid:
            try:
                provider = self.context.get_provider_by_id(pid)
                if provider:
                    return provider
                logger.warning(f"[jargon] 配置的模型 {pid} 不存在，回退默认模型")
            except Exception as e:
                logger.warning(f"[jargon] 获取指定模型失败: {e}")
        try:
            provider = self.context.get_using_provider(umo)
            if provider:
                return provider
        except Exception as e:
            logger.debug(f"[jargon] get_using_provider 失败: {e}")
        providers = self.context.get_all_providers()
        return providers[0] if providers else None

    async def _chat(self, prompt: str, umo: str | None = None) -> str:
        provider = self._get_provider(umo)
        if not provider:
            logger.warning("[jargon] 没有可用的 LLM 提供商，跳过")
            return ""
        try:
            resp = await provider.text_chat(prompt=prompt)
            return (resp.completion_text or "").strip()
        except Exception as e:
            logger.error(f"[jargon] LLM 调用失败: {e}")
            return ""

    # ---------- 提取 ----------

    async def learn_from_chat(self, umo: str, messages: list[dict]) -> None:
        """对一批群聊消息执行一次黑话提取。

        messages: [{"source_id": int, "sender": str, "text": str}, ...]
        """
        lock = self._learn_locks.setdefault(umo, asyncio.Lock())
        async with lock:
            try:
                await self._learn_locked(umo, messages)
            except Exception as e:
                logger.error(f"[jargon] 黑话提取异常: {e}")

    async def _learn_locked(self, umo: str, messages: list[dict]) -> None:
        chat_str = "\n".join(
            f'<message source_id="{m["source_id"]}">{m["sender"]}: {m["text"]}</message>'
            for m in messages
        )
        prompt = EXTRACT_PROMPT.replace("{extract_max}", str(self.extract_max))
        prompt = prompt.replace("{chat_str}", chat_str)
        raw = await self._chat(prompt, umo)
        items = _parse_json(raw)
        if not isinstance(items, list):
            logger.debug("[jargon] 提取结果解析失败或为空")
            return

        by_source = {m["source_id"]: m for m in messages}
        seen: set[str] = set()
        infer_tasks = []
        for item in items[: self.extract_max]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not _is_valid_term(content) or content in seen:
                continue
            seen.add(content)

            # 用 source_id 回溯证据片段（该消息 + 前后各 1 条邻居）
            try:
                sid = int(item.get("source_id"))
            except (TypeError, ValueError):
                sid = None
            evidence = self._build_evidence(messages, by_source, sid)

            rec, should_infer = self.store.upsert_candidate(content, umo, evidence)
            if should_infer and rec:
                infer_tasks.append(rec["id"])

        if seen:
            logger.info(f"[jargon] 疑似黑话: {', '.join(sorted(seen))}")
        for jargon_id in infer_tasks:
            asyncio.create_task(self.infer_meaning(jargon_id, umo))

    @staticmethod
    def _build_evidence(
        messages: list[dict], by_source: dict[int, dict], sid: int | None
    ) -> str | None:
        """围绕 source_id 截取证据片段，最多 3 条消息、200 字符。"""
        if sid is None or sid not in by_source:
            return None
        idx = next(
            (i for i, m in enumerate(messages) if m["source_id"] == sid), None
        )
        if idx is None:
            return None
        window = messages[max(0, idx - 1) : idx + 2]
        snippet = "\n".join(f"{m['sender']}: {m['text']}" for m in window)
        return snippet[:200]

    # ---------- 含义推断（双路对照） ----------

    async def infer_meaning(self, jargon_id: int, umo: str | None = None) -> None:
        async with self._infer_sem:
            try:
                await self._infer_locked(jargon_id, umo)
            except Exception as e:
                logger.error(f"[jargon] 含义推断异常 id={jargon_id}: {e}")

    async def _infer_locked(self, jargon_id: int, umo: str | None) -> None:
        rec = self.store.get(jargon_id)
        if not rec or rec["created_by"] == "manual":
            return
        content = rec["content"]
        count = rec["count"]

        try:
            evidence: list[str] = json.loads(rec["evidence"] or "[]")
        except (json.JSONDecodeError, TypeError):
            evidence = []
        evidence = [e for e in evidence if e and e.strip()]
        if not evidence:
            logger.warning(f"[jargon] {content} 无可用证据，跳过本次推断")
            self.store.touch_last_inference(jargon_id, count)
            return

        previous_meaning = (rec["meaning"] or "").strip()
        previous_meaning_section = ""
        previous_meaning_instruction = ""
        if previous_meaning:
            previous_meaning_section = (
                f"\n**上一次推断的含义（仅供参考）**\n{previous_meaning}\n"
            )
            previous_meaning_instruction = (
                "- 请参考上一次推断的含义，结合新的上下文，给出更准确或更新的推断\n"
            )

        # 步骤 1：带上下文推断
        ctx_text = "\n\n".join(
            f"【对话片段 {i}】\n{e}" for i, e in enumerate(evidence, 1)
        )
        prompt1 = (
            INFER_WITH_CONTEXT_PROMPT.replace("{content}", content)
            .replace("{raw_content_list}", ctx_text)
            .replace("{previous_meaning_section}", previous_meaning_section)
            .replace("{previous_meaning_instruction}", previous_meaning_instruction)
        )
        raw1 = await self._chat(prompt1, umo)
        inference1 = _parse_json(raw1)
        if not isinstance(inference1, dict):
            logger.warning(f"[jargon] {content} 推断1解析失败")
            return
        if inference1.get("no_info") or not str(
            inference1.get("meaning") or ""
        ).strip():
            logger.info(f"[jargon] {content} 信息不足，搁置待下次")
            self.store.touch_last_inference(jargon_id, count)
            return
        meaning1 = str(inference1["meaning"]).strip()

        # 步骤 2：裸词条推断
        prompt2 = INFER_CONTENT_ONLY_PROMPT.replace("{content}", content)
        raw2 = await self._chat(prompt2, umo)
        inference2 = _parse_json(raw2)
        if not isinstance(inference2, dict):
            logger.warning(f"[jargon] {content} 推断2解析失败")
            return

        # 步骤 3：对照判定——相似则不是黑话，有差异才是黑话
        prompt3 = COMPARE_PROMPT.replace(
            "{inference1}", json.dumps(inference1, ensure_ascii=False)
        ).replace("{inference2}", json.dumps(inference2, ensure_ascii=False))
        raw3 = await self._chat(prompt3, umo)
        comparison = _parse_json(raw3)
        if not isinstance(comparison, dict):
            logger.warning(f"[jargon] {content} 对比解析失败")
            return

        is_similar = bool(comparison.get("is_similar", False))
        is_jargon = not is_similar
        finalized_meaning = meaning1 if is_jargon else previous_meaning
        is_complete = count >= INFERENCE_THRESHOLDS[-1]

        self.store.set_inference_result(
            jargon_id, is_jargon, finalized_meaning, count, is_complete
        )
        if is_jargon:
            logger.info(f"[黑话]{content} 的含义是 {finalized_meaning}")
        else:
            logger.info(f"[jargon] {content} 不是黑话")
