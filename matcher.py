"""使用侧：纯机械匹配。

移植自 MaiBot 的 jargon_context_matcher 思路：
- 不做向量检索、不调用 LLM，仅做归一化子串包含匹配
- 按词条 count 打分排序，取前 N 条注入 prompt
"""

import re

_SPACE_RE = re.compile(r"\s+")


def normalize(text: object) -> str:
    """小写化并折叠空白，用于机械匹配。"""
    return _SPACE_RE.sub(" ", str(text or "").strip().lower())


def match_jargons(
    records: list[dict], texts: list[str], limit: int = 10
) -> list[dict]:
    """在若干条文本中机械命中黑话，按权重取前 limit 条。

    Args:
        records: store.get_matchable() 返回的词条列表
        texts: 待匹配文本（按时间顺序，越靠前越早）
        limit: 最大返回条数
    """
    matches: dict[str, dict] = {}
    for index, text in enumerate(texts):
        normalized_text = normalize(text)
        if not normalized_text:
            continue
        for rec in records:
            key = normalize(rec.get("content"))
            if not key or key in matches:
                continue
            if key not in normalized_text:
                continue
            # 打分 = 词条出现次数 - 消息位置微调（越早的消息权重略低）
            score = float(rec.get("count", 0)) - index * 0.01
            matches[key] = {
                "content": rec["content"],
                "meaning": rec["meaning"],
                "count": rec.get("count", 0),
                "score": score,
                "first_index": index,
            }

    return sorted(
        matches.values(),
        key=lambda m: (-m["score"], m["first_index"], -len(m["content"]), m["content"]),
    )[: max(1, int(limit))]


def format_injection(matches: list[dict]) -> str:
    """把命中结果格式化为注入 system prompt 的参考块。"""
    lines = [
        "以下黑话来自当前上下文中其他用户消息的机械匹配，仅作理解聊天语境的参考："
    ]
    for i, m in enumerate(matches, 1):
        lines.append(f"{i}. {m['content']}：{m['meaning']}")
    return "\n".join(lines)
