"""黑话数据存储层（SQLite）。

表结构移植自 MaiBot 的 Jargon 模型：
- content / meaning / count
- chat_scopes: JSON dict {unified_msg_origin: 出现次数}，用于会话作用域
- evidence: JSON list[str]，保留推断用的上下文片段（MaiBot 存消息引用，这里直接存文本快照，避免依赖 AstrBot 消息库）
- last_inference_count / is_complete: 阶梯阈值推断调度
- created_by: ai / manual，手动录入的词条永不被 AI 覆盖
"""

import json
import sqlite3
import threading
import time

# count 到达这些阈值时各触发一次含义推断（移植自 MaiBot JARGON_INFERENCE_THRESHOLDS）
INFERENCE_THRESHOLDS = (4, 8, 25, 100)
MAX_EVIDENCE = 6  # 每个词条最多保留的上下文证据片段数


class JargonStore:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS jargon(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE,
                    meaning TEXT NOT NULL DEFAULT '',
                    count INTEGER NOT NULL DEFAULT 0,
                    is_jargon INTEGER NOT NULL DEFAULT 0,
                    is_complete INTEGER NOT NULL DEFAULT 0,
                    is_global INTEGER NOT NULL DEFAULT 0,
                    created_by TEXT NOT NULL DEFAULT 'ai',
                    chat_scopes TEXT NOT NULL DEFAULT '{}',
                    evidence TEXT NOT NULL DEFAULT '[]',
                    last_inference_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jargon_content ON jargon(content)"
            )
            self._conn.commit()

    # ---------- 基础工具 ----------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
        return dict(row) if row else None

    def get(self, jargon_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jargon WHERE id = ?", (jargon_id,)
            ).fetchone()
        return self._row_to_dict(row)

    def get_by_content(self, content: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jargon WHERE content = ?", (content,)
            ).fetchone()
        return self._row_to_dict(row)

    # ---------- 学习侧 ----------

    def upsert_candidate(
        self, content: str, umo: str, evidence_snippet: str | None
    ) -> tuple[dict | None, bool]:
        """写入/更新一条黑话候选。

        Returns:
            (词条记录, 是否应触发含义推断)
        """
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jargon WHERE content = ?", (content,)
            ).fetchone()
            if row is None:
                scopes = {umo: 1}
                evidence = [evidence_snippet] if evidence_snippet else []
                cur = self._conn.execute(
                    """INSERT INTO jargon
                    (content, count, chat_scopes, evidence, created_at, updated_at)
                    VALUES (?, 1, ?, ?, ?, ?)""",
                    (content, json.dumps(scopes, ensure_ascii=False),
                     json.dumps(evidence, ensure_ascii=False), now, now),
                )
                self._conn.commit()
                new_row = self._conn.execute(
                    "SELECT * FROM jargon WHERE id = ?", (cur.lastrowid,)
                ).fetchone()
                return self._row_to_dict(new_row), False

            rec = dict(row)
            if rec["created_by"] == "manual":
                # 手动记录：AI 只看不改（移植 MaiBot 的行为）
                return rec, False

            scopes = json.loads(rec["chat_scopes"] or "{}")
            scopes[umo] = scopes.get(umo, 0) + 1
            evidence = json.loads(rec["evidence"] or "[]")
            if evidence_snippet and evidence_snippet not in evidence:
                evidence.append(evidence_snippet)
                evidence = evidence[-MAX_EVIDENCE:]

            new_count = rec["count"] + 1
            self._conn.execute(
                """UPDATE jargon SET count = ?, chat_scopes = ?, evidence = ?,
                updated_at = ? WHERE id = ?""",
                (new_count, json.dumps(scopes, ensure_ascii=False),
                 json.dumps(evidence, ensure_ascii=False), now, rec["id"]),
            )
            self._conn.commit()
            rec["count"] = new_count

        should_infer = self._should_infer(rec)
        return rec, should_infer

    @staticmethod
    def _should_infer(rec: dict) -> bool:
        """阶梯阈值调度：count 跨过 4/8/25/100 时各推断一次。

        移植自 MaiBot JargonMiner._should_infer_meaning。
        """
        if rec["created_by"] == "manual" or rec["is_complete"]:
            return False
        count = rec["count"]
        last = rec["last_inference_count"]
        if count < INFERENCE_THRESHOLDS[0] or count <= last:
            return False
        next_threshold = next((t for t in INFERENCE_THRESHOLDS if t > last), None)
        if next_threshold is None:
            return False
        return count >= next_threshold

    def touch_last_inference(self, jargon_id: int, count: int) -> None:
        """仅更新 last_inference_count（用于 no_info 等情况，避免同阈值重复尝试）。"""
        with self._lock:
            self._conn.execute(
                "UPDATE jargon SET last_inference_count = ?, updated_at = ? WHERE id = ?",
                (count, time.time(), jargon_id),
            )
            self._conn.commit()

    def set_inference_result(
        self, jargon_id: int, is_jargon: bool, meaning: str,
        last_inference_count: int, is_complete: bool,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE jargon SET is_jargon = ?, meaning = ?,
                last_inference_count = ?, is_complete = ?, updated_at = ?
                WHERE id = ? AND created_by != 'manual'""",
                (int(is_jargon), meaning, last_inference_count,
                 int(is_complete), time.time(), jargon_id),
            )
            self._conn.commit()

    # ---------- 匹配侧 ----------

    def get_matchable(self, umo: str) -> list[dict]:
        """取当前会话可用的黑话：全局词条，或 chat_scopes 命中本会话的词条。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jargon WHERE is_jargon = 1 AND meaning != '' "
                "ORDER BY count DESC"
            ).fetchall()
        result = []
        for row in rows:
            rec = dict(row)
            if rec["is_global"]:
                result.append(rec)
                continue
            try:
                scopes = json.loads(rec["chat_scopes"] or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if umo in scopes:
                result.append(rec)
        return result

    # ---------- 管理指令 ----------

    def manual_add(self, content: str, meaning: str, umo: str) -> bool:
        """手动录入词条，created_by=manual，不会被 AI 覆盖。返回是否新建。"""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM jargon WHERE content = ?", (content,)
            ).fetchone()
            if row:
                self._conn.execute(
                    """UPDATE jargon SET meaning = ?, is_jargon = 1,
                    created_by = 'manual', updated_at = ? WHERE id = ?""",
                    (meaning, now, row["id"]),
                )
                self._conn.commit()
                return False
            scopes = {umo: 1}
            self._conn.execute(
                """INSERT INTO jargon
                (content, meaning, count, is_jargon, created_by, chat_scopes,
                 created_at, updated_at)
                VALUES (?, ?, 1, 1, 'manual', ?, ?, ?)""",
                (content, meaning, json.dumps(scopes, ensure_ascii=False), now, now),
            )
            self._conn.commit()
            return True

    def delete(self, content: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM jargon WHERE content = ?", (content,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def toggle_global(self, content: str) -> bool | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, is_global FROM jargon WHERE content = ?", (content,)
            ).fetchone()
            if not row:
                return None
            new_val = 0 if row["is_global"] else 1
            self._conn.execute(
                "UPDATE jargon SET is_global = ?, updated_at = ? WHERE id = ?",
                (new_val, time.time(), row["id"]),
            )
            self._conn.commit()
            return bool(new_val)

    def query_terms(self, search: str = "", kind: str = "all",
                    offset: int = 0, limit: int = 50) -> tuple[int, list[dict]]:
        """WebUI 用：搜索 + 过滤 + 分页。返回 (总数, 词条列表)。

        kind: all / jargon(已确认) / candidate(候选) / manual(手动)
        """
        where, params = [], []
        if kind == "jargon":
            where.append("is_jargon = 1")
        elif kind == "candidate":
            where.append("is_jargon = 0")
        elif kind == "manual":
            where.append("created_by = 'manual'")
        if search:
            where.append("(content LIKE ? OR meaning LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like])
        sql_where = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) c FROM jargon{sql_where}", params
            ).fetchone()["c"]
            rows = self._conn.execute(
                f"SELECT * FROM jargon{sql_where} "
                "ORDER BY count DESC, id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return total, [dict(r) for r in rows]

    def list_terms(self, offset: int = 0, limit: int = 15,
                   only_jargon: bool = True) -> list[dict]:
        sql = "SELECT * FROM jargon"
        if only_jargon:
            sql += " WHERE is_jargon = 1"
        sql += " ORDER BY count DESC, id DESC LIMIT ? OFFSET ?"
        with self._lock:
            rows = self._conn.execute(sql, (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) c FROM jargon").fetchone()["c"]
            learned = self._conn.execute(
                "SELECT COUNT(*) c FROM jargon WHERE is_jargon = 1"
            ).fetchone()["c"]
            manual = self._conn.execute(
                "SELECT COUNT(*) c FROM jargon WHERE created_by = 'manual'"
            ).fetchone()["c"]
            complete = self._conn.execute(
                "SELECT COUNT(*) c FROM jargon WHERE is_complete = 1"
            ).fetchone()["c"]
        return {"total": total, "learned": learned,
                "manual": manual, "complete": complete}
