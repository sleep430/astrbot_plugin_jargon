"""群黑话自动学习系统。

整体思路移植自 MaiBot (https://github.com/Mai-with-u/MaiBot)：
- 学：攒消息 -> LLM 提取候选 -> 入库计次 -> 阶梯阈值触发双路对照推断
- 用：on_llm_request 时纯机械匹配命中词条 -> 注入 system_prompt

指令（需管理员）：
    /jargon list [页码]   查看已学会的黑话
    /jargon add 词 含义    手动录入（不会被 AI 覆盖）
    /jargon del 词        删除词条
    /jargon global 词     切换词条全局作用域
    /jargon stat          统计信息
    /jargon block 群号     把群拉进黑名单（不再学习该群）
    /jargon unblock 群号   把群移出黑名单
    /jargon blocklist      查看黑名单
"""

import asyncio
import json
import re
from collections import defaultdict, deque

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

try:
    from astrbot.api.web import request as web_request
except Exception:  # 旧版本 AstrBot 无 web 模块
    web_request = None

from .matcher import format_injection, match_jargons
from .miner import JargonMiner
from .store import JargonStore

BUFFER_MAX = 200  # 每个会话最多缓存的消息条数


@register(
    "astrbot_plugin_jargon",
    "月咏小眠&睡觉",
    "群黑话自动学习系统：提取-对照推断-机械匹配注入（思路移植自 MaiBot）",
    "v1.0.0",
    "https://github.com/Mai-with-u/MaiBot",
)
class JargonPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or {}

        self.enabled = bool(self.config.get("enabled", True))
        self.learn_frequency = max(5, int(self.config.get("learn_frequency", 40)))
        self.inject_max = max(1, int(self.config.get("inject_max", 10)))
        self.listen_private = bool(self.config.get("listen_private", False))
        self.blacklist: set = self._parse_blacklist(
            self.config.get("blacklist_groups", "")
        )

        data_dir = StarTools.get_data_dir("astrbot_plugin_jargon")
        data_dir.mkdir(parents=True, exist_ok=True)
        self.store = JargonStore(str(data_dir / "jargon.db"))
        self.miner = JargonMiner(self.context, self.store, self.config)

        # 每个会话的消息缓冲：umo -> deque[{"source_id","sender","text"}]
        self._buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=BUFFER_MAX)
        )
        self._counters: dict[str, int] = defaultdict(int)  # source_id 计数器
        self._since_learn: dict[str, int] = defaultdict(int)  # 距上次提取的条数

        self._web_api_registered = False
        self._register_web_api_routes()

    async def initialize(self):
        logger.info("[jargon] 黑话学习系统已加载")

    async def terminate(self):
        self._buffers.clear()

    # ---------- 黑名单 ----------

    @staticmethod
    def _parse_blacklist(raw) -> set:
        """把配置里的黑名单解析成群号集合。兼容字符串/列表。"""
        if not raw:
            return set()
        if isinstance(raw, (list, tuple, set)):
            items = raw
        else:
            items = re.split(r"[,，;\s]+", str(raw))
        return {str(x).strip() for x in items if str(x).strip()}

    def _session_id_of(self, event: AstrMessageEvent) -> str:
        """取当前会话的群号（群聊优先取 group_id，退回 umo 末段）。"""
        try:
            gid = getattr(event.message_obj, "group_id", None)
            if gid:
                return str(gid)
        except Exception:
            pass
        umo = event.unified_msg_origin or ""
        return umo.split(":")[-1] if umo else ""

    def _is_blocked(self, event: AstrMessageEvent) -> bool:
        """判断当前会话是否在黑名单里。"""
        if not self.blacklist:
            return False
        umo = event.unified_msg_origin or ""
        sid = self._session_id_of(event)
        parts = {p for p in re.split(r"[:/]", umo) if p}
        for b in self.blacklist:
            if b and (b == sid or b == umo or b in parts):
                return True
        return False

    def _save_blacklist(self):
        self.config["blacklist_groups"] = ",".join(sorted(self.blacklist))
        self._save_config()

    # ---------- 学习侧：被动收集 ----------

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=100)
    async def collect_group(self, event: AstrMessageEvent):
        await self._collect(event)

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=100)
    async def collect_private(self, event: AstrMessageEvent):
        if self.listen_private:
            await self._collect(event)

    async def _collect(self, event: AstrMessageEvent):
        if not self.enabled:
            return
        if self._is_blocked(event):
            return  # 黑名单群不学习
        if event.get_sender_id() == event.get_self_id():
            return  # 不学自己说的话
        text = (event.get_message_str() or "").strip()
        if not text or text.startswith("/"):
            return
        if len(text) > 500:
            text = text[:500]

        umo = event.unified_msg_origin
        self._counters[umo] += 1
        try:
            sender = event.get_sender_name() or event.get_sender_id()
        except Exception:
            sender = event.get_sender_id()

        self._buffers[umo].append(
            {"source_id": self._counters[umo], "sender": sender, "text": text}
        )
        self._since_learn[umo] += 1

        if self._since_learn[umo] >= self.learn_frequency:
            self._since_learn[umo] = 0
            snapshot = list(self._buffers[umo])
            self._buffers[umo].clear()
            asyncio.create_task(self.miner.learn_from_chat(umo, snapshot))

    # ---------- 使用侧：机械匹配注入 ----------

    @filter.on_llm_request()
    async def inject_jargon(self, event: AstrMessageEvent, req):
        if not self.enabled:
            return
        if self._is_blocked(event):
            return  # 黑名单群不注入
        umo = event.unified_msg_origin
        records = self.store.get_matchable(umo)
        if not records:
            return

        # 待匹配文本：当前 prompt + 最近若干条用户上下文
        texts: list[str] = []
        for ctx in (req.contexts or [])[-20:]:
            if ctx.get("role") != "user":
                continue
            content = ctx.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                if parts:
                    texts.append(" ".join(parts))
        if req.prompt:
            texts.append(req.prompt)

        matches = match_jargons(records, texts, limit=self.inject_max)
        if not matches:
            return

        block = format_injection(matches)
        req.system_prompt = (req.system_prompt or "") + "\n\n" + block
        logger.debug(
            f"[jargon] 注入黑话参考 {len(matches)} 条: "
            + ", ".join(m["content"] for m in matches)
        )

    # ---------- 管理指令 ----------

    @filter.command_group("jargon")
    def jargon(self):
        pass

    @staticmethod
    def _tokens(event: AstrMessageEvent) -> list[str]:
        """解析 /jargon xxx ... 指令参数。"""
        text = (event.get_message_str() or "").strip().lstrip("/").strip()
        parts = text.split()
        # parts[0] == "jargon"
        return parts[1:] if len(parts) > 1 else []

    @jargon.command("list")
    async def jargon_list(self, event: AstrMessageEvent):
        tokens = self._tokens(event)
        page = 1
        if len(tokens) >= 2 and tokens[1].isdigit():
            page = max(1, int(tokens[1]))
        size = 15
        terms = self.store.list_terms(offset=(page - 1) * size, limit=size)
        if not terms:
            yield event.plain_result("这一页没有黑话记录。")
            return
        lines = [f"黑话列表（第 {page} 页）："]
        for t in terms:
            scope = "全局" if t["is_global"] else "本群"
            mark = "（手动）" if t["created_by"] == "manual" else ""
            lines.append(
                f"· {t['content']}：{t['meaning']} "
                f"[x{t['count']} {scope}{mark}]"
            )
        yield event.plain_result("\n".join(lines))

    @jargon.command("add")
    async def jargon_add(self, event: AstrMessageEvent):
        if not event.is_admin():
            yield event.plain_result("只有管理员可以手动录入黑话。")
            return
        tokens = self._tokens(event)
        if len(tokens) < 3:
            yield event.plain_result("用法：/jargon add 词 含义")
            return
        word = tokens[1]
        meaning = " ".join(tokens[2:])
        created = self.store.manual_add(word, meaning, event.unified_msg_origin)
        action = "已录入" if created else "已更新"
        yield event.plain_result(f"{action}黑话「{word}」：{meaning}")

    @jargon.command("del")
    async def jargon_del(self, event: AstrMessageEvent):
        if not event.is_admin():
            yield event.plain_result("只有管理员可以删除黑话。")
            return
        tokens = self._tokens(event)
        if len(tokens) < 2:
            yield event.plain_result("用法：/jargon del 词")
            return
        ok = self.store.delete(tokens[1])
        yield event.plain_result(
            f"已删除「{tokens[1]}」。" if ok else f"没有找到「{tokens[1]}」。"
        )

    @jargon.command("global")
    async def jargon_global(self, event: AstrMessageEvent):
        if not event.is_admin():
            yield event.plain_result("只有管理员可以调整作用域。")
            return
        tokens = self._tokens(event)
        if len(tokens) < 2:
            yield event.plain_result("用法：/jargon global 词")
            return
        result = self.store.toggle_global(tokens[1])
        if result is None:
            yield event.plain_result(f"没有找到「{tokens[1]}」。")
        else:
            state = "全局" if result else "本群"
            yield event.plain_result(f"「{tokens[1]}」已切换为{state}词条。")

    @jargon.command("stat")
    async def jargon_stat(self, event: AstrMessageEvent):
        s = self.store.stats()
        yield event.plain_result(
            f"黑话统计：候选 {s['total']} 条，已确认黑话 {s['learned']} 条，"
            f"手动录入 {s['manual']} 条，完成全部推断 {s['complete']} 条。"
        )

    @jargon.command("block")
    async def jargon_block(self, event: AstrMessageEvent):
        if not event.is_admin():
            yield event.plain_result("只有管理员可以设置黑话黑名单。")
            return
        tokens = self._tokens(event)
        if len(tokens) < 2:
            yield event.plain_result("用法：/jargon block 群号")
            return
        gid = tokens[1].strip()
        if gid in self.blacklist:
            yield event.plain_result(f"群 {gid} 本来就在黑名单里。")
            return
        self.blacklist.add(gid)
        self._save_blacklist()
        yield event.plain_result(f"已把群 {gid} 拉进黑话黑名单，不再学它的消息。")

    @jargon.command("unblock")
    async def jargon_unblock(self, event: AstrMessageEvent):
        if not event.is_admin():
            yield event.plain_result("只有管理员可以设置黑话黑名单。")
            return
        tokens = self._tokens(event)
        if len(tokens) < 2:
            yield event.plain_result("用法：/jargon unblock 群号")
            return
        gid = tokens[1].strip()
        if gid not in self.blacklist:
            yield event.plain_result(f"群 {gid} 不在黑名单里。")
            return
        self.blacklist.discard(gid)
        self._save_blacklist()
        yield event.plain_result(f"已把群 {gid} 移出黑话黑名单。")

    @jargon.command("blocklist")
    async def jargon_blocklist(self, event: AstrMessageEvent):
        if not self.blacklist:
            yield event.plain_result("黑话黑名单是空的。")
            return
        yield event.plain_result(
            "黑话黑名单（这些群不学习）：" + ", ".join(sorted(self.blacklist))
        )

    # ---------- WebUI API ----------

    def _register_web_api_routes(self):
        if self._web_api_registered:
            return
        if not hasattr(self.context, "register_web_api") or web_request is None:
            logger.warning("[jargon] 当前 AstrBot 版本不支持插件 WebUI，跳过注册")
            return
        prefix = "/astrbot_plugin_jargon/jargon"
        routes = [
            (f"{prefix}/list", self.api_jargon_list, ["GET", "POST"], "黑话列表查询"),
            (f"{prefix}/add", self.api_jargon_add, ["POST"], "手动录入/更新词条"),
            (f"{prefix}/delete", self.api_jargon_delete, ["POST"], "删除词条"),
            (f"{prefix}/toggle_global", self.api_jargon_toggle_global, ["POST"], "切换全局作用域"),
            (f"{prefix}/switch", self.api_jargon_switch, ["POST"], "插件总开关"),
            (f"{prefix}/config", self.api_jargon_config, ["GET", "POST"], "读写插件配置"),
        ]
        for route, handler, methods, desc in routes:
            try:
                self.context.register_web_api(route, handler, methods, desc)
            except Exception as e:
                logger.error(f"[jargon] 注册 WebUI 路由失败 {route}: {e}")
                return
        self._web_api_registered = True
        logger.info("[jargon] WebUI 路由已注册")

    @staticmethod
    async def _read_web_payload() -> dict:
        payload = {}
        try:
            payload = await web_request.json(default={}) or {}
        except TypeError:
            try:
                payload = await web_request.json() or {}
            except Exception:
                payload = {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        # GET 请求走 query 参数
        try:
            for key in ("search", "kind", "page", "size", "content"):
                val = web_request.query.get(key)
                if val is not None and key not in payload:
                    payload[key] = val
        except Exception:
            pass
        return payload

    @staticmethod
    def _serialize_term(rec: dict) -> dict:
        try:
            scopes = json.loads(rec.get("chat_scopes") or "{}")
        except (json.JSONDecodeError, TypeError):
            scopes = {}
        try:
            evidence = json.loads(rec.get("evidence") or "[]")
        except (json.JSONDecodeError, TypeError):
            evidence = []
        return {
            "id": rec.get("id"),
            "content": rec.get("content", ""),
            "meaning": rec.get("meaning", ""),
            "count": rec.get("count", 0),
            "is_jargon": bool(rec.get("is_jargon")),
            "is_complete": bool(rec.get("is_complete")),
            "is_global": bool(rec.get("is_global")),
            "created_by": rec.get("created_by", "ai"),
            "scopes": scopes,
            "evidence": evidence,
            "updated_at": rec.get("updated_at", 0),
        }

    async def api_jargon_list(self):
        payload = await self._read_web_payload()
        search = str(payload.get("search") or "").strip()[:100]
        kind = str(payload.get("kind") or "all").strip()
        if kind not in ("all", "jargon", "candidate", "manual"):
            kind = "all"
        try:
            page = max(1, int(payload.get("page") or 1))
        except (TypeError, ValueError):
            page = 1
        try:
            size = min(200, max(1, int(payload.get("size") or 50)))
        except (TypeError, ValueError):
            size = 50
        total, rows = self.store.query_terms(
            search=search, kind=kind, offset=(page - 1) * size, limit=size
        )
        return {
            "status": "ok",
            "message": "",
            "data": {
                "enabled": self.enabled,
                "stats": self.store.stats(),
                "total": total,
                "page": page,
                "size": size,
                "items": [self._serialize_term(r) for r in rows],
            },
        }

    async def api_jargon_add(self):
        payload = await self._read_web_payload()
        content = str(payload.get("content") or "").strip()[:64]
        meaning = str(payload.get("meaning") or "").strip()[:500]
        if not content or not meaning:
            return {"status": "error", "message": "词条和含义都不能为空", "data": {}}
        created = self.store.manual_add(content, meaning, "webui")
        return {
            "status": "ok",
            "message": f"「{content}」已{'录入' if created else '更新'}（手动词条不会被 AI 覆盖）",
            "data": {"created": created},
        }

    async def api_jargon_delete(self):
        payload = await self._read_web_payload()
        content = str(payload.get("content") or "").strip()[:64]
        if not content:
            return {"status": "error", "message": "缺少词条", "data": {}}
        ok = self.store.delete(content)
        return {
            "status": "ok" if ok else "error",
            "message": f"已删除「{content}」" if ok else f"没有找到「{content}」",
            "data": {},
        }

    async def api_jargon_toggle_global(self):
        payload = await self._read_web_payload()
        content = str(payload.get("content") or "").strip()[:64]
        if not content:
            return {"status": "error", "message": "缺少词条", "data": {}}
        result = self.store.toggle_global(content)
        if result is None:
            return {"status": "error", "message": f"没有找到「{content}」", "data": {}}
        return {
            "status": "ok",
            "message": f"「{content}」已切换为{'全局' if result else '会话'}词条",
            "data": {"is_global": result},
        }

    async def api_jargon_switch(self):
        payload = await self._read_web_payload()
        enabled = payload.get("enabled")
        self.enabled = bool(enabled)
        try:
            self.config["enabled"] = self.enabled
            if hasattr(self.config, "save_config"):
                self.config.save_config()
        except Exception as e:
            logger.warning(f"[jargon] 开关写入配置失败: {e}")
        return {
            "status": "ok",
            "message": f"黑话学习已{'开启' if self.enabled else '关闭'}",
            "data": {"enabled": self.enabled},
        }

    def _available_providers(self) -> list[dict]:
        providers = []
        try:
            for p in self.context.get_all_providers():
                try:
                    meta = p.meta()
                    providers.append({
                        "id": meta.id,
                        "model": meta.model or "",
                        "type": meta.type or "",
                    })
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"[jargon] 获取模型列表失败: {e}")
        return providers

    def _save_config(self):
        try:
            if hasattr(self.config, "save_config"):
                self.config.save_config()
        except Exception as e:
            logger.warning(f"[jargon] 配置持久化失败: {e}")

    async def api_jargon_config(self):
        payload = await self._read_web_payload()
        if payload:  # POST 写配置
            changed = []
            if "provider_id" in payload:
                pid = str(payload.get("provider_id") or "").strip()
                if pid and not any(
                    p["id"] == pid for p in self._available_providers()
                ):
                    return {"status": "error",
                            "message": f"模型 {pid} 不存在", "data": {}}
                self.config["provider_id"] = pid
                changed.append("provider_id")
            for key, caster, lo, hi in (
                ("learn_frequency", int, 5, 10000),
                ("extract_max", int, 1, 100),
                ("inject_max", int, 1, 50),
            ):
                if key in payload:
                    try:
                        val = max(lo, min(hi, caster(payload[key])))
                    except (TypeError, ValueError):
                        continue
                    self.config[key] = val
                    changed.append(key)
            if "blacklist_groups" in payload:
                self.blacklist = self._parse_blacklist(payload.get("blacklist_groups"))
                self.config["blacklist_groups"] = ",".join(sorted(self.blacklist))
                changed.append("blacklist_groups")
            # 同步运行期参数
            self.learn_frequency = max(5, int(self.config.get("learn_frequency", 40)))
            self.inject_max = max(1, int(self.config.get("inject_max", 10)))
            self.miner.extract_max = int(self.config.get("extract_max", 30))
            if changed:
                self._save_config()
        return {
            "status": "ok",
            "message": "配置已保存" if payload else "",
            "data": {
                "provider_id": str(self.config.get("provider_id") or ""),
                "learn_frequency": int(self.config.get("learn_frequency", 40)),
                "extract_max": int(self.config.get("extract_max", 30)),
                "inject_max": int(self.config.get("inject_max", 10)),
                "blacklist_groups": ",".join(sorted(self.blacklist)),
                "providers": self._available_providers(),
            },
        }
