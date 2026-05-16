import re
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Optional

import aiohttp
from PIL import Image, ImageDraw, ImageFont

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import ComponentType, File, Image as AstrImage, Reply
from astrbot.api.star import Context, Star

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "schemas.db"

_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
]

DEFAULT_SELECT_KEYS = "_;'4567890"
DEFAULT_MAX_LEN = 4


def _find_font() -> Optional[str]:
    import os
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schemas (
            name        TEXT PRIMARY KEY,
            owner_id    TEXT NOT NULL,
            select_keys TEXT NOT NULL DEFAULT '',
            max_len     INTEGER NOT NULL DEFAULT 4,
            punct_key   TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS codes (
            schema_name TEXT NOT NULL,
            code        TEXT NOT NULL,
            word        TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_codes ON codes(schema_name, word)"
    )
    return conn


class IMSchemasPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.font_path = _find_font()
        if not self.font_path:
            logger.warning(
                "[im_schemas] 未找到 CJK 字体，图片中文字可能显示为方块。"
                "请安装 wqy-microhei 或 noto-cjk 字体。"
            )

    # ── 数据库操作 ──────────────────────────────────────────────────────────

    def _schema_exists(self, name: str) -> bool:
        conn = _open_db()
        row = conn.execute(
            "SELECT 1 FROM schemas WHERE name = ?", (name,)
        ).fetchone()
        conn.close()
        return row is not None

    def _schema_owner(self, name: str) -> Optional[str]:
        conn = _open_db()
        row = conn.execute(
            "SELECT owner_id FROM schemas WHERE name = ?", (name,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _import_schema(
        self,
        name: str,
        owner_id: str,
        entries: list[tuple[str, str]],
        select_keys: str,
        max_len: int,
        punct_key: str,
    ) -> int:
        conn = _open_db()
        conn.execute(
            """
            INSERT INTO schemas(name, owner_id, select_keys, max_len, punct_key)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                owner_id    = excluded.owner_id,
                select_keys = excluded.select_keys,
                max_len     = excluded.max_len,
                punct_key   = excluded.punct_key
            """,
            (name, owner_id, select_keys, max_len, punct_key),
        )
        conn.execute("DELETE FROM codes WHERE schema_name = ?", (name,))
        conn.executemany(
            "INSERT INTO codes(schema_name, code, word) VALUES (?, ?, ?)",
            [(name, code, word) for code, word in entries],
        )
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM codes WHERE schema_name = ?", (name,)
        ).fetchone()[0]
        conn.close()
        return count

    def _delete_schema(self, name: str) -> None:
        conn = _open_db()
        conn.execute("DELETE FROM codes WHERE schema_name = ?", (name,))
        conn.execute("DELETE FROM schemas WHERE name = ?", (name,))
        conn.commit()
        conn.close()

    def _query_codes(self, schema_name: str, word: str) -> list[str]:
        conn = _open_db()
        rows = conn.execute(
            """
            SELECT code FROM codes
            WHERE schema_name = ? AND word = ?
            ORDER BY length(code), code
            """,
            (schema_name, word),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def _schema_info(self, name: str) -> Optional[dict]:
        conn = _open_db()
        row = conn.execute(
            "SELECT select_keys, max_len, punct_key FROM schemas WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            conn.close()
            return None
        select_keys, max_len, punct_key = row
        # 码元：该码表所有编码中出现的不重复字符
        chars_rows = conn.execute(
            "SELECT DISTINCT code FROM codes WHERE schema_name = ?", (name,)
        ).fetchall()
        conn.close()
        chars: set[str] = set()
        for (code,) in chars_rows:
            chars.update(code)
        return {
            "select_keys": select_keys or DEFAULT_SELECT_KEYS,
            "max_len": max_len,
            "punct_key": punct_key,
            "chars": "".join(sorted(chars)),
        }

    # ── 解析 TSV 码表 ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_tsv(content: str) -> list[tuple[str, str]]:
        """解析 TSV 码表，第一列编码，第二列字词，返回 [(code, word), ...]。"""
        entries: list[tuple[str, str]] = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            code, word = parts[0].strip(), parts[1].strip()
            if code and word:
                entries.append((code, word))
        return entries

    # ── 文件下载 ────────────────────────────────────────────────────────────

    async def _download_text(self, url: str) -> Optional[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    raw = await resp.read()
            for enc in ("utf-8", "gbk", "utf-16"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            logger.warning(f"[im_schemas] 下载文件失败: {e}")
        return None

    # ── 图片生成 ────────────────────────────────────────────────────────────

    def _make_image(self, schema_name: str, word: str, codes: list[str]) -> bytes:
        PAD = 24
        HEADER_H = 52
        WORD_H = 72
        ROW_H = 38
        IDX_W = 52
        CODE_W = 220
        TABLE_W = IDX_W + CODE_W
        IMG_W = TABLE_W + PAD * 2
        n_rows = max(len(codes), 1)
        IMG_H = HEADER_H + WORD_H + ROW_H * (n_rows + 1) + PAD

        C_BG     = (245, 247, 250)
        C_HDR_BG = (66, 133, 244)
        C_HDR_FG = (255, 255, 255)
        C_WORD_FG = (30, 30, 30)
        C_TH_BG  = (210, 227, 252)
        C_TH_FG  = (50, 70, 110)
        C_ODD    = (255, 255, 255)
        C_EVEN   = (237, 244, 255)
        C_ROW_FG = (40, 40, 40)
        C_BORDER = (190, 200, 215)
        C_EMPTY  = (160, 80, 80)

        img = Image.new("RGB", (IMG_W, IMG_H), C_BG)
        draw = ImageDraw.Draw(img)

        def font(size: int):
            if self.font_path:
                try:
                    return ImageFont.truetype(self.font_path, size)
                except Exception:
                    pass
            return ImageFont.load_default()

        f_hdr  = font(18)
        f_word = font(34)
        f_th   = font(15)
        f_row  = font(16)

        def wh(text: str, f) -> tuple[int, int]:
            bb = draw.textbbox((0, 0), text, font=f)
            return bb[2] - bb[0], bb[3] - bb[1]

        def center(text: str, f, x: int, y: int, w: int, h: int, color):
            tw, th = wh(text, f)
            draw.text((x + (w - tw) / 2, y + (h - th) / 2), text, font=f, fill=color)

        # 标题栏
        draw.rectangle([(0, 0), (IMG_W, HEADER_H)], fill=C_HDR_BG)
        center(f"词提：{schema_name}", f_hdr, 0, 0, IMG_W, HEADER_H, C_HDR_FG)

        # 字词行
        label = f"字词：{word}"
        _, th = wh(label, f_word)
        draw.text((PAD, HEADER_H + (WORD_H - th) / 2), label, font=f_word, fill=C_WORD_FG)

        ty = HEADER_H + WORD_H
        tx = PAD

        # 表头
        draw.rectangle([(tx, ty), (tx + TABLE_W, ty + ROW_H)], fill=C_TH_BG)
        center("#",   f_th, tx,          ty, IDX_W,  ROW_H, C_TH_FG)
        center("编码", f_th, tx + IDX_W, ty, CODE_W, ROW_H, C_TH_FG)
        draw.line([(tx, ty + ROW_H), (tx + TABLE_W, ty + ROW_H)], fill=C_BORDER, width=1)
        ty += ROW_H

        if not codes:
            draw.rectangle([(tx, ty), (tx + TABLE_W, ty + ROW_H)], fill=C_ODD)
            draw.text((tx + 12, ty + (ROW_H - 16) / 2), "无结果", font=f_row, fill=C_EMPTY)
        else:
            for i, code in enumerate(codes):
                bg = C_ODD if i % 2 == 0 else C_EVEN
                draw.rectangle([(tx, ty), (tx + TABLE_W, ty + ROW_H)], fill=bg)
                center(str(i + 1), f_row, tx,          ty, IDX_W,  ROW_H, C_ROW_FG)
                draw.text(
                    (tx + IDX_W + 12, ty + (ROW_H - 16) / 2),
                    code, font=f_row, fill=C_ROW_FG,
                )
                draw.line(
                    [(tx, ty + ROW_H), (tx + TABLE_W, ty + ROW_H)],
                    fill=C_BORDER, width=1,
                )
                ty += ROW_H

        table_bottom = HEADER_H + WORD_H + ROW_H * (n_rows + 1)
        draw.rectangle(
            [(tx, HEADER_H + WORD_H), (tx + TABLE_W, table_bottom)],
            outline=C_BORDER, width=1,
        )

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # ── 私聊：收到文件时给出上传指引 ──────────────────────────────────────

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def handle_private_file(self, event: AstrMessageEvent):
        """私聊收到文件时，提示用户引用回复该文件并发送上传指令。"""
        chain = event.get_messages()
        file_seg: Optional[File] = next(
            (seg for seg in chain if seg.type == ComponentType.File), None
        )
        if not file_seg:
            return
        if not (file_seg.name or "").endswith(".txt"):
            await event.send(event.plain_result("只支持 .txt 格式的 TSV 码表文件。"))
            event.stop_event()
            return
        await event.send(
            event.plain_result(
                "文件已收到。请引用回复上方文件消息，并发送：\n"
                "上传词提 [词提名]\n\n"
                "可附加参数（空格分隔，均可省略）：\n"
                "  选重键=_;'4567890\n"
                "  最大长度=4\n"
                "  标点引导键=\n\n"
                "示例：上传词提 五笔86 选重键=_;' 最大长度=4"
            )
        )
        event.stop_event()

    # ── 上传词提 ────────────────────────────────────────────────────────────

    @filter.command("上传词提")
    async def cmd_upload(self, event: AstrMessageEvent):
        """
        用法：引用回复码表文件消息，发送
          上传词提 [词提名] [选重键=...] [最大长度=N] [标点引导键=...]
        """
        args_str = event.message_str.strip()
        if not args_str:
            yield event.plain_result("用法：上传词提 [词提名] [选重键=...] [最大长度=N] [标点引导键=...]")
            return

        # 解析词提名（第一个 token）和可选参数
        tokens = args_str.split()
        # event.message_str 包含指令前缀，跳过它
        if tokens and tokens[0] == "上传词提":
            tokens = tokens[1:]
        if not tokens:
            yield event.plain_result("用法：上传词提 [词提名] [选重键=...] [最大长度=N] [标点引导键=...]")
            return
        schema_name = tokens[0]
        select_keys = DEFAULT_SELECT_KEYS
        max_len = DEFAULT_MAX_LEN
        punct_key = ""

        for tok in tokens[1:]:
            if tok.startswith("选重键="):
                select_keys = tok[4:]
            elif tok.startswith("最大长度="):
                try:
                    max_len = int(tok[5:])
                except ValueError:
                    yield event.plain_result("最大长度必须为整数。")
                    return
            elif tok.startswith("标点引导键="):
                punct_key = tok[6:]

        # 从引用回复中找 File
        chain = event.get_messages()
        reply_seg: Optional[Reply] = next(
            (seg for seg in chain if seg.type == ComponentType.Reply), None
        )

        file_seg: Optional[File] = None
        if reply_seg and reply_seg.chain:
            file_seg = next(
                (seg for seg in reply_seg.chain if seg.type == ComponentType.File),
                None,
            )

        if file_seg is None:
            yield event.plain_result(
                "未找到码表文件。请引用回复您上传的 .txt 文件消息，再发送此指令。"
            )
            return

        url = file_seg.url
        if not url:
            yield event.plain_result("无法获取文件下载链接，请重新发送文件后重试。")
            return

        yield event.plain_result("正在下载并解析码表，请稍候…")

        content = await self._download_text(url)
        if content is None:
            yield event.plain_result("文件下载失败，请检查网络后重试。")
            return

        entries = self._parse_tsv(content)
        if not entries:
            yield event.plain_result(
                "未能解析出有效条目。请确认文件为 TSV 格式：第一列编码，第二列字词，Tab 分隔。"
            )
            return

        user_id = event.get_sender_id()
        count = self._import_schema(
            schema_name, user_id, entries, select_keys, max_len, punct_key
        )
        yield event.plain_result(
            f"词提「{schema_name}」导入成功！共 {count:,} 条编码。\n"
            f"查询：{schema_name} <字词>\n"
            f"信息：%{schema_name}"
        )

    # ── 查码：[词提名] <字词> ───────────────────────────────────────────────

    @filter.regex(r"^(\S+)\s+(\S+)$")
    async def cmd_query(self, event: AstrMessageEvent):
        text = event.message_str.strip()
        m = re.match(r"^(\S+)\s+(\S+)$", text)
        if not m:
            return
        schema_name, word = m.group(1), m.group(2)
        if not self._schema_exists(schema_name):
            return  # 不是词提名，忽略

        codes = self._query_codes(schema_name, word)
        img_bytes = self._make_image(schema_name, word, codes)
        yield event.chain_result([AstrImage.fromBytes(img_bytes)])

    # ── 词提信息：%[词提名] ─────────────────────────────────────────────────

    @filter.regex(r"^%(\S+)$")
    async def cmd_info(self, event: AstrMessageEvent):
        text = event.message_str.strip()
        m = re.match(r"^%(\S+)$", text)
        if not m:
            return
        schema_name = m.group(1)
        info = self._schema_info(schema_name)
        if not info:
            yield event.plain_result(f"词提「{schema_name}」不存在。")
            return
        yield event.plain_result(
            f"词提：{schema_name}\n"
            f"选重键：{info['select_keys']}\n"
            f"最大长度：{info['max_len']}\n"
            f"标点引导键：{info['punct_key'] or '（无）'}\n"
            f"码元：{info['chars']}"
        )

    # ── 删除词提 ────────────────────────────────────────────────────────────

    @filter.command("删除词提")
    async def cmd_delete(self, event: AstrMessageEvent):
        schema_name = event.message_str.strip()
        if schema_name.startswith("删除词提"):
            schema_name = schema_name[len("删除词提"):].strip()
        if not schema_name:
            yield event.plain_result("用法：删除词提 [词提名]")
            return
        owner = self._schema_owner(schema_name)
        if owner is None:
            yield event.plain_result(f"词提「{schema_name}」不存在。")
            return
        user_id = event.get_sender_id()
        if owner != user_id:
            yield event.plain_result(f"只有词提「{schema_name}」的上传者才能删除它。")
            return
        self._delete_schema(schema_name)
        yield event.plain_result(f"词提「{schema_name}」已删除。")
