import re
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Optional

import aiohttp
try:
    from fonttools.ttLib import TTCollection, TTFont
except ImportError:
    import importlib
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fonttools>=4.0.0"])
    importlib.invalidate_caches()
    from fonttools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFont

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import ComponentType, File, Image as AstrImage, Reply
from astrbot.api.star import Context, Star

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "schemas.db"

FONTS_DIR = Path(__file__).parent / "fonts"

# 字体加载顺序：ChaiPUA → SourceHanSansSC → Plangothic，按优先级尝试
_BUNDLED_FONTS = [
    FONTS_DIR / "ChaiPUA-0.2.7.ttf",
    FONTS_DIR / "SourceHanSansSC-Regular.otf",
    FONTS_DIR / "Plangothic.ttc",
]


def _build_cmap(font_path: Path) -> frozenset[int]:
    """读取字体文件的 cmap，返回其覆盖的 Unicode 码位集合。"""
    codepoints: set[int] = set()
    try:
        suffix = font_path.suffix.lower()
        if suffix == ".ttc":
            col = TTCollection(str(font_path))
            for tt in col.fonts:
                cmap = tt.getBestCmap()
                if cmap:
                    codepoints.update(cmap.keys())
        else:
            tt = TTFont(str(font_path))
            cmap = tt.getBestCmap()
            if cmap:
                codepoints.update(cmap.keys())
    except Exception as e:
        logger.warning(f"[im_schemas] 读取字体 cmap 失败 {font_path.name}: {e}")
    return frozenset(codepoints)


# 模块级预加载：(cmap码位集, 字体路径) 列表，启动时构建一次
_FONT_CMAPS: list[tuple[frozenset[int], Path]] = [
    (_build_cmap(p), p)
    for p in _BUNDLED_FONTS
    if p.exists()
]


def _load_fonts(size: int) -> list[ImageFont.FreeTypeFont]:
    """按优先级加载所有可用字体，返回 Pillow 字体列表。"""
    fonts = []
    for _, p in _FONT_CMAPS:
        try:
            fonts.append(ImageFont.truetype(str(p), size))
        except Exception:
            pass
    if not fonts:
        fonts.append(ImageFont.load_default())
    return fonts


def _pick_font(ch: str, fonts: list) -> ImageFont.FreeTypeFont:
    """用 cmap 精确判断哪个字体有该字符的字形，找不到则用第一个字体。"""
    cp = ord(ch)
    for (cmap, _), f in zip(_FONT_CMAPS, fonts):
        if cp in cmap:
            return f
    return fonts[0]


def _render_text_with_fallback(draw, pos, text: str, fonts: list, fill):
    """逐字符渲染，通过 cmap 为每个字符选用正确的字体。"""
    x, y = pos
    for ch in text:
        f = _pick_font(ch, fonts)
        draw.text((x, y), ch, font=f, fill=fill)
        bb = draw.textbbox((0, 0), ch, font=f)
        x += bb[2] - bb[0]


DEFAULT_SELECT_KEYS = "_;'4567890"
DEFAULT_MAX_LEN = 4


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

        fonts_hdr  = _load_fonts(18)
        fonts_word = _load_fonts(34)
        fonts_th   = _load_fonts(15)
        fonts_row  = _load_fonts(16)

        def wh(text: str, fonts) -> tuple[int, int]:
            total_w = 0
            max_h = 0
            for ch in text:
                f = fonts[0]
                for ff in fonts:
                    try:
                        bb = draw.textbbox((0, 0), ch, font=ff)
                        if bb[2] > bb[0]:
                            f = ff
                            break
                    except Exception:
                        pass
                bb = draw.textbbox((0, 0), ch, font=f)
                total_w += bb[2] - bb[0]
                max_h = max(max_h, bb[3] - bb[1])
            return total_w, max_h

        def center(text: str, fonts, x: int, y: int, w: int, h: int, color):
            tw, th = wh(text, fonts)
            _render_text_with_fallback(draw, (x + (w - tw) / 2, y + (h - th) / 2), text, fonts, color)

        # 标题栏
        draw.rectangle([(0, 0), (IMG_W, HEADER_H)], fill=C_HDR_BG)
        center(f"词提：{schema_name}", fonts_hdr, 0, 0, IMG_W, HEADER_H, C_HDR_FG)

        # 字词行
        label = f"字词：{word}"
        _, th = wh(label, fonts_word)
        _render_text_with_fallback(draw, (PAD, HEADER_H + (WORD_H - th) / 2), label, fonts_word, C_WORD_FG)

        ty = HEADER_H + WORD_H
        tx = PAD

        # 表头
        draw.rectangle([(tx, ty), (tx + TABLE_W, ty + ROW_H)], fill=C_TH_BG)
        center("#",   fonts_th, tx,          ty, IDX_W,  ROW_H, C_TH_FG)
        center("编码", fonts_th, tx + IDX_W, ty, CODE_W, ROW_H, C_TH_FG)
        draw.line([(tx, ty + ROW_H), (tx + TABLE_W, ty + ROW_H)], fill=C_BORDER, width=1)
        ty += ROW_H

        if not codes:
            draw.rectangle([(tx, ty), (tx + TABLE_W, ty + ROW_H)], fill=C_ODD)
            _render_text_with_fallback(draw, (tx + 12, ty + (ROW_H - 16) / 2), "无结果", fonts_row, C_EMPTY)
        else:
            for i, code in enumerate(codes):
                bg = C_ODD if i % 2 == 0 else C_EVEN
                draw.rectangle([(tx, ty), (tx + TABLE_W, ty + ROW_H)], fill=bg)
                center(str(i + 1), fonts_row, tx,          ty, IDX_W,  ROW_H, C_ROW_FG)
                _, ch_h = wh(code, fonts_row)
                _render_text_with_fallback(
                    draw,
                    (tx + IDX_W + 12, ty + (ROW_H - ch_h) / 2),
                    code, fonts_row, C_ROW_FG,
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
