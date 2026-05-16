import re
import sqlite3
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Optional

import aiohttp
try:
    from fontTools.ttLib import TTCollection, TTFont
except ImportError:
    import importlib
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fonttools>=4.0.0"])
    importlib.invalidate_caches()
    for key in list(sys.modules.keys()):
        if "fonttools" in key.lower() or "fontTools" in key:
            del sys.modules[key]
    from fontTools.ttLib import TTCollection, TTFont
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


def _ref_ascent(fonts: list) -> int:
    """多字体回退时取最大 ascent 作为统一基线参考。
    各字体 hhea 度量不同（ChaiPUA 0.86em / SourceHan 1.16em / Plangothic 0.88em），
    若不统一基线，PUA 字符会比正文字体明显偏上。"""
    return max((f.getmetrics()[0] for f in fonts), default=0)


def _render_text_with_fallback(draw, pos, text: str, fonts: list, fill):
    """逐字符渲染，所有候选字体共享同一条基线，避免 PUA 等回退字体高度不齐。"""
    x, y_top = pos
    baseline = y_top + _ref_ascent(fonts)
    for ch in text:
        f = _pick_font(ch, fonts)
        draw.text((x, baseline), ch, font=f, fill=fill, anchor="ls")
        bb = draw.textbbox((0, 0), ch, font=f, anchor="ls")
        x += bb[2] - bb[0]


DEFAULT_SELECT_KEYS = "_;'4567890"
DEFAULT_MAX_LEN = 4


def _is_passthrough(ch: str) -> bool:
    """数字、字母、标点、空白等无需查码的字符（直接上屏，不算缺字）。"""
    if ch.isspace() or ch.isdigit():
        return True
    if ch.isascii() and ch.isalpha():
        return True
    cat = unicodedata.category(ch)
    # P* = punctuation, S* = symbol
    return cat.startswith("P") or cat.startswith("S")


def _is_punct(ch: str) -> bool:
    """标点（用于判定是否触发前一个字词的首选键省略）。"""
    return unicodedata.category(ch).startswith("P")


def _char_difficulty(ch: str) -> int:
    """字符本身的难度分：按 Unicode 区段估算稀有度。"""
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF:   # CJK 基本区（常用字）
        return 0
    if 0x3400 <= cp <= 0x4DBF:   # CJK 扩展 A
        return 1
    if 0x20000 <= cp <= 0x2A6DF: # CJK 扩展 B
        return 2
    if cp >= 0x2A700:             # CJK 扩展 C 及以上
        return 3
    return 0


def _text_difficulty(chars: list[str]) -> tuple[str, int]:
    score = sum(_char_difficulty(ch) for ch in chars if not _is_passthrough(ch))
    if score == 0:
        label = "淼"
    elif score <= 2:
        label = "易"
    elif score <= 5:
        label = "中"
    else:
        label = "难"
    return label, score


def _select_symbol(pos: int, select_keys: str) -> str:
    idx = pos - 2
    if 0 <= idx < len(select_keys):
        return select_keys[idx]
    return "?"


def _code_display(code: str, max_len: int, pos: int = 1, select_keys: str = "") -> str:
    if pos > 1:
        return code + _select_symbol(pos, select_keys)
    return code + "_" if len(code) < max_len else code


def _key_presses(code: str, max_len: int, pos: int = 1) -> int:
    if pos > 1:
        return len(code) + 1
    return len(code) + (1 if len(code) < max_len else 0)


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

    def _query_codes_with_positions(self, schema_name: str, word: str) -> list[tuple[str, int]]:
        conn = _open_db()
        codes = conn.execute(
            "SELECT code FROM codes WHERE schema_name = ? AND word = ? ORDER BY length(code), code",
            (schema_name, word),
        ).fetchall()
        result = []
        for (code,) in codes:
            words_for_code = [r[0] for r in conn.execute(
                "SELECT word FROM codes WHERE schema_name = ? AND code = ? ORDER BY rowid",
                (schema_name, code),
            ).fetchall()]
            try:
                pos = words_for_code.index(word) + 1
            except ValueError:
                pos = 1
            result.append((code, pos))
        conn.close()
        return result

    def _query_char_codes(self, schema_name: str, chars: list[str]) -> list[Optional[tuple[str, int]]]:
        conn = _open_db()
        result: list[Optional[tuple[str, int]]] = []
        for char in chars:
            row = conn.execute(
                "SELECT code FROM codes WHERE schema_name = ? AND word = ? ORDER BY length(code), code LIMIT 1",
                (schema_name, char),
            ).fetchone()
            if row is None:
                result.append(None)
                continue
            code = row[0]
            words_for_code = [r[0] for r in conn.execute(
                "SELECT word FROM codes WHERE schema_name = ? AND code = ? ORDER BY rowid",
                (schema_name, code),
            ).fetchall()]
            try:
                pos = words_for_code.index(char) + 1
            except ValueError:
                pos = 1
            result.append((code, pos))
        conn.close()
        return result

    def _schema_info(self, name: str) -> Optional[dict]:
        conn = _open_db()
        row = conn.execute(
            "SELECT owner_id, select_keys, max_len, punct_key FROM schemas WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            conn.close()
            return None
        owner_id, select_keys, max_len, punct_key = row
        chars_rows = conn.execute(
            "SELECT DISTINCT code FROM codes WHERE schema_name = ?", (name,)
        ).fetchall()
        conn.close()
        chars: set[str] = set()
        for (code,) in chars_rows:
            chars.update(code)
        return {
            "owner_id": owner_id,
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

    @staticmethod
    def _measure(draw, text: str, fonts: list) -> tuple[int, int, int]:
        """Returns (width, glyph_height, bottom_offset).
        基于统一基线（max ascent）计算，与 _render_text_with_fallback 保持一致：
          baseline = y_top + ref_ascent
          bottom_offset = ref_ascent + max(per-char descent below baseline)
          glyph_height  = bottom_offset - min(per-char top above baseline)
        这样多字体回退（如 ChaiPUA 与正文字体）的尺寸不会因度量差异而错位。
        """
        if not text:
            return 0, 0, 0
        ref_asc = _ref_ascent(fonts)
        w = 0
        max_below = 0     # 基线下最远像素
        min_above = 0     # 基线上最近像素（相对基线，负值；0 表示恰好在基线）
        for ch in text:
            f = _pick_font(ch, fonts)
            # anchor="ls": x=左边，y=基线
            bb = draw.textbbox((0, 0), ch, font=f, anchor="ls")
            w += bb[2] - bb[0]
            max_below = max(max_below, bb[3])   # 基线下方
            min_above = min(min_above, bb[1])   # 基线上方（≤0）
        bot = ref_asc + max_below
        gh = bot - (ref_asc + min_above)
        return w, gh, bot

    def _make_image(self, schema_name: str, word: str, owner_id: str, max_len: int, select_keys: str) -> bytes:
        if len(word) == 1:
            codes_with_pos = self._query_codes_with_positions(schema_name, word)
            return self._make_single_char_image(schema_name, word, codes_with_pos, owner_id, max_len, select_keys)
        chars = list(word)
        char_codes = self._query_char_codes(schema_name, chars)
        return self._make_multi_char_image(schema_name, word, char_codes, owner_id, max_len, select_keys)

    def _make_single_char_image(
        self,
        schema_name: str,
        word: str,
        codes_with_pos: list[tuple[str, int]],
        owner_id: str,
        max_len: int,
        select_keys: str,
    ) -> bytes:
        PAD = 24
        fonts_char = _load_fonts(80)
        fonts_info = _load_fonts(18)
        fonts_dafa = _load_fonts(16)

        probe = Image.new("RGB", (1, 1))
        pdraw = ImageDraw.Draw(probe)

        def msr(text, fonts):
            return self._measure(pdraw, text, fonts)

        # Build 打法 string
        dafa_parts = []
        for code, pos in codes_with_pos:
            dafa_parts.append(f"{code}({pos})")
        dafa_str = "  ".join(dafa_parts)
        if codes_with_pos:
            dafa_str += f"  共{len(codes_with_pos)}个"

        missing_placeholder = "??????"
        char_w, char_gh, char_bot = msr(word, fonts_char)
        info1 = f"方案: {schema_name}"
        info2 = f"来源: {owner_id}"
        i1w, i1gh, _ = msr(info1, fonts_info)
        i2w, i2gh, _ = msr(info2, fonts_info)
        dafa_label = "打法:"
        dlw, dlgh, dlbot = msr(dafa_label, fonts_info)
        dfw, dfgh, dfbot = msr(dafa_str if dafa_str else missing_placeholder, fonts_dafa)

        # info block height (glyph heights + gap)
        info_block_h = i1gh + 8 + i2gh
        # top section: char on left, info on right — height is the larger bottom offset
        top_bot = max(char_bot, info_block_h)

        info_x = PAD + char_w + 24
        IMG_W = max(info_x + max(i1w, i2w) + PAD,
                    PAD + dlw + 8 + dfw + PAD,
                    400)
        IMG_H = PAD + top_bot + 16 + dlbot + 8 + dfbot + PAD

        img = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Large character
        _render_text_with_fallback(draw, (PAD, PAD), word, fonts_char, (30, 30, 30))

        # Info block — vertically centred against the char bottom
        info_y = PAD + (top_bot - info_block_h) // 2
        _render_text_with_fallback(draw, (info_x, info_y), info1, fonts_info, (80, 80, 80))
        _render_text_with_fallback(draw, (info_x, info_y + i1gh + 8), info2, fonts_info, (80, 80, 80))

        # 打法 section
        y = PAD + top_bot + 16
        _render_text_with_fallback(draw, (PAD, y), dafa_label, fonts_info, (80, 80, 80))
        y += dlbot + 8
        color = (40, 40, 40) if dafa_str else (180, 60, 220)
        _render_text_with_fallback(draw, (PAD + 8, y), dafa_str or missing_placeholder, fonts_dafa, color)

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _make_multi_char_image(
        self,
        schema_name: str,
        word: str,
        char_codes: list[Optional[tuple[str, int]]],
        owner_id: str,
        max_len: int,
        select_keys: str,
    ) -> bytes:
        PAD = 24
        CELL_GAP = 16
        fonts_stats = _load_fonts(18)
        fonts_char = _load_fonts(36)
        fonts_code = _load_fonts(16)

        probe = Image.new("RGB", (1, 1))
        pdraw = ImageDraw.Draw(probe)

        def msr(text, fonts):
            return self._measure(pdraw, text, fonts)

        chars = list(word)
        n = len(chars)
        is_passthrough = [_is_passthrough(c) for c in chars]
        # 自编码：标点/数字未在码表中定义时，用字符自身作为编码
        is_self_coded = [
            is_passthrough[i] and char_codes[i] is None for i in range(n)
        ]

        def _next_self_coded(i: int) -> bool:
            # 仅自编码标点会触发前一字的首选键省略；数字/字母不省。
            return (
                i + 1 < n
                and is_self_coded[i + 1]
                and _is_punct(chars[i + 1])
            )

        # 每字键数：缺字 → None；自编码标点 → 1；普通编码 → 按 _key_presses 计；
        # 若下一个字是自编码标点且当前编码不足 max_len 且无选重，则省略首选键。
        per_presses: list[Optional[int]] = []
        for i in range(n):
            if is_self_coded[i]:
                per_presses.append(1)
            elif char_codes[i] is not None:
                code, pos = char_codes[i]
                if _next_self_coded(i) and pos == 1 and len(code) < max_len:
                    per_presses.append(len(code))  # 首选键省略
                else:
                    per_presses.append(_key_presses(code, max_len, pos))
            else:
                per_presses.append(None)

        # 缺字：仅统计需查码却未找到的字符（数字/标点不计）
        missing = sum(1 for p in per_presses if p is None)
        sel_count = sum(1 for c in char_codes if c is not None and c[1] > 1)
        counted_presses = [p for p in per_presses if p is not None]
        avg_len = (
            sum(counted_presses) / len(counted_presses) if counted_presses else 0.0
        )
        difficulty, diff_score = _text_difficulty(chars)

        line1 = f"难度: {difficulty}({diff_score})"
        line2 = f"【{schema_name}】"
        line3 = f"来源: {owner_id}    码长: {avg_len:.6f}"
        line4 = f"字数: {n}    选重: {sel_count}    缺字: {missing}"

        _, sgh, sbot = msr("难度: A", fonts_stats)
        _, char_gh, char_bot = msr("我", fonts_char)
        _, code_gh, code_bot = msr("abc", fonts_code)

        # 计算每个单元的显示码串
        def _cell_code_str(i: int) -> Optional[str]:
            if is_self_coded[i]:
                return chars[i]
            if char_codes[i] is None:
                return None
            code, pos = char_codes[i]
            # 下一个字是自编码标点 + 当前不到 max_len + 无选重 → 省略首选键
            if _next_self_coded(i) and pos == 1 and len(code) < max_len:
                return code
            return _code_display(code, max_len, pos, select_keys)

        # Per-cell widths: max of char width and code width, plus gap
        cell_widths = []
        for i, ch in enumerate(chars):
            cw, _, _ = msr(ch, fonts_char)
            cs = _cell_code_str(i)
            if cs is None:
                codew, _, _ = msr("??????", fonts_code)
            else:
                codew, _, _ = msr(cs, fonts_code)
            cell_widths.append(max(cw, codew) + CELL_GAP)

        stats_w = max(msr(l, fonts_stats)[0] for l in [line1, line2, line3, line4])
        MAX_IMG_W = 900
        content_w = sum(cell_widths)
        IMG_W = max(min(PAD * 2 + content_w, MAX_IMG_W), PAD * 2 + stats_w, 400)
        row_max_w = IMG_W - PAD * 2

        # 将单元按可用宽度切分成若干行
        rows: list[list[int]] = []
        cur: list[int] = []
        cur_w = 0
        for i, w in enumerate(cell_widths):
            if cur and cur_w + w > row_max_w:
                rows.append(cur)
                cur = [i]
                cur_w = w
            else:
                cur.append(i)
                cur_w += w
        if cur:
            rows.append(cur)

        LINE_H = sbot + 6          # stats line spacing (bottom offset + gap)
        STATS_H = LINE_H * 4
        UL_GAP = 4                 # gap between char bottom and underline
        CODE_GAP = 5               # gap between underline and code top
        ROW_H = char_bot + UL_GAP + 1 + CODE_GAP + code_bot
        ROW_GAP = 12
        IMG_H = (
            PAD + STATS_H + 16
            + ROW_H * len(rows) + ROW_GAP * (len(rows) - 1)
            + PAD
        )

        img = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Stats block
        y = PAD
        for line in [line1, line2, line3, line4]:
            _render_text_with_fallback(draw, (PAD, y), line, fonts_stats, (50, 50, 50))
            y += LINE_H

        # Character + underline + code grid，按行渲染
        grid_top = PAD + STATS_H + 16
        for r, row in enumerate(rows):
            char_y = grid_top + r * (ROW_H + ROW_GAP)
            ul_y = char_y + char_bot + UL_GAP
            code_y = ul_y + 1 + CODE_GAP

            x = PAD
            for i in row:
                ch = chars[i]
                cw, _, _ = msr(ch, fonts_char)
                cell_w = cell_widths[i] - CELL_GAP

                code_str = _cell_code_str(i)
                if code_str is None:
                    code_str = "??????"
                    is_select = False
                    is_missing = True
                    is_self = False
                else:
                    is_missing = False
                    is_self = is_self_coded[i]
                    is_select = (
                        not is_self
                        and char_codes[i] is not None
                        and char_codes[i][1] > 1
                    )
                codew, _, _ = msr(code_str, fonts_code)

                if is_missing:
                    char_color = (180, 60, 220)
                    code_color = (180, 60, 220)
                elif is_select:
                    char_color = (200, 40, 40)
                    code_color = (200, 40, 40)
                elif is_self:
                    char_color = (80, 110, 160)
                    code_color = (80, 110, 160)
                else:
                    char_color = (30, 30, 30)
                    code_color = (80, 80, 80)

                char_x = x + (cell_w - cw) // 2
                _render_text_with_fallback(draw, (char_x, char_y), ch, fonts_char, char_color)

                draw.line([(x, ul_y), (x + cell_w, ul_y)], fill=(140, 140, 140), width=1)

                code_x = x + (cell_w - codew) // 2
                _render_text_with_fallback(draw, (code_x, code_y), code_str, fonts_code, code_color)

                x += cell_widths[i]

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

    # ── 查码：[词提名] <字词>，或引用回复+[词提名] ─────────────────────────

    @staticmethod
    def _extract_reply_text(event: AstrMessageEvent) -> str:
        """从引用回复链中抽取所有纯文本片段，去掉空白字符。"""
        chain = event.get_messages()
        reply_seg: Optional[Reply] = next(
            (seg for seg in chain if seg.type == ComponentType.Reply), None
        )
        if not reply_seg or not getattr(reply_seg, "chain", None):
            return ""
        parts: list[str] = []
        for seg in reply_seg.chain:
            txt = getattr(seg, "text", None)
            if isinstance(txt, str) and txt:
                parts.append(txt)
        return re.sub(r"\s+", "", "".join(parts))

    @filter.regex(r"^(\S+)(?:\s+(\S+))?$")
    async def cmd_query(self, event: AstrMessageEvent):
        text = event.message_str.strip()
        m = re.match(r"^(\S+)(?:\s+(\S+))?$", text)
        if not m:
            return
        schema_name, word = m.group(1), m.group(2)
        if schema_name.startswith("%"):
            return
        info = self._schema_info(schema_name)
        if not info:
            return

        if not word:
            word = self._extract_reply_text(event)
            if not word:
                return

        try:
            img_bytes = self._make_image(schema_name, word, info["owner_id"], info["max_len"], info["select_keys"])
        except Exception as e:
            logger.exception(f"[im_schemas] 生成查询图片失败: {e}")
            yield event.plain_result(
                f"查询「{word}」时渲染图片失败，可能码表中含有当前字体无法绘制的字符。"
            )
            return
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
