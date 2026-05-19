import json
import re
import sqlite3
import string
import threading
import unicodedata
from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import NamedTuple, Optional

import aiohttp
try:
    from fontTools.ttLib import TTCollection, TTFont
except ImportError:
    raise ImportError(
        "fonttools>=4.0.0 is required. Please install it with: pip install fonttools>=4.0.0"
    ) from None
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import ComponentType, File, Image as AstrImage, Reply
from astrbot.api.star import Context, Star, StarTools
from astrbot.core import AstrBotConfig

DATA_DIR = StarTools.get_data_dir("im_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
_FONT_CMAPS: list[tuple[frozenset[int], Path]] = []
_FONT_CMAPS_INITIALIZED = False
_FONT_CMAPS_LOCK = threading.Lock()


def _ensure_font_cmaps_loaded() -> None:
    """延迟初始化字体 cmap，避免模块导入时阻塞。"""
    global _FONT_CMAPS, _FONT_CMAPS_INITIALIZED
    if _FONT_CMAPS_INITIALIZED:
        return
    with _FONT_CMAPS_LOCK:
        if _FONT_CMAPS_INITIALIZED:
            return
        _FONT_CMAPS[:] = [
            (_build_cmap(p), p)
            for p in _BUNDLED_FONTS
            if p.exists()
        ]
        _FONT_CMAPS_INITIALIZED = True


_FONT_CACHE: dict[int, tuple[ImageFont.FreeTypeFont, ...]] = {}
_FONT_CACHE_LOCK = threading.Lock()


def _load_fonts(size: int) -> tuple[ImageFont.FreeTypeFont, ...]:
    """按优先级加载所有可用字体，返回 Pillow 字体元组。
    使用手动缓存而非 lru_cache，以便在字体加载失败后能够重试。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    with _FONT_CACHE_LOCK:
        # Double-check after acquiring lock
        if size in _FONT_CACHE:
            return _FONT_CACHE[size]

        _ensure_font_cmaps_loaded()
        fonts: list[ImageFont.FreeTypeFont] = []
        for _, p in _FONT_CMAPS:
            try:
                fonts.append(ImageFont.truetype(str(p), size))
            except Exception:
                pass
        if not fonts:
            fonts.append(ImageFont.load_default())

        result = tuple(fonts)
        _FONT_CACHE[size] = result
        return result


def _pick_font_idx(ch: str) -> int:
    """根据 cmap 选择字体下标；找不到返回 0（首字体兜底）。"""
    _ensure_font_cmaps_loaded()
    cp = ord(ch)
    for i, (cmap, _) in enumerate(_FONT_CMAPS):
        if cp in cmap:
            return i
    return 0


def _pick_font(ch: str, fonts) -> ImageFont.FreeTypeFont:
    """选择字体，确保 fonts 非空时返回有效字体。"""
    if not fonts:
        return ImageFont.load_default()
    idx = _pick_font_idx(ch)
    return fonts[idx] if idx < len(fonts) else fonts[0]


@lru_cache(maxsize=8)
def _ref_ascent(fonts: tuple) -> int:
    """多字体回退时取最大 ascent 作为统一基线参考。
    各字体 hhea 度量不同（ChaiPUA 0.86em / SourceHan 1.16em / Plangothic 0.88em），
    若不统一基线，PUA 字符会比正文字体明显偏上。"""
    return max((f.getmetrics()[0] for f in fonts), default=0)


# 单字符 bbox 缓存：(size, font_idx, ch) → (left, top, right, bottom)
# 使用 anchor="ls"，1000 字渲染中重复字符极多，命中率高
@lru_cache(maxsize=65536)
def _char_bbox(size: int, font_idx: int, ch: str) -> tuple[int, int, int, int]:
    fonts = _load_fonts(size)
    f = fonts[font_idx] if font_idx < len(fonts) else fonts[0]
    # 创建临时 probe 进行测量，避免共享可变状态
    probe = Image.new("RGB", (1, 1))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), ch, font=f, anchor="ls")
    probe.close()
    return bbox


def _char_advance(size: int, ch: str) -> tuple[int, int, int, int]:
    """返回 (font_idx, advance_w, bb_top, bb_bottom)，命中字符级缓存。"""
    fi = _pick_font_idx(ch)
    bb = _char_bbox(size, fi, ch)
    return fi, bb[2] - bb[0], bb[1], bb[3]


def _render_text_with_fallback(draw, pos, text: str, fonts, fill, stroke_width: int = 0, baseline: Optional[int] = None):
    """逐字符渲染，所有候选字体共享同一条基线，避免 PUA 等回退字体高度不齐。
    复用 _char_bbox 缓存的 advance 宽度，避免每字一次 textbbox。
    stroke_width>0 时用同色描边模拟伪粗体（不改变 advance，仅微涨笔画）。
    baseline 显式指定时用作渲染基线，便于不同字号在同一基线上对齐。"""
    x, y_top = pos
    size = fonts[0].size if fonts else 0
    if baseline is None:
        baseline = y_top + _ref_ascent(fonts)
    for ch in text:
        fi, adv, _, _ = _char_advance(size, ch)
        f = fonts[fi] if fi < len(fonts) else fonts[0]
        draw.text(
            (x, baseline), ch, font=f, fill=fill, anchor="ls",
            stroke_width=stroke_width,
            stroke_fill=fill if stroke_width else None,
        )
        x += adv


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


_RANK_DATA_DIR = Path(__file__).parent / "rank_data"
_RANK_LETTER_DIGIT = string.ascii_letters + string.digits
_RANK_PUNCT_KEEP = set(":,.;!'\"")

_RANK_ZONG_CIPIN: dict = {}
_RANK_VALID_CHAR: set = set()
_RANK_PRE: set = set()
_RANK_DATA_INITIALIZED = False
_RANK_DATA_LOCK = threading.Lock()


def _ensure_rank_data_loaded() -> None:
    """延迟初始化难度测算数据，避免模块导入时阻塞。"""
    global _RANK_ZONG_CIPIN, _RANK_VALID_CHAR, _RANK_PRE, _RANK_DATA_INITIALIZED
    if _RANK_DATA_INITIALIZED:
        return
    with _RANK_DATA_LOCK:
        if _RANK_DATA_INITIALIZED:
            return
        try:
            with open(_RANK_DATA_DIR / "zongCiPin.json", encoding="utf-8") as f:
                _RANK_ZONG_CIPIN = json.load(f)
            with open(_RANK_DATA_DIR / "validChar.json", encoding="utf-8") as f:
                _RANK_VALID_CHAR = set(json.load(f).get("k", []))
            with open(_RANK_DATA_DIR / "pre.json", encoding="utf-8") as f:
                _RANK_PRE = set(json.load(f).get("k", []))
        except (OSError, ValueError) as e:
            logger.warning(f"[im_schemas] 难度测算数据加载失败: {e}")
        _RANK_DATA_INITIALIZED = True


def _text_difficulty(chars: list[str]) -> tuple[str, object]:
    """依据「字根月饼」的 rank 算法测算文本难度。
    返回 (难度等级中文, 分数)；分数 >100 时返回字符串「爆表{score}」。
    数据缺失或异常时返回 ("--", -1)。"""
    _ensure_rank_data_loaded()
    if not _RANK_VALID_CHAR or not _RANK_ZONG_CIPIN:
        return "--", -1

    s = "".join(chars).strip()
    # 过滤无效字符：空格仅在英数之间保留；西文标点仅在紧贴英数时保留
    ns_chars: list[str] = []
    for pos, c in enumerate(s):
        if c in _RANK_VALID_CHAR:
            ns_chars.append(c)
            continue
        if (c == " "
                and 0 < pos < len(s) - 1
                and s[pos - 1] in _RANK_LETTER_DIGIT
                and s[pos + 1] in _RANK_LETTER_DIGIT):
            ns_chars.append(c)
            continue
        if c in _RANK_PUNCT_KEEP and (
            (pos > 0 and s[pos - 1] in _RANK_LETTER_DIGIT)
            or (pos < len(s) - 1 and s[pos + 1] in _RANK_LETTER_DIGIT)
        ):
            ns_chars.append(c)
    s = "".join(ns_chars)
    if not s:
        return "--", -1

    n = len(s)
    # dp[i] = (累计码长, 词数, 上一切点)，最少码长优先；同码长按词数最少
    NEG = (-1, -1, -1)
    dp: list[tuple[int, int, int]] = [NEG] * (n + 1)
    dp[0] = (0, 0, -1)

    for pos in range(n):
        cur_len = 1
        while pos + cur_len <= n:
            w = s[pos:pos + cur_len]
            in_zong = w in _RANK_ZONG_CIPIN
            if cur_len == 1 or in_zong:
                wl = _RANK_ZONG_CIPIN[w][-1] if in_zong else 1
                new_pos = pos + cur_len
                tar = dp[new_pos]
                base = dp[pos]
                new_code_len = base[0] + wl
                new_word_cnt = base[1] + 1
                if (tar[0] == -1 or tar[0] > new_code_len
                        or (tar[0] == new_code_len and tar[1] > new_word_cnt)):
                    dp[new_pos] = (new_code_len, new_word_cnt, pos)
            if w in _RANK_PRE:
                cur_len += 1
            else:
                break

    if dp[n][0] == -1:
        return "--", -1

    water = 1.0
    hard = 0.0
    cur_pos = n
    while cur_pos != 0:
        pre_pos = dp[cur_pos][2]
        if pre_pos < 0:
            break
        w = s[pre_pos:cur_pos]
        if w in _RANK_ZONG_CIPIN:
            freq = _RANK_ZONG_CIPIN[w][0]
            if len(w) == 1:
                hard += min(10, pow(freq, 1.5) / 100000)
            else:
                water += 2000 / (freq + 2000)
        cur_pos = pre_pos

    score = round(hard / water, 2)

    if score < 0.1:
        label = "淼"
    elif score < 0.3:
        label = "水"
    elif score < 0.8:
        label = "易"
    elif score < 5:
        label = "普"
    elif score < 15:
        label = "难"
    else:
        label = "虐"

    if score > 100:
        return label, f"爆表{score}"
    return label, score


def _select_symbol(pos: int, select_keys: str) -> str:
    idx = pos - 1
    if 0 <= idx < len(select_keys):
        return select_keys[idx]
    return "?"


def _explicit_select_pos(code: str, select_keys: str) -> Optional[int]:
    """若 code 末位是 select_keys[1:] 中的字符，返回该选重位序（≥2），否则 None。
    select_keys[0] 是首选键（如空格 / `_`），不算选重。"""
    if not code or len(select_keys) < 2:
        return None
    idx = select_keys.find(code[-1])
    return idx + 1 if idx >= 1 else None


def _pure_code_len(code: str, select_keys: str) -> int:
    """编码的"纯长度"：末位若是任何 select_keys 字符（首选键或选重键）则扣除。
    用于按编码长度上色：y_ → 1，nim; → 3，nim → 3，aaaa → 4。"""
    if code and select_keys and code[-1] in select_keys:
        return len(code) - 1
    return len(code)


def _code_display(code: str, max_len: int, pos: int = 1, select_keys: str = "") -> str:
    if pos > 1:
        sym = _select_symbol(pos, select_keys)
        # code 已显式带选重键时不再重复拼接
        if code.endswith(sym):
            return code
        return code + sym
    if len(code) < max_len:
        first_key = select_keys[0] if select_keys else "_"
        return code + first_key
    return code


def _key_presses(code: str, max_len: int, pos: int = 1, select_keys: str = "") -> int:
    if pos > 1:
        sym = _select_symbol(pos, select_keys)
        # code 已显式带选重键时不再多算一键
        if code.endswith(sym):
            return len(code)
        return len(code) + 1
    return len(code) + (1 if len(code) < max_len else 0)


def _key_presses_last(code: str, max_len: int, pos: int = 1, select_keys: str = "", candidate_count: int = 1) -> int:
    """末位字词的按键数：打满码（len==max_len）且唯一候选时不需要补首选键。"""
    if pos > 1:
        sym = _select_symbol(pos, select_keys)
        if code.endswith(sym):
            return len(code)
        return len(code) + 1
    if len(code) < max_len or candidate_count > 1:
        return len(code) + 1
    return len(code)


def _code_display_last(code: str, max_len: int, pos: int = 1, select_keys: str = "", candidate_count: int = 1) -> str:
    """末位字词的显示码串：打满码（len==max_len）且唯一候选时不补首选键。"""
    if pos > 1:
        sym = _select_symbol(pos, select_keys)
        if code.endswith(sym):
            return code
        return code + sym
    if len(code) < max_len or candidate_count > 1:
        first_key = select_keys[0] if select_keys else "_"
        return code + first_key
    return code


def _omit_in_punct_context(
    code: str,
    pos: int,
    max_len: int,
    select_keys: str,
) -> Optional[tuple[int, str]]:
    """段后是自编码标点时的省略规则；可省略时返回 (按键数, 显示码串)，否则 None。

    - 首选键省略：pos==1 且 len(code)<max_len，省末位首选键（补齐到 max_len 的尾键不必敲）。
    - 末位 select_keys 省略：max_len<=0 且码表中条目末位本就是 select_keys 中的字符
      （含首选键 / 选重键），省末位（max_len<=0 表示作者声明不限码长，标点本身就是上屏触发）。
    """
    if pos == 1 and len(code) < max_len:
        return len(code), code
    if max_len <= 0 and code and select_keys and code[-1] in select_keys:
        return len(code) - 1, code[:-1]
    return None


_PAIR_EQUIVALENCE: dict[str, float] = {}
_PAIR_EQUIVALENCE_INITIALIZED = False
_PAIR_EQUIVALENCE_LOCK = threading.Lock()


def _ensure_pair_equivalence_loaded() -> None:
    """延迟初始化按键对当量表，避免模块导入时阻塞。"""
    global _PAIR_EQUIVALENCE, _PAIR_EQUIVALENCE_INITIALIZED
    if _PAIR_EQUIVALENCE_INITIALIZED:
        return
    with _PAIR_EQUIVALENCE_LOCK:
        if _PAIR_EQUIVALENCE_INITIALIZED:
            return
        path = Path(__file__).parent / "pair_equivalence.txt"
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2 and len(parts[0]) == 2:
                        try:
                            _PAIR_EQUIVALENCE[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
        except OSError:
            logger.warning("[im_schemas] 未找到 pair_equivalence.txt，当量计算不可用。")
        _PAIR_EQUIVALENCE_INITIALIZED = True


# QWERTY 热力图布局：(左缩进 in key 单位, [(label, key), ...])
_QWERTY_ROWS: list[tuple[float, list[tuple[str, str]]]] = [
    (0.0,  [("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"), ("6", "6"),
            ("7", "7"), ("8", "8"), ("9", "9"), ("0", "0"), ("-", "-"), ("=", "=")]),
    (0.5,  [("Q", "q"), ("W", "w"), ("E", "e"), ("R", "r"), ("T", "t"),
            ("Y", "y"), ("U", "u"), ("I", "i"), ("O", "o"), ("P", "p"),
            ("[", "["), ("]", "]")]),
    (0.75, [("A", "a"), ("S", "s"), ("D", "d"), ("F", "f"), ("G", "g"),
            ("H", "h"), ("J", "j"), ("K", "k"), ("L", "l"), (";", ";"),
            ("'", "'")]),
    (1.25, [("Z", "z"), ("X", "x"), ("C", "c"), ("V", "v"), ("B", "b"),
            ("N", "n"), ("M", "m"), (",", ","), (".", "."), ("/", "/")]),
]
_KEY_SIZE = 44
_KEY_GAP = 4
_KEYBOARD_W = 13 * _KEY_SIZE + 12 * _KEY_GAP
_KEYBOARD_H = 5 * _KEY_SIZE + 4 * _KEY_GAP


def _build_key_counts(key_seq: str) -> dict[str, int]:
    """统计每个键的按下次数；`_` 是首选键代号（多数方案配为空格），统一计入空格键。
    全角/中文标点按物理键归并：`、。—…` 分别落到 `/ . - .`。"""
    # NFKC 把全角字母/数字/标点（FF00 区段）统一成半角
    seq = unicodedata.normalize("NFKC", key_seq)
    cjk_map = {
        "。": ".", "、": "/", "—": "-", "…": ".",
        "「": "[", "」": "]", "『": "[", "』": "]", "【": "[", "】": "]",
        "《": "<", "》": ">", "·": "`",
    }
    counts: dict[str, int] = {}
    for ch in seq:
        c = cjk_map.get(ch, ch).lower()
        if c == "_":
            c = " "
        counts[c] = counts.get(c, 0) + 1
    return counts


def _pair_equivalence_avg(key_seq: str) -> Optional[float]:
    """整段打法所有相邻按键对的均当量；不足两键或无命中返回 None。"""
    _ensure_pair_equivalence_loaded()
    if len(key_seq) < 2 or not _PAIR_EQUIVALENCE:
        return None
    seq = key_seq.lower()
    total = 0.0
    n = 0
    for i in range(len(seq) - 1):
        v = _PAIR_EQUIVALENCE.get(seq[i:i + 2])
        if v is not None:
            total += v
            n += 1
    return total / n if n else None


_DB_INITIALIZED = False
_DB_LOCK = threading.Lock()

# Schema name cache: set of all schema names, updated on import/delete
_SCHEMA_NAMES_CACHE: set[str] = set()
_SCHEMA_NAMES_CACHE_LOCK = threading.Lock()


def _init_db(conn: sqlite3.Connection) -> None:
    # 写入 db 文件头的持久 PRAGMA：跨连接生效，但 auto_vacuum 切换需要一次 VACUUM 提交。
    if conn.execute("PRAGMA auto_vacuum").fetchone()[0] != 2:  # 2 = INCREMENTAL
        conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
        has_tables = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
        ).fetchone() is not None
        if has_tables:
            conn.execute("VACUUM")
    conn.execute("PRAGMA journal_mode = WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS schemas (
            name         TEXT PRIMARY KEY,
            owner_id     TEXT NOT NULL,
            owner_alias  TEXT NOT NULL DEFAULT '',
            select_keys  TEXT NOT NULL DEFAULT '',
            max_len      INTEGER NOT NULL DEFAULT 4,
            punct_key    TEXT NOT NULL DEFAULT ''
        )
    """)
    # 迁移：旧库可能缺少 owner_alias 列
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(schemas)").fetchall()
    }
    if "owner_alias" not in cols:
        conn.execute(
            "ALTER TABLE schemas ADD COLUMN owner_alias TEXT NOT NULL DEFAULT ''"
        )
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
    # 反向索引：按 (schema, code) 查同码字词，用于计算选重位序
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_codes_by_code ON codes(schema_name, code)"
    )


@contextmanager
def _open_db():
    """Context manager for database connections with automatic initialization and cleanup."""
    global _DB_INITIALIZED
    conn = sqlite3.connect(DB_PATH)
    # per-connection PRAGMA：WAL 下默认 synchronous=FULL，调到 NORMAL 减少 fsync。
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        with _DB_LOCK:
            if not _DB_INITIALIZED:
                _init_db(conn)
                _DB_INITIALIZED = True
        yield conn
    finally:
        conn.close()


class Segment(NamedTuple):
    """一个渲染单元：可能是词组、单字、自编码标点/数字、或缺字。

    - text:            原文（≥1 个字符）
    - code:            码表中的原始编码；自编码或缺字时为 None
    - pos:             同码字词中的位序（1 起；自编码 / 缺字时为 1）
    - is_self_coded:   是否为自编码 passthrough（标点 / 数字 / 字母）
    - is_missing:      是否为缺字（需查码却未命中，仅出现在单字段）
    - candidate_count: 该编码下的候选字词总数（1 表示唯一候选）
    """
    text: str
    code: Optional[str]
    pos: int
    is_self_coded: bool
    is_missing: bool
    candidate_count: int = 1


class IMSchemasPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

    def _all_group_schemas(self) -> list[str]:
        """读取配置中的 all 组方案名列表，过滤掉空串和不存在的方案。"""
        names = self.config.get("all_schemas", []) or []
        return [n for n in (str(s).strip() for s in names) if n and self._schema_exists(n)]

    def _is_all_trigger(self, name: str) -> bool:
        """判断 name 是否匹配配置中的 all 组触发词（支持正则全匹配）。"""
        pat = (self.config.get("all_trigger", "") or "").strip()
        if not pat:
            return False
        try:
            return re.fullmatch(pat, name) is not None
        except re.error:
            return name == pat

    # ── 数据库操作 ──────────────────────────────────────────────────────────

    def _schema_exists(self, name: str) -> bool:
        # Check cache first
        if name in _SCHEMA_NAMES_CACHE:
            return True
        # If not in cache, check database and update cache
        with _open_db() as conn:
            row = conn.execute(
                "SELECT 1 FROM schemas WHERE name = ?", (name,)
            ).fetchone()
            exists = row is not None
            if exists:
                with _SCHEMA_NAMES_CACHE_LOCK:
                    _SCHEMA_NAMES_CACHE.add(name)
            return exists

    def _schema_owner(self, name: str) -> Optional[str]:
        with _open_db() as conn:
            row = conn.execute(
                "SELECT owner_id FROM schemas WHERE name = ?", (name,)
            ).fetchone()
            return row[0] if row else None

    @staticmethod
    def _display_owner(owner_id: str, owner_alias: str) -> str:
        """返回用于展示的来源：有别名用别名，否则用 owner_id。"""
        return owner_alias if owner_alias else owner_id

    def _count_user_schemas(self, owner_id: str) -> int:
        with _open_db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM schemas WHERE owner_id = ?", (owner_id,)
            ).fetchone()
            return int(row[0]) if row else 0

    def _import_schema(
        self,
        name: str,
        owner_id: str,
        entries: list[tuple[str, str]],
    ) -> int:
        """导入或刷新词提的码表条目。

        词提首次创建时使用默认 select_keys / max_len / punct_key；
        若同名词提已存在，仅刷新 owner_id 与码表条目，
        保留用户先前通过「设置词提」配置的三项参数。
        """
        with _open_db() as conn:
            conn.execute(
                """
                INSERT INTO schemas(name, owner_id, select_keys, max_len, punct_key)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    owner_id = excluded.owner_id
                """,
                (name, owner_id, DEFAULT_SELECT_KEYS, DEFAULT_MAX_LEN, ""),
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
            # 把更新过程中产生的空 page 归还 OS，避免文件单调膨胀。
            conn.execute("PRAGMA incremental_vacuum")
        # Update schema name cache
        with _SCHEMA_NAMES_CACHE_LOCK:
            _SCHEMA_NAMES_CACHE.add(name)
        return count

    def _update_schema_setting(self, name: str, field: str, value) -> None:
        """更新单个词提配置字段（select_keys / max_len / punct_key / owner_alias）。"""
        QUERIES = {
            "select_keys": "UPDATE schemas SET select_keys = ? WHERE name = ?",
            "max_len": "UPDATE schemas SET max_len = ? WHERE name = ?",
            "punct_key": "UPDATE schemas SET punct_key = ? WHERE name = ?",
            "来源": "UPDATE schemas SET owner_alias = ? WHERE name = ?",
        }
        sql = QUERIES.get(field)
        if sql is None:
            raise ValueError(f"unsupported schema field: {field}")

        with _open_db() as conn:
            conn.execute(sql, (value, name))
            conn.commit()

    def _delete_schema(self, name: str) -> None:
        with _open_db() as conn:
            conn.execute("DELETE FROM codes WHERE schema_name = ?", (name,))
            conn.execute("DELETE FROM schemas WHERE name = ?", (name,))
            conn.commit()
            conn.execute("PRAGMA incremental_vacuum")
        # Remove from schema name cache
        with _SCHEMA_NAMES_CACHE_LOCK:
            _SCHEMA_NAMES_CACHE.discard(name)

    def _max_word_len(self, schema_name: str, conn: Optional[sqlite3.Connection] = None) -> int:
        """该方案中最长词组的字数；用于限制 DP 候选子串长度。"""
        if conn is None:
            with _open_db() as conn:
                row = conn.execute(
                    "SELECT MAX(LENGTH(word)) FROM codes WHERE schema_name = ?",
                    (schema_name,),
                ).fetchone()
        else:
            row = conn.execute(
                "SELECT MAX(LENGTH(word)) FROM codes WHERE schema_name = ?",
                (schema_name,),
            ).fetchone()
        # SQLite LENGTH 对 UTF-8 字符串返回字符数（非字节数），可直接用
        return int(row[0]) if row and row[0] else 1

    def _query_word_codes(
        self, schema_name: str, words: list[str], conn: Optional[sqlite3.Connection] = None
    ) -> dict[str, tuple[str, int, int]]:
        """批量查询任意词（含单字与词组）的最优 (code, pos, candidate_count)。
        最优定义：编码长度最短，相同长度按 code 字典序最小。
        pos 是同码字词中（按 rowid 排序）的位序；candidate_count 是该编码下的候选总数。"""
        unique_words = list({w for w in words if w})
        if not unique_words:
            return {}

        def _fetch_in_with_conn(sql_tpl: str, items: list[str], c: sqlite3.Connection) -> list[tuple]:
            CHUNK = 500
            out: list[tuple] = []
            for i in range(0, len(items), CHUNK):
                batch = items[i:i + CHUNK]
                placeholders = ",".join("?" * len(batch))
                out.extend(c.execute(
                    sql_tpl.format(ph=placeholders),
                    (schema_name, *batch),
                ).fetchall())
            return out

        if conn is None:
            with _open_db() as conn:
                rows = _fetch_in_with_conn(
                    "SELECT word, code FROM codes WHERE schema_name = ? AND word IN ({ph})",
                    unique_words,
                    conn,
                )

                best_code: dict[str, str] = {}
                for w, code in rows:
                    cur = best_code.get(w)
                    if cur is None or (len(code), code) < (len(cur), cur):
                        best_code[w] = code

                if not best_code:
                    return {}

                unique_codes = list(set(best_code.values()))
                rows = _fetch_in_with_conn(
                    "SELECT code, word FROM codes WHERE schema_name = ? AND code IN ({ph}) ORDER BY rowid",
                    unique_codes,
                    conn,
                )

                words_by_code: dict[str, list[str]] = {}
                for code, w in rows:
                    words_by_code.setdefault(code, []).append(w)

                result: dict[str, tuple[str, int, int]] = {}
                for w, code in best_code.items():
                    words_for_code = words_by_code.get(code, [])
                    try:
                        pos = words_for_code.index(w) + 1
                    except ValueError:
                        pos = 1
                    result[w] = (code, pos, len(words_for_code))
                return result
        else:
            rows = _fetch_in_with_conn(
                "SELECT word, code FROM codes WHERE schema_name = ? AND word IN ({ph})",
                unique_words,
                conn,
            )

            best_code: dict[str, str] = {}
            for w, code in rows:
                cur = best_code.get(w)
                if cur is None or (len(code), code) < (len(cur), cur):
                    best_code[w] = code

            if not best_code:
                return {}

            unique_codes = list(set(best_code.values()))
            rows = _fetch_in_with_conn(
                "SELECT code, word FROM codes WHERE schema_name = ? AND code IN ({ph}) ORDER BY rowid",
                unique_codes,
                conn,
            )

            words_by_code: dict[str, list[str]] = {}
            for code, w in rows:
                words_by_code.setdefault(code, []).append(w)

            result: dict[str, tuple[str, int, int]] = {}
            for w, code in best_code.items():
                words_for_code = words_by_code.get(code, [])
                try:
                    pos = words_for_code.index(w) + 1
                except ValueError:
                    pos = 1
                result[w] = (code, pos, len(words_for_code))
            return result

    def _query_segments(
        self,
        schema_name: str,
        word: str,
        max_len: int,
        select_keys: str,
        force_single: bool = False,
    ) -> list[Segment]:
        """用 DP 把 word 切成最少按键数的若干 Segment。

        规则：
          - 候选切片仅考虑 ≤ 该方案最长词长 的子串
          - passthrough 字符（数字/字母/标点/空白）若码表中无定义则单独成段、自编码 1 键；
            若有定义则按码表编码处理，但仍单独成段（不与其他字符合并成词组）
          - 若一段普通编码满足「下一段是自编码标点 + pos==1 + len(code)<max_len」，
            则首选键省略，按 len(code) 计费（与原有渲染逻辑一致）
          - force_single=True 时只生成单字候选，强制按字逐一切分。
        """
        n = len(word)
        if n == 0:
            return []

        chars = list(word)
        passthrough = [_is_passthrough(c) for c in chars]

        # Use a single connection for both queries
        with _open_db() as conn:
            max_word_len = 1 if force_single else self._max_word_len(schema_name, conn)

            # 收集所有候选子串：
            # - 单字候选：所有 chars[i:i+1]，含 passthrough 字符（若码表中有定义可用）
            # - 词组候选：长度 ≥2 且不跨越 passthrough 边界（passthrough 字符不参与组词）
            candidates: set[str] = set()
            for i in range(n):
                candidates.add(word[i:i + 1])
                if passthrough[i] or force_single:
                    continue
                limit = min(n, i + max_word_len)
                for j in range(i + 2, limit + 1):
                    if passthrough[j - 1]:
                        break
                    candidates.add(word[i:j])

            codes = self._query_word_codes(schema_name, list(candidates), conn)
            # 码表里若直接收录了末位为选重键的条目（如 aa;），用末位键覆盖 pos，
            # 让后续渲染/统计自然把它当作选重处理。
            codes = {
                w: ((code, _explicit_select_pos(code, select_keys) or pos, cnt))
                for w, (code, pos, cnt) in codes.items()
            }

            INF = float("inf")
            # dp[i] = 处理前 i 个字符的最少按键数；choice[i] = (start, segment) 用于回溯
            dp: list[float] = [INF] * (n + 1)
            choice: list[Optional[tuple[int, Segment]]] = [None] * (n + 1)
            dp[0] = 0.0

            for i in range(n):
                if dp[i] == INF:
                    continue
                ch = chars[i]
                had_any = False

                # 单字（含 passthrough 字符）：先试码表
                single_hit = codes.get(ch)
                if single_hit is not None:
                    code, pos, cnt = single_hit
                    seg = Segment(
                        text=ch, code=code, pos=pos,
                        is_self_coded=False, is_missing=False, candidate_count=cnt,
                    )
                    cost = dp[i] + _key_presses(code, max_len, pos, select_keys)
                    if cost < dp[i + 1]:
                        dp[i + 1] = cost
                        choice[i + 1] = (i, seg)
                    had_any = True

                # passthrough 自编码兜底（无论是否同时有码表条目，自编码 1 键也参与比较）
                if passthrough[i]:
                    seg = Segment(
                        text=ch, code=None, pos=1,
                        is_self_coded=True, is_missing=False,
                    )
                    cost = dp[i] + 1
                    if cost < dp[i + 1]:
                        dp[i + 1] = cost
                        choice[i + 1] = (i, seg)
                    had_any = True
                    # passthrough 不参与组词
                    continue

                # 词组候选：长度 ≥2 且不跨越 passthrough（force_single 时跳过）
                if force_single:
                    continue
                limit = min(n, i + max_word_len)
                for j in range(i + 2, limit + 1):
                    if passthrough[j - 1]:
                        break
                    sub = word[i:j]
                    hit = codes.get(sub)
                    if hit is None:
                        continue
                    code, pos, cnt = hit
                    seg = Segment(
                        text=sub, code=code, pos=pos,
                        is_self_coded=False, is_missing=False, candidate_count=cnt,
                    )
                    cost = dp[i] + _key_presses(code, max_len, pos, select_keys)
                    if cost < dp[j]:
                        dp[j] = cost
                        choice[j] = (i, seg)
                    had_any = True

                # 缺字兜底：保证 dp 总能推进；缺字不计入码长，cost 不增加
                if not had_any:
                    seg = Segment(
                        text=ch, code=None, pos=1,
                        is_self_coded=False, is_missing=True,
                    )
                    cost = dp[i]
                    if cost < dp[i + 1]:
                        dp[i + 1] = cost
                        choice[i + 1] = (i, seg)

            # 回溯
            segs: list[Segment] = []
            idx = n
            while idx > 0:
                step = choice[idx]
                if step is None:
                    segs.append(Segment(
                        text=word[idx - 1], code=None, pos=1,
                        is_self_coded=False, is_missing=True,
                    ))
                    idx -= 1
                    continue
                start, seg = step
                segs.append(seg)
                idx = start
            segs.reverse()
            return segs

    def _query_codes(self, schema_name: str, word: str) -> list[str]:
        with _open_db() as conn:
            rows = conn.execute(
                """
                SELECT code FROM codes
                WHERE schema_name = ? AND word = ?
                ORDER BY length(code), code
                """,
                (schema_name, word),
            ).fetchall()
            return [r[0] for r in rows]

    def _query_codes_with_positions(self, schema_name: str, word: str) -> list[tuple[str, int]]:
        with _open_db() as conn:
            codes = [
                r[0] for r in conn.execute(
                    "SELECT code FROM codes WHERE schema_name = ? AND word = ? ORDER BY length(code), code",
                    (schema_name, word),
                ).fetchall()
            ]
            if not codes:
                return []

            # 批量取这些 code 下的所有同码字词，一次 SQL 解决
            placeholders = ",".join("?" * len(codes))
            rows = conn.execute(
                f"SELECT code, word FROM codes WHERE schema_name = ? AND code IN ({placeholders}) ORDER BY rowid",
                (schema_name, *codes),
            ).fetchall()

            words_by_code: dict[str, list[str]] = {}
            for code, w in rows:
                words_by_code.setdefault(code, []).append(w)

            result: list[tuple[str, int]] = []
            for code in codes:
                words_for_code = words_by_code.get(code, [])
                try:
                    pos = words_for_code.index(word) + 1
                except ValueError:
                    pos = 1
                result.append((code, pos))
            return result

    def _lookup_by_code(
        self, schema_name: str, codes: list[str]
    ) -> dict[str, list[str]]:
        """精确反查：每个唯一 code → 该 code 下所有字词（按 rowid，即码表录入顺序）。
        与上屏位序对齐：列表第 i 项的位序为 i+1。"""
        unique_codes = list({c for c in codes if c})
        if not unique_codes:
            return {}
        with _open_db() as conn:
            out: dict[str, list[str]] = {c: [] for c in unique_codes}
            CHUNK = 500
            for i in range(0, len(unique_codes), CHUNK):
                batch = unique_codes[i:i + CHUNK]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT code, word FROM codes WHERE schema_name = ? AND code IN ({placeholders}) ORDER BY rowid",
                    (schema_name, *batch),
                ).fetchall()
                for code, word in rows:
                    out[code].append(word)
            return out

    def _list_all_schemas(self) -> list[tuple[str, str, str]]:
        """返回所有词提的 (name, owner_id, owner_alias) 列表，按 name 字典序。"""
        with _open_db() as conn:
            rows = conn.execute(
                "SELECT name, owner_id, owner_alias FROM schemas ORDER BY name"
            ).fetchall()
            return [(r[0], r[1], r[2]) for r in rows]

    def _schema_info(self, name: str) -> Optional[dict]:
        with _open_db() as conn:
            row = conn.execute(
                "SELECT owner_id, owner_alias, select_keys, max_len, punct_key FROM schemas WHERE name = ?",
                (name,),
            ).fetchone()
            if not row:
                return None
            owner_id, owner_alias, select_keys, max_len, punct_key = row
            chars_rows = conn.execute(
                "SELECT DISTINCT code FROM codes WHERE schema_name = ?", (name,)
            ).fetchall()
            chars: set[str] = set()
            for (code,) in chars_rows:
                chars.update(code)
            return {
                "owner_id": owner_id,
                "owner_alias": owner_alias,
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
    def _decorate(content: Image.Image) -> Image.Image:
        """把内容图嵌入装饰画布：淡绿渐变背景 + 圆角白卡 + 顶部彩条 + 柔和阴影。
        参考 tmp/text_to_img.html 的视觉风格。"""
        cw, ch = content.size
        OUTER = 28          # 卡片到画布边缘的留白
        RADIUS = 22         # 卡片圆角半径
        SHADOW_BLUR = 18
        SHADOW_OFFSET = 6
        SHADOW_PAD = SHADOW_BLUR + SHADOW_OFFSET
        BAR_INSET = 28      # 顶部彩条左右内缩
        BAR_H = 3

        W = cw + OUTER * 2
        H = ch + OUTER * 2

        # 1) 渐变背景：上浅下更浅的青绿，叠两个角落径向高光
        bg = Image.new("RGB", (W, H), (247, 250, 248))
        bg_draw = ImageDraw.Draw(bg)
        for y in range(H):
            t = y / max(1, H - 1)
            # #f7faf8 → #f3f7f5
            r = int(247 + (243 - 247) * t)
            g = int(250 + (247 - 250) * t)
            b = int(248 + (245 - 248) * t)
            bg_draw.line([(0, y), (W, y)], fill=(r, g, b))
        # 左上 / 右上柔光（半径 ~ 较短边的 0.7）
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        rg = max(W, H)
        # 左上 #5fc999 ~ 8% 不透明
        gd.ellipse(
            [-rg // 2, -rg // 2, rg // 2, rg // 2],
            fill=(95, 201, 153, 28),
        )
        # 右上 #3eaf7c ~ 6% 不透明
        gd.ellipse(
            [W - rg // 2, -rg // 2, W + rg // 2, rg // 2],
            fill=(62, 175, 124, 22),
        )
        glow = glow.filter(ImageFilter.GaussianBlur(60))
        bg = bg.convert("RGBA")
        bg.alpha_composite(glow)

        # 2) 卡片阴影：黑 12% 的圆角矩形 → 高斯模糊 → 偏移
        shadow_canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow_canvas)
        sd.rounded_rectangle(
            [OUTER, OUTER + SHADOW_OFFSET, OUTER + cw, OUTER + ch + SHADOW_OFFSET],
            radius=RADIUS,
            fill=(45, 138, 95, 38),
        )
        shadow_canvas = shadow_canvas.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
        bg.alpha_composite(shadow_canvas)

        # 3) 圆角白卡（用 mask 把内容图剪成圆角），再加 1px 浅边
        card = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
        # 把传入的内容（白底）原样贴到 card 上
        if content.mode != "RGBA":
            card.paste(content.convert("RGB"), (0, 0))
        else:
            card.alpha_composite(content)
        mask = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, cw, ch], radius=RADIUS, fill=255,
        )
        card.putalpha(mask)
        # 边框：在 mask 上用浅色描边，单独贴
        border = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        ImageDraw.Draw(border).rounded_rectangle(
            [0, 0, cw - 1, ch - 1], radius=RADIUS,
            outline=(232, 236, 239, 255), width=1,
        )
        card.alpha_composite(border)
        bg.alpha_composite(card, dest=(OUTER, OUTER))

        # 4) 顶部彩条（左右淡出 → 中间饱和）
        bar = Image.new("RGBA", (cw - BAR_INSET * 2, BAR_H), (0, 0, 0, 0))
        bar_draw = ImageDraw.Draw(bar)
        bw = bar.width
        for x in range(bw):
            t = x / max(1, bw - 1)
            if t < 0.2:
                color = (62, 175, 124, int(255 * (t / 0.2) * 0.6))
            elif t < 0.5:
                k = (t - 0.2) / 0.3
                r = int(62 + (79 - 62) * k)
                g = int(175 + (209 - 175) * k)
                b = int(124 + (168 - 124) * k)
                color = (r, g, b, int(255 * 0.6))
            elif t < 0.8:
                k = (t - 0.5) / 0.3
                r = int(79 + (62 - 79) * k)
                g = int(209 + (175 - 209) * k)
                b = int(168 + (124 - 168) * k)
                color = (r, g, b, int(255 * 0.6))
            else:
                k = (t - 0.8) / 0.2
                color = (62, 175, 124, int(255 * (1 - k) * 0.6))
            bar_draw.line([(x, 0), (x, BAR_H)], fill=color)
        bg.alpha_composite(bar, dest=(OUTER + BAR_INSET, OUTER))

        return bg.convert("RGB")

    @staticmethod
    def _to_png_bytes(img: Image.Image, decorate: bool = True) -> bytes:
        """统一出口：可选地套上装饰边框，再编码为 PNG 字节。"""
        if decorate:
            img = IMSchemasPlugin._decorate(img)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _measure(draw, text: str, fonts) -> tuple[int, int, int]:
        """Returns (width, glyph_height, bottom_offset).
        基于统一基线（max ascent）计算，与 _render_text_with_fallback 保持一致：
          baseline = y_top + ref_ascent
          bottom_offset = ref_ascent + max(per-char descent below baseline)
          glyph_height  = bottom_offset - min(per-char top above baseline)
        这样多字体回退（如 ChaiPUA 与正文字体）的尺寸不会因度量差异而错位。
        命中字符级 bbox 缓存，重复字符不再重新测量。
        """
        if not text:
            return 0, 0, 0
        ref_asc = _ref_ascent(fonts)
        size = fonts[0].size if fonts else 0
        w = 0
        max_below = 0     # 基线下最远像素
        min_above = 0     # 基线上最近像素（相对基线，负值；0 表示恰好在基线）
        for ch in text:
            _, adv, top, bot = _char_advance(size, ch)
            w += adv
            if bot > max_below:
                max_below = bot
            if top < min_above:
                min_above = top
        bot_off = ref_asc + max_below
        gh = bot_off - (ref_asc + min_above)
        return w, gh, bot_off

    def _make_image(self, schema_name: str, word: str, owner_id: str, owner_alias: str, max_len: int, select_keys: str, force_single: bool = False, show_keyboard: bool = True, decorate: bool = True) -> bytes:
        if len(word) == 1:
            codes_with_pos = self._query_codes_with_positions(schema_name, word)
            # 末位为选重键的条目，按选重键位序覆盖 pos
            codes_with_pos = [
                (code, _explicit_select_pos(code, select_keys) or pos)
                for code, pos in codes_with_pos
            ]
            return self._make_single_char_image(schema_name, word, codes_with_pos, owner_id, owner_alias, max_len, select_keys, decorate=decorate)
        segments = self._query_segments(schema_name, word, max_len, select_keys, force_single=force_single)
        return self._make_multi_char_image(schema_name, word, segments, owner_id, owner_alias, max_len, select_keys, show_keyboard=show_keyboard, decorate=decorate)

    def _compute_stats(
        self,
        word: str,
        segments: list[Segment],
        max_len: int,
        select_keys: str,
    ) -> dict:
        """计算一个方案对一段文本的统计：码长 / 选重 / 缺字 / 当量。
        码长按「段」累计：每段一组按键。提取为公用以供 all 组使用。"""
        n = len(segments)

        def _next_self_coded(i: int) -> bool:
            return i + 1 < n and segments[i + 1].is_self_coded and _is_punct(segments[i + 1].text)

        per_presses: list[int] = []
        key_seq_parts: list[str] = []
        for i, seg in enumerate(segments):
            is_last = (i == n - 1)
            if seg.is_missing:
                # 缺字显示为 6 个问号，码长按 6 计入（每个缺字字符算 6 键）
                per_presses.append(6 * len(seg.text))
            elif seg.is_self_coded:
                per_presses.append(1)
                key_seq_parts.append(seg.text)
            else:
                code, pos = seg.code, seg.pos
                omit = _omit_in_punct_context(code, pos, max_len, select_keys) if _next_self_coded(i) else None
                if omit is not None:
                    presses, disp = omit
                    per_presses.append(presses)
                    key_seq_parts.append(disp)
                elif is_last:
                    per_presses.append(_key_presses_last(code, max_len, pos, select_keys, seg.candidate_count))
                    key_seq_parts.append(_code_display_last(code, max_len, pos, select_keys, seg.candidate_count))
                else:
                    per_presses.append(_key_presses(code, max_len, pos, select_keys))
                    key_seq_parts.append(_code_display(code, max_len, pos, select_keys))

        missing = sum(len(seg.text) for seg in segments if seg.is_missing)
        sel_count = sum(1 for seg in segments if not seg.is_self_coded and not seg.is_missing and seg.pos > 1)
        # 码长按「字」均摊；缺字按 6 键计入分子与分母
        counted_chars = sum(len(seg.text) for seg in segments)
        total_presses = sum(per_presses)
        avg_len = total_presses / counted_chars if counted_chars else 0.0
        equivalence = _pair_equivalence_avg("".join(key_seq_parts))
        six_speed = 360 / avg_len if avg_len else None
        return {
            "avg_len": avg_len,
            "sel_count": sel_count,
            "missing": missing,
            "equivalence": equivalence,
            "six_speed": six_speed,
        }

    def _all_sort_key(self, stats: dict) -> tuple:
        """all 组排序：六击速度↓ → 选重↑ → 当量↑（None 视为 +inf）→ 缺字↑。"""
        eq = stats["equivalence"]
        spd = stats["six_speed"]
        return (
            float("inf") if spd is None else -spd,
            stats["sel_count"],
            float("inf") if eq is None else eq,
            stats["missing"],
        )

    def _make_single_char_image(
        self,
        schema_name: str,
        word: str,
        codes_with_pos: list[tuple[str, int]],
        owner_id: str,
        owner_alias: str,
        max_len: int,
        select_keys: str,
        decorate: bool = True,
    ) -> bytes:
        PAD = 24
        fonts_char = _load_fonts(80)
        fonts_info = _load_fonts(18)
        fonts_dafa = _load_fonts(16)
        fonts_dafa_sel = _load_fonts(18)  # 选重码略大一号，更显眼

        probe = Image.new("RGB", (1, 1))
        pdraw = ImageDraw.Draw(probe)

        def msr(text, fonts):
            return self._measure(pdraw, text, fonts)

        # Build 打法 string
        # 同时构造分段着色 / 字体列表：每条候选可能是选重（红色 + 略大粗体）或普通
        DEFAULT_COLOR = (40, 40, 40)
        SELECT_COLOR = (200, 40, 40)
        dafa_parts = []
        # 每段：(text, color, fonts, stroke_width)
        dafa_pieces: list[tuple[str, tuple[int, int, int], tuple, int]] = []
        SEP = "  "
        for idx, (code, pos) in enumerate(codes_with_pos):
            piece = f"{code}({pos})"
            dafa_parts.append(piece)
            if idx > 0:
                dafa_pieces.append((SEP, DEFAULT_COLOR, fonts_dafa, 0))
            if pos > 1:
                dafa_pieces.append((piece, SELECT_COLOR, fonts_dafa_sel, 1))
            else:
                dafa_pieces.append((piece, DEFAULT_COLOR, fonts_dafa, 0))
        dafa_str = SEP.join(dafa_parts)
        if codes_with_pos:
            tail = f"{SEP}共{len(codes_with_pos)}个"
            dafa_str += tail
            dafa_pieces.append((tail, DEFAULT_COLOR, fonts_dafa, 0))

        missing_placeholder = "??????"
        char_w, char_gh, char_bot = msr(word, fonts_char)
        info1 = f"方案: {schema_name}"
        info2 = f"来源: {self._display_owner(owner_id, owner_alias)}"
        i1w, i1gh, _ = msr(info1, fonts_info)
        i2w, i2gh, _ = msr(info2, fonts_info)
        dafa_label = "打法:"
        dlw, dlgh, dlbot = msr(dafa_label, fonts_info)
        # 打法行整体度量：选重段用 fonts_dafa_sel（更大），普通段用 fonts_dafa
        if dafa_pieces:
            dfw = 0
            dfbot = 0
            piece_widths: list[int] = []
            for piece, _color, pfonts, _sw in dafa_pieces:
                pw, _, pbot = msr(piece, pfonts)
                piece_widths.append(pw)
                dfw += pw
                if pbot > dfbot:
                    dfbot = pbot
        else:
            piece_widths = []
            dfw, _, dfbot = msr(missing_placeholder, fonts_dafa)

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
        if dafa_str:
            # 共享基线：用最大字号的 ascent 作为基线锚点，混排不同字号也对齐
            ref_asc = max(_ref_ascent(p[2]) for p in dafa_pieces)
            shared_baseline = y + ref_asc
            x = PAD + 8
            for (piece, color, pfonts, sw), pw in zip(dafa_pieces, piece_widths):
                _render_text_with_fallback(
                    draw, (x, y), piece, pfonts, color,
                    stroke_width=sw, baseline=shared_baseline,
                )
                x += pw
        else:
            _render_text_with_fallback(draw, (PAD + 8, y), missing_placeholder, fonts_dafa, (180, 60, 220))

        probe.close()
        return self._to_png_bytes(img, decorate=decorate)

    def _draw_keyboard_heatmap(
        self,
        draw: ImageDraw.ImageDraw,
        origin: tuple[int, int],
        counts: dict[str, int],
    ) -> None:
        """在 origin 处绘制一个 13 列宽的 QWERTY 键盘热力图。
        色阶：未按 → 浅灰；按下 → 浅黄到深红的线性渐变（按 max(counts) 归一化）。"""
        x0, y0 = origin
        max_n = max(counts.values(), default=0)
        fonts_label = _load_fonts(14)
        fonts_count = _load_fonts(11)

        def heat_color(n: int) -> tuple[int, int, int]:
            if n <= 0 or max_n <= 0:
                return (238, 238, 238)
            t = n / max_n
            # 浅黄 (255, 245, 180) → 深红 (200, 40, 40)
            r = int(255 + (200 - 255) * t)
            g = int(245 + (40 - 245) * t)
            b = int(180 + (40 - 180) * t)
            return (r, g, b)

        def draw_key(kx: int, ky: int, kw: int, kh: int, label: str, key: str) -> None:
            n = counts.get(key, 0)
            fill = heat_color(n)
            draw.rectangle(
                [(kx, ky), (kx + kw, ky + kh)],
                fill=fill, outline=(160, 160, 160), width=1,
            )
            t = (n / max_n) if max_n else 0
            label_color = (255, 255, 255) if t >= 0.55 else (40, 40, 40)
            count_color = (255, 255, 255) if t >= 0.55 else (90, 90, 90)
            lw, _, lbot = self._measure(draw, label, fonts_label)
            _render_text_with_fallback(
                draw, (kx + (kw - lw) // 2, ky + 6),
                label, fonts_label, label_color,
            )
            count_text = str(n) if n > 0 else ""
            if count_text:
                cw, _, _ = self._measure(draw, count_text, fonts_count)
                _render_text_with_fallback(
                    draw, (kx + (kw - cw) // 2, ky + kh - 16),
                    count_text, fonts_count, count_color,
                )

        for r, (indent, row) in enumerate(_QWERTY_ROWS):
            ky = y0 + r * (_KEY_SIZE + _KEY_GAP)
            kx = x0 + int(indent * (_KEY_SIZE + _KEY_GAP))
            for label, key in row:
                draw_key(kx, ky, _KEY_SIZE, _KEY_SIZE, label, key)
                kx += _KEY_SIZE + _KEY_GAP

        # 第 5 行：空格键（占据中间 6 个键宽 + 间隔）
        space_y = y0 + 4 * (_KEY_SIZE + _KEY_GAP)
        space_w = 6 * _KEY_SIZE + 5 * _KEY_GAP
        space_x = x0 + (_KEYBOARD_W - space_w) // 2
        draw_key(space_x, space_y, space_w, _KEY_SIZE, "Space", " ")

    def _make_multi_char_image(
        self,
        schema_name: str,
        word: str,
        segments: list[Segment],
        owner_id: str,
        owner_alias: str,
        max_len: int,
        select_keys: str,
        show_keyboard: bool = True,
        decorate: bool = True,
    ) -> bytes:
        PAD = 24
        CELL_GAP = 16
        fonts_stats = _load_fonts(28)
        fonts_char = _load_fonts(36)
        fonts_code = _load_fonts(28)
        fonts_code_sel = _load_fonts(30)  # 选重码略大一号，更显眼

        probe = Image.new("RGB", (1, 1))
        pdraw = ImageDraw.Draw(probe)

        def msr(text, fonts):
            return self._measure(pdraw, text, fonts)

        n = len(segments)

        def _next_self_coded(i: int) -> bool:
            # 仅自编码标点会触发前一段的首选键省略；数字/字母不省。
            return i + 1 < n and segments[i + 1].is_self_coded and _is_punct(segments[i + 1].text)

        # 每段按键数 + 显示码串
        per_presses: list[int] = []
        cell_codes: list[Optional[str]] = []
        key_seq_parts: list[str] = []
        for i, seg in enumerate(segments):
            is_last = (i == n - 1)
            if seg.is_missing:
                # 缺字显示为 6 个问号，码长按 6 计入
                per_presses.append(6 * len(seg.text))
                cell_codes.append(None)
            elif seg.is_self_coded:
                per_presses.append(1)
                cell_codes.append("＿" if seg.text == " " else seg.text)
                key_seq_parts.append(seg.text)
            else:
                code, pos = seg.code, seg.pos
                omit = _omit_in_punct_context(code, pos, max_len, select_keys) if _next_self_coded(i) else None
                if omit is not None:
                    presses, disp = omit
                    per_presses.append(presses)
                    cell_codes.append(disp)
                    key_seq_parts.append(disp)
                elif is_last:
                    per_presses.append(_key_presses_last(code, max_len, pos, select_keys, seg.candidate_count))
                    disp = _code_display_last(code, max_len, pos, select_keys, seg.candidate_count)
                    cell_codes.append(disp)
                    key_seq_parts.append(disp)
                else:
                    per_presses.append(_key_presses(code, max_len, pos, select_keys))
                    disp = _code_display(code, max_len, pos, select_keys)
                    cell_codes.append(disp)
                    key_seq_parts.append(disp)

        chars = list(word)
        missing = sum(len(seg.text) for seg in segments if seg.is_missing)
        sel_count = sum(1 for seg in segments if not seg.is_self_coded and not seg.is_missing and seg.pos > 1)
        counted_chars = sum(len(seg.text) for seg in segments)
        total_presses = sum(per_presses)
        avg_len = total_presses / counted_chars if counted_chars else 0.0
        difficulty, diff_score = _text_difficulty(chars)
        equivalence = _pair_equivalence_avg("".join(key_seq_parts))
        six_speed = 360 / avg_len if avg_len else None

        line1 = f"难度: {difficulty}({diff_score})"
        line2 = f"【{schema_name}】"
        eq_str = f"{equivalence:.6f}" if equivalence is not None else "--"
        spd_str = f"{six_speed:.2f}" if six_speed is not None else "--"
        line3 = f"来源: {self._display_owner(owner_id, owner_alias)}        字数: {len(chars)}        缺字: {missing}"
        line4 = f"码长: {avg_len:.6f}        选重: {sel_count}        当量: {eq_str}"
        line5 = f"六击速度: {spd_str}"

        _, sgh, sbot = msr("难度: A", fonts_stats)
        _, char_gh, char_bot = msr("我", fonts_char)
        _, code_gh, code_bot = msr("abc", fonts_code)
        _, _, code_sel_bot = msr("abc", fonts_code_sel)
        # 行高用更大的那一个，避免选重码下沿被裁
        code_row_bot = max(code_bot, code_sel_bot)

        # Per-cell 宽度：max(段文字宽度, 段编码宽度) + CELL_GAP
        # 选重段的编码用 fonts_code_sel 测量（更宽一点）
        cell_widths = []
        for i, seg in enumerate(segments):
            cw, _, _ = msr(seg.text, fonts_char)
            cs = cell_codes[i]
            if cs is None:
                codew, _, _ = msr("??????", fonts_code)
            else:
                is_sel_cell = (
                    not seg.is_self_coded and not seg.is_missing and seg.pos > 1
                )
                codew, _, _ = msr(cs, fonts_code_sel if is_sel_cell else fonts_code)
            cell_widths.append(max(cw, codew) + CELL_GAP)

        stats_w = max(msr(l, fonts_stats)[0] for l in [line1, line2, line3, line4, line5])
        MAX_IMG_W = 1200
        content_w = sum(cell_widths)
        STATS_KB_GAP = 24
        top_w = (stats_w + STATS_KB_GAP + _KEYBOARD_W) if show_keyboard else stats_w
        IMG_W = max(min(PAD * 2 + content_w, MAX_IMG_W), PAD * 2 + top_w, 400)
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

        LINE_H = sbot + 6
        STATS_H = LINE_H * 5
        UL_GAP = 4
        CODE_GAP = 5
        ROW_H = char_bot + UL_GAP + 1 + CODE_GAP + code_row_bot
        ROW_GAP = 12
        # 顶部带键盘时，留下 stats 和键盘里更高的那个
        top_block_h = max(STATS_H, _KEYBOARD_H) if show_keyboard else STATS_H
        IMG_H = (
            PAD + top_block_h + 16
            + ROW_H * len(rows) + ROW_GAP * (len(rows) - 1)
            + PAD
        )

        img = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 顶部 stats 在左、键盘热力图在右上
        stats_y = PAD + (top_block_h - STATS_H) // 2 if show_keyboard else PAD
        y = stats_y
        for line in [line1, line2, line3, line4, line5]:
            _render_text_with_fallback(draw, (PAD, y), line, fonts_stats, (50, 50, 50))
            y += LINE_H

        if show_keyboard:
            kb_x = IMG_W - PAD - _KEYBOARD_W
            kb_y = PAD + (top_block_h - _KEYBOARD_H) // 2
            counts = _build_key_counts("".join(key_seq_parts))
            self._draw_keyboard_heatmap(draw, (kb_x, kb_y), counts)

        grid_top = PAD + top_block_h + 16
        for r, row in enumerate(rows):
            char_y = grid_top + r * (ROW_H + ROW_GAP)
            ul_y = char_y + char_bot + UL_GAP
            code_y = ul_y + 1 + CODE_GAP

            x = PAD
            for i in row:
                seg = segments[i]
                cw, _, _ = msr(seg.text, fonts_char)
                cell_w = cell_widths[i] - CELL_GAP

                code_str = cell_codes[i]
                if code_str is None:
                    code_str = "??????"
                    is_select = False
                    is_missing = True
                    is_self = False
                    is_phrase = False
                else:
                    is_missing = False
                    is_self = seg.is_self_coded
                    is_select = (not is_self) and seg.pos > 1
                    is_phrase = (not is_self) and len(seg.text) > 1
                code_fonts = fonts_code_sel if is_select else fonts_code
                code_stroke = 1 if is_select else 0
                codew, _, _ = msr(code_str, code_fonts)

                if is_missing:
                    char_color = (180, 60, 220)
                    code_color = (180, 60, 220)
                elif is_self:
                    char_color = (80, 110, 160)
                    code_color = (80, 110, 160)
                elif is_phrase:
                    # 词组按「不含末位选键的纯编码长度」分色：1/2/3 鲜绿/橙/蓝，≥4 灰
                    code_len = _pure_code_len(seg.code or "", select_keys)
                    if code_len <= 1:
                        char_color = (0, 200, 80)
                    elif code_len == 2:
                        char_color = (255, 140, 0)
                    elif code_len == 3:
                        char_color = (30, 120, 230)
                    else:
                        char_color = (130, 130, 130)
                    code_color = (200, 40, 40) if is_select else char_color
                elif is_select:
                    char_color = (200, 40, 40)
                    code_color = (200, 40, 40)
                else:
                    char_color = (30, 30, 30)
                    code_color = (10, 10, 10)

                char_x = x + (cell_w - cw) // 2
                _render_text_with_fallback(draw, (char_x, char_y), seg.text, fonts_char, char_color)

                draw.line([(x, ul_y), (x + cell_w, ul_y)], fill=(140, 140, 140), width=1)

                code_x = x + (cell_w - codew) // 2
                _render_text_with_fallback(
                    draw, (code_x, code_y), code_str, code_fonts, code_color,
                    stroke_width=code_stroke,
                )

                x += cell_widths[i]

        probe.close()
        return self._to_png_bytes(img, decorate=decorate)

    # ── all 组：聚合多个方案的查询结果 ──────────────────────────────────────

    @staticmethod
    def _all_title_text(word: str, race_header: Optional[str], n_schemas: int) -> str:
        """all 组标题：赛文显示首行+字数+难度，普通文本只显示字数+难度。
        title 不再回显查询原文，避免冗长。"""
        chars = list(word)
        difficulty, diff_score = _text_difficulty(chars)
        info = f"字数: {len(chars)}        难度: {difficulty}({diff_score})        （{n_schemas} 个方案）"
        if race_header:
            return f"{race_header}\n{info}"
        return info

    def _make_all_detail_image(self, word: str, schema_names: list[str], force_single: bool = False, race_header: Optional[str] = None) -> bytes:
        """≤10 字符：为 all 组每个方案生成完整打法图，竖向拼接。"""
        items: list[tuple[str, dict, bytes]] = []
        for name in schema_names:
            info = self._schema_info(name)
            if not info:
                continue
            segments = self._query_segments(name, word, info["max_len"], info["select_keys"], force_single=force_single)
            stats = self._compute_stats(word, segments, info["max_len"], info["select_keys"])
            png = self._make_image(name, word, info["owner_id"], info["owner_alias"], info["max_len"], info["select_keys"], force_single=force_single, show_keyboard=False, decorate=False)
            items.append((name, stats, png))

        if not items:
            return self._make_all_empty_image(word)

        items.sort(key=lambda t: self._all_sort_key(t[1]))

        sub_imgs: list[Image.Image] = []
        try:
            for _, _, b in items:
                sub_imgs.append(Image.open(BytesIO(b)).convert("RGB"))

            PAD = 24
            TITLE_GAP = 16
            fonts_title = _load_fonts(20)
            probe = Image.new("RGB", (1, 1))
            pdraw = ImageDraw.Draw(probe)
            title = self._all_title_text(word, race_header, len(items))
            title_lines = title.split("\n")
            line_widths: list[int] = []
            line_bots: list[int] = []
            for ln in title_lines:
                lw, _, lbot = self._measure(pdraw, ln, fonts_title)
                line_widths.append(lw)
                line_bots.append(lbot)
            LINE_GAP = 4
            title_h = sum(line_bots) + LINE_GAP * (len(title_lines) - 1)
            title_w = max(line_widths) if line_widths else 0

            GAP = 16
            body_w = max(im.width for im in sub_imgs)
            IMG_W = max(body_w, PAD * 2 + title_w)
            body_h = sum(im.height for im in sub_imgs) + GAP * (len(sub_imgs) - 1)
            IMG_H = PAD + title_h + TITLE_GAP + body_h

            canvas = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            ty = PAD
            for ln, lbot in zip(title_lines, line_bots):
                _render_text_with_fallback(draw, (PAD, ty), ln, fonts_title, (30, 30, 30))
                ty += lbot + LINE_GAP
            y = PAD + title_h + TITLE_GAP
            for im in sub_imgs:
                canvas.paste(im, (0, y))
                y += im.height + GAP

            result = self._to_png_bytes(canvas)
            probe.close()
            canvas.close()
            return result
        finally:
            for im in sub_imgs:
                im.close()

    def _make_all_summary_image(self, word: str, schema_names: list[str], force_single: bool = False, race_header: Optional[str] = None) -> bytes:
        """>10 字符：每个方案一行，仅展示 来源 / 码长 / 选重 / 缺字 / 当量。"""
        rows_data: list[dict] = []
        for name in schema_names:
            info = self._schema_info(name)
            if not info:
                continue
            segments = self._query_segments(name, word, info["max_len"], info["select_keys"], force_single=force_single)
            stats = self._compute_stats(word, segments, info["max_len"], info["select_keys"])
            rows_data.append({
                "name": name,
                "owner_id": info["owner_id"],
                "owner_alias": info["owner_alias"],
                **stats,
            })

        if not rows_data:
            return self._make_all_empty_image(word)

        rows_data.sort(key=lambda d: self._all_sort_key(d))

        PAD = 24
        ROW_GAP = 10
        fonts_title = _load_fonts(20)
        fonts_head = _load_fonts(16)
        fonts_cell = _load_fonts(16)

        probe = Image.new("RGB", (1, 1))
        pdraw = ImageDraw.Draw(probe)

        def msr(text, fonts):
            return self._measure(pdraw, text, fonts)

        title = self._all_title_text(word, race_header, len(rows_data))
        title_lines = title.split("\n")
        line_widths: list[int] = []
        line_bots: list[int] = []
        for ln in title_lines:
            lw, _, lbot = msr(ln, fonts_title)
            line_widths.append(lw)
            line_bots.append(lbot)
        TITLE_LINE_GAP = 4
        title_h = sum(line_bots) + TITLE_LINE_GAP * (len(title_lines) - 1)
        tw = max(line_widths) if line_widths else 0

        headers = ["方案", "来源", "码长", "选重", "缺字", "当量", "六击速度"]

        def _row_cells(d: dict) -> list[str]:
            eq = d["equivalence"]
            spd = d["six_speed"]
            return [
                d["name"],
                self._display_owner(d["owner_id"], d.get("owner_alias", "")),
                f"{d['avg_len']:.6f}",
                str(d["sel_count"]),
                str(d["missing"]),
                f"{eq:.6f}" if eq is not None else "--",
                f"{spd:.2f}" if spd is not None else "--",
            ]

        all_rows_text = [headers] + [_row_cells(d) for d in rows_data]
        n_cols = len(headers)
        col_widths = [0] * n_cols
        for row in all_rows_text:
            for i, cell in enumerate(row):
                fonts = fonts_head if row is headers else fonts_cell
                w, _, _ = msr(cell, fonts)
                if w > col_widths[i]:
                    col_widths[i] = w

        COL_GAP = 24
        _, hgh, hbot = msr("方", fonts_head)
        _, cgh, cbot = msr("方", fonts_cell)
        ROW_H = max(hbot, cbot) + ROW_GAP

        IMG_W = PAD * 2 + sum(col_widths) + COL_GAP * (n_cols - 1)
        IMG_W = max(IMG_W, PAD * 2 + tw, 400)
        IMG_H = PAD + title_h + 16 + ROW_H * len(all_rows_text) + PAD

        img = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        ty = PAD
        for ln, lbot in zip(title_lines, line_bots):
            _render_text_with_fallback(draw, (PAD, ty), ln, fonts_title, (30, 30, 30))
            ty += lbot + TITLE_LINE_GAP

        y = PAD + title_h + 16
        # 表头
        x = PAD
        for i, cell in enumerate(headers):
            _render_text_with_fallback(draw, (x, y), cell, fonts_head, (60, 60, 60))
            x += col_widths[i] + COL_GAP
        y += ROW_H
        # 表头分割线
        draw.line([(PAD, y - ROW_GAP // 2), (IMG_W - PAD, y - ROW_GAP // 2)],
                  fill=(180, 180, 180), width=1)

        # 数据行
        for d in rows_data:
            cells = _row_cells(d)
            x = PAD
            color = (180, 60, 220) if d["missing"] > 0 else (40, 40, 40)
            for i, cell in enumerate(cells):
                _render_text_with_fallback(draw, (x, y), cell, fonts_cell, color)
                x += col_widths[i] + COL_GAP
            y += ROW_H

        probe.close()
        return self._to_png_bytes(img)

    def _make_all_empty_image(self, word: str) -> bytes:
        """all 组未配置或方案均不存在时的占位图。"""
        PAD = 24
        fonts = _load_fonts(18)
        probe = Image.new("RGB", (1, 1))
        pdraw = ImageDraw.Draw(probe)
        text = "all 组未配置任何有效方案，请管理员在插件配置中设置 all_schemas。"
        w, _, bot = self._measure(pdraw, text, fonts)
        IMG_W = max(PAD * 2 + w, 400)
        IMG_H = PAD * 2 + bot
        img = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        _render_text_with_fallback(draw, (PAD, PAD), text, fonts, (180, 60, 60))
        probe.close()
        return self._to_png_bytes(img)

    def _make_reverse_image(
        self,
        schema_name: str,
        codes: list[str],
        owner_id: str,
        owner_alias: str,
        select_keys: str,
        decorate: bool = True,
    ) -> bytes:
        """编码反查图：列出每个 code 在该方案下对应的字词，按 rowid 顺序给出位序。
        位序 1（首选）默认色，位序 ≥2（选重）用红色与正常查询一致。"""
        PAD = 24
        LINE_GAP = 6
        BLOCK_GAP = 18
        fonts_title = _load_fonts(20)
        fonts_code = _load_fonts(26)
        fonts_word = _load_fonts(22)
        fonts_info = _load_fonts(16)

        probe = Image.new("RGB", (1, 1))
        pdraw = ImageDraw.Draw(probe)

        def msr(text, fonts):
            return self._measure(pdraw, text, fonts)

        # 去重保序
        seen: set[str] = set()
        uniq_codes: list[str] = []
        for c in codes:
            if c and c not in seen:
                uniq_codes.append(c)
                seen.add(c)

        lookup = self._lookup_by_code(schema_name, uniq_codes)

        title = f"【{schema_name}】反查　来源: {self._display_owner(owner_id, owner_alias)}"
        _, _, tbot = msr(title, fonts_title)

        DEFAULT_COLOR = (40, 40, 40)
        SELECT_COLOR = (200, 40, 40)
        HEAD_COLOR = (30, 30, 30)
        MISS_COLOR = (180, 60, 220)
        INFO_COLOR = (120, 120, 120)
        SEP = "　"

        # 估算图宽：取 title / 各 code 头 / 各 word 行的最大宽度
        MAX_IMG_W = 1200
        widest = msr(title, fonts_title)[0]
        for c in uniq_codes:
            words = lookup.get(c, [])
            head_text = f"{c}　共 {len(words)} 个" if words else f"{c}　（未命中）"
            widest = max(widest, msr(head_text, fonts_code)[0])
            if words:
                # 不强求单行容纳全部字词；只保证最长单个 piece 能放下
                for i, w in enumerate(words, 1):
                    pw, _, _ = msr(f"{i}.{w}", fonts_word)
                    widest = max(widest, pw + 16)
        IMG_W = max(min(MAX_IMG_W, PAD * 2 + widest), 400)
        avail_w = IMG_W - PAD * 2 - 8

        # 排版：把每个 code 块切成若干行，行内是 (text, color) 段
        Row = list[tuple[str, tuple[int, int, int]]]
        blocks: list[tuple[str, str, tuple[int, int, int], list[Row]]] = []
        for c in uniq_codes:
            words = lookup.get(c, [])
            if not words:
                head_suffix, suffix_color = "（未命中）", MISS_COLOR
                rows: list[Row] = []
            else:
                head_suffix, suffix_color = f"共 {len(words)} 个", INFO_COLOR
                rows = []
                cur: Row = []
                cur_w = 0
                sep_w = msr(SEP, fonts_word)[0]
                for i, w in enumerate(words, 1):
                    color = SELECT_COLOR if i > 1 else DEFAULT_COLOR
                    piece = f"{i}.{w}"
                    pw, _, _ = msr(piece, fonts_word)
                    add_w = (sep_w if cur else 0) + pw
                    if cur and cur_w + add_w > avail_w:
                        rows.append(cur)
                        cur = [(piece, color)]
                        cur_w = pw
                    else:
                        if cur:
                            cur.append((SEP, DEFAULT_COLOR))
                            cur_w += sep_w
                        cur.append((piece, color))
                        cur_w += pw
                if cur:
                    rows.append(cur)
            blocks.append((c, head_suffix, suffix_color, rows))

        _, _, code_bot = msr("a", fonts_code)
        _, _, word_bot = msr("我", fonts_word)
        _, _, info_bot = msr("a", fonts_info)

        IMG_H = PAD + tbot + 16
        for _c, _suf, _sc, rows in blocks:
            IMG_H += code_bot + 4
            for _ in rows:
                IMG_H += word_bot + LINE_GAP
            IMG_H += BLOCK_GAP
        if blocks:
            IMG_H -= BLOCK_GAP
        IMG_H += PAD

        img = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        _render_text_with_fallback(draw, (PAD, PAD), title, fonts_title, HEAD_COLOR)
        y = PAD + tbot + 16

        for c, suffix, suffix_color, rows in blocks:
            _render_text_with_fallback(draw, (PAD, y), c, fonts_code, HEAD_COLOR)
            hw, _, _ = msr(c, fonts_code)
            _render_text_with_fallback(
                draw, (PAD + hw + 12, y + (code_bot - info_bot)),
                suffix, fonts_info, suffix_color,
            )
            y += code_bot + 4
            for row in rows:
                x = PAD + 16
                for piece, color in row:
                    _render_text_with_fallback(draw, (x, y), piece, fonts_word, color)
                    pw, _, _ = msr(piece, fonts_word)
                    x += pw
                y += word_bot + LINE_GAP
            y += BLOCK_GAP

        probe.close()
        return self._to_png_bytes(img, decorate=decorate)

    def _make_reverse_all_image(
        self, codes: list[str], schema_names: list[str]
    ) -> bytes:
        """all 组反查：为每个方案生成反查图，竖向拼接。"""
        sub_imgs: list[Image.Image] = []
        try:
            for name in schema_names:
                info = self._schema_info(name)
                if not info:
                    continue
                png = self._make_reverse_image(
                    name, codes, info["owner_id"], info["owner_alias"], info["select_keys"], decorate=False,
                )
                sub_imgs.append(Image.open(BytesIO(png)).convert("RGB"))

            if not sub_imgs:
                return self._make_all_empty_image(" ".join(codes))

            GAP = 16
            IMG_W = max(im.width for im in sub_imgs)
            IMG_H = sum(im.height for im in sub_imgs) + GAP * (len(sub_imgs) - 1)
            canvas = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
            y = 0
            for im in sub_imgs:
                canvas.paste(im, (0, y))
                y += im.height + GAP
            result = self._to_png_bytes(canvas)
            canvas.close()
            return result
        finally:
            for im in sub_imgs:
                im.close()

    def _make_schemas_list_image(self, schemas: list[tuple[str, str, str]]) -> bytes:
        """渲染所有词提名图片：首行统计，正文为「词提名（来源）、…」自动换行。"""
        PAD = 24
        fonts_title = _load_fonts(20)
        fonts_body = _load_fonts(18)

        probe = Image.new("RGB", (1, 1))
        pdraw = ImageDraw.Draw(probe)

        def msr(text, fonts):
            return self._measure(pdraw, text, fonts)

        owner_count = len({owner for _, owner, _ in schemas})
        title = f"共 {len(schemas)} 个词提，来自 {owner_count} 个来源"

        items = [f"{name}（{self._display_owner(owner, alias)}）" for name, owner, alias in schemas]
        sep = "、"

        MAX_IMG_W = 1200
        tw, _, tbot = msr(title, fonts_title)
        IMG_W = max(min(PAD * 2 + max((msr(it + sep, fonts_body)[0] for it in items), default=0), MAX_IMG_W),
                    PAD * 2 + tw, 400)
        avail_w = IMG_W - PAD * 2

        # 贪心断行：把 items 用 sep 拼接，按可用宽度切行
        lines: list[str] = []
        cur = ""
        cur_w = 0
        for i, it in enumerate(items):
            piece = it if i == len(items) - 1 else it + sep
            pw, _, _ = msr(piece, fonts_body)
            if cur and cur_w + pw > avail_w:
                lines.append(cur)
                cur = piece
                cur_w = pw
            else:
                cur += piece
                cur_w += pw
        if cur:
            lines.append(cur)
        if not lines:
            lines = ["（暂无词提）"]

        _, _, body_bot = msr("方", fonts_body)
        LINE_H = body_bot + 8
        IMG_H = PAD + tbot + 16 + LINE_H * len(lines) + PAD

        img = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        _render_text_with_fallback(draw, (PAD, PAD), title, fonts_title, (30, 30, 30))
        y = PAD + tbot + 16
        for line in lines:
            _render_text_with_fallback(draw, (PAD, y), line, fonts_body, (40, 40, 40))
            y += LINE_H

        probe.close()
        return self._to_png_bytes(img)

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
                "khkh上传词提 [词提名]\n\n"
                "示例：khkh上传词提 五笔86\n\n"
                "首次上传后，可单独设置以下参数（不会被后续重新上传重置）：\n"
                "  khkh设置词提 [词提名] 选重键=_;'4567890\n"
                "  khkh设置词提 [词提名] 最大长度=4\n"
                "  khkh设置词提 [词提名] 标点引导键=/"
            )
        )
        event.stop_event()

    # ── 上传词提 ────────────────────────────────────────────────────────────

    @filter.command("上传词提")
    async def cmd_upload(self, event: AstrMessageEvent):
        """
        用法：引用回复码表文件消息，发送
          khkh上传词提 [词提名]
        """
        args_str = event.message_str.strip()
        if not args_str:
            yield event.plain_result("用法：khkh上传词提 [词提名]")
            return

        # 解析词提名（第一个 token），event.message_str 包含指令前缀，跳过它
        tokens = args_str.split()
        if tokens and tokens[0] == "上传词提":
            tokens = tokens[1:]
        if not tokens:
            yield event.plain_result("用法：khkh上传词提 [词提名]")
            return
        if len(tokens) > 1:
            yield event.plain_result(
                "上传词提仅接受词提名一个参数。\n"
                "请使用「khkh设置词提 [词提名] 选重键=…/最大长度=…/标点引导键=…」单独设置。"
            )
            return
        schema_name = tokens[0]
        if len(schema_name) <= 1:
            yield event.plain_result("词提名不能为单个字符，请使用至少两个字符的名称。")
            return

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
        if not event.is_admin():
            existing_owner = self._schema_owner(schema_name)
            if existing_owner is not None and existing_owner != user_id:
                yield event.plain_result(f"词提「{schema_name}」已存在且不属于你，无法覆盖。")
                return
            try:
                limit = int(self.config.get("member_max_schemas", 3))
            except (TypeError, ValueError):
                limit = 3
            if limit > 0 and existing_owner is None and self._count_user_schemas(user_id) >= limit:
                yield event.plain_result(
                    f"你已上传 {limit} 个词提，达到上限。请先使用「删除词提 [词提名]」删除已有词提后再上传新的。"
                )
                return

        count = self._import_schema(schema_name, user_id, entries)
        yield event.plain_result(
            f"词提「{schema_name}」导入成功！共 {count:,} 条编码。\n"
            f"查询：{schema_name} <字词>\n"
            f"信息：%{schema_name}"
        )

    # ── 查码：[词提名] <字词>，或引用回复+[词提名] ─────────────────────────

    @staticmethod
    def _extract_reply_text(event: AstrMessageEvent) -> tuple[str, Optional[str]]:
        """从引用回复链中抽取所有纯文本片段，去掉空白字符。
        返回 (body, race_header)：body 为查询用纯文本，race_header 仅当回复符合赛文
        格式（末段以五个"-"开头）时为赛文首行信息，否则为 None。"""
        chain = event.get_messages()
        reply_seg: Optional[Reply] = next(
            (seg for seg in chain if seg.type == ComponentType.Reply), None
        )
        if not reply_seg or not getattr(reply_seg, "chain", None):
            return "", None
        parts: list[str] = []
        for seg in reply_seg.chain:
            txt = getattr(seg, "text", None)
            if isinstance(txt, str) and txt:
                parts.append(txt)
        raw = "".join(parts)

        paragraphs = [p for p in re.split(r"\r?\n", raw) if p.strip()]
        race_header: Optional[str] = None
        if len(paragraphs) >= 3 and paragraphs[-1].lstrip().startswith("-----"):
            race_header = paragraphs[0].strip()
            paragraphs = paragraphs[1:-1]
            raw = "\n".join(paragraphs)

        return re.sub(r"[^\S\n]+", "", raw).replace("\n", " "), race_header

    @filter.regex(r"^(?:@\S+\s+)*(\S+)(?:\s+(.+))?$")
    async def cmd_query(self, event: AstrMessageEvent):
        """
        用法：查询方案打法，发送
          [词提名] <字词>或文本
        """
        text = event.message_str.strip()
        # 引用回复时，QQ 客户端会在消息开头自动追加 `@对方` 提及，剥离后再做查询匹配
        text = re.sub(r"^(?:@\S+\s+)+", "", text)
        m = re.match(r"^(\S+)(?:\s+(.+))?$", text)
        if not m:
            return
        head, rest = m.group(1), m.group(2)
        if head.startswith("%"):
            return

        # 快速过滤：head 必须是已知词提名或 all 触发词，否则不处理
        if not (self._is_all_trigger(head) or self._schema_exists(head)):
            # 检查是否为多词提对比或反查模式
            if not head.startswith("/"):
                return

        prefix_pat = (self.config.get("single_char_prefix", "") or "").strip()

        def _strip_force_prefix(token: str) -> tuple[str, bool]:
            """剥离强制单字引导键前缀，返回 (词提名, 是否带前缀)。
            仅当剥离后是已知词提名 / all 触发词时才认为前缀生效。"""
            if not prefix_pat:
                return token, False
            try:
                pm = re.match(prefix_pat, token)
            except re.error:
                return token, False
            if not pm or pm.end() <= 0 or pm.end() >= len(token):
                return token, False
            stripped = token[pm.end():]
            if self._is_all_trigger(stripped) or self._schema_exists(stripped):
                return stripped, True
            return token, False

        schema_name, force_single = _strip_force_prefix(head)

        rest_tokens = rest.split() if rest else []

        # 编码反查：/[词提名] <code1> <code2> …，或 /[all触发词] <code1> <code2> …
        # head 以 `/` 开头即视为反查触发；剥离 `/` 后再走 force_single 前缀剥离与词提名校验。
        # 注意：`/` 也是 AstrBot 的指令前缀，凡是以 `/` 开头的消息一律不再走通用查询，
        # 让 filter.command 注册的指令（如 /上传词提、/删除词提）自行处理，避免抢答。
        if head.startswith("/"):
            if len(head) > 1:
                rev_name, _ = _strip_force_prefix(head[1:])
                is_all = self._is_all_trigger(rev_name)
                if (is_all or self._schema_exists(rev_name)) and rest_tokens:
                    try:
                        if is_all:
                            schema_names = self._all_group_schemas()
                            img_bytes = self._make_reverse_all_image(rest_tokens, schema_names)
                        else:
                            info = self._schema_info(rev_name)
                            img_bytes = self._make_reverse_image(
                                rev_name, rest_tokens, info["owner_id"], info["owner_alias"], info["select_keys"]
                            )
                    except Exception as e:
                        logger.exception(f"[im_schemas] 生成反查图片失败: {e}")
                        yield event.plain_result(
                            f"反查「{' '.join(rest_tokens)}」时渲染图片失败。"
                        )
                        return
                    yield event.chain_result([AstrImage.fromBytes(img_bytes)])
                    return
            return

        # 多词提对比：head 与 rest_tokens 全部为已知词提名（或 all 触发词，允许各自带 ! 前缀），
        # 且消息引用了文本，视作一次性 all 组对比，仅渲染用户列出的这几套词提。
        # 任一 token 带 ! 前缀，都对整组生效（force_single）。
        rest_stripped: list[tuple[str, bool]] = [_strip_force_prefix(t) for t in rest_tokens]
        head_is_known = self._is_all_trigger(schema_name) or self._schema_exists(schema_name)
        rest_all_known = all(
            self._is_all_trigger(n) or self._schema_exists(n) for n, _ in rest_stripped
        )
        if rest_tokens and head_is_known and rest_all_known:
            reply_text, reply_race_header = self._extract_reply_text(event)
            if reply_text:
                multi_force_single = force_single or any(f for _, f in rest_stripped)
                names: list[str] = []
                seen: set[str] = set()
                for n in [schema_name, *(n for n, _ in rest_stripped)]:
                    expanded = self._all_group_schemas() if self._is_all_trigger(n) else [n]
                    for en in expanded:
                        if en not in seen:
                            names.append(en)
                            seen.add(en)
                if not names:
                    return
                try:
                    if len(reply_text) > 10:
                        img_bytes = self._make_all_summary_image(
                            reply_text, names,
                            force_single=multi_force_single,
                            race_header=reply_race_header,
                        )
                    else:
                        img_bytes = self._make_all_detail_image(
                            reply_text, names,
                            force_single=multi_force_single,
                            race_header=reply_race_header,
                        )
                except Exception as e:
                    logger.exception(f"[im_schemas] 生成多词提对比图片失败: {e}")
                    yield event.plain_result(
                        f"查询「{reply_text}」时渲染多词提对比图片失败。"
                    )
                    return
                yield event.chain_result([AstrImage.fromBytes(img_bytes)])
                return

        # 未进入多词提对比，回到 [词提名] <字词> 形态：rest 整段作查询字词（允许含空格）
        word = rest if rest else None

        # all 组：聚合多个方案的结果
        if self._is_all_trigger(schema_name):
            race_header: Optional[str] = None
            if not word:
                word, race_header = self._extract_reply_text(event)
                if not word:
                    return
            schema_names = self._all_group_schemas()
            try:
                if len(word) > 10:
                    img_bytes = self._make_all_summary_image(
                        word, schema_names,
                        force_single=force_single,
                        race_header=race_header,
                    )
                else:
                    img_bytes = self._make_all_detail_image(
                        word, schema_names,
                        force_single=force_single,
                        race_header=race_header,
                    )
            except Exception as e:
                logger.exception(f"[im_schemas] 生成 all 组查询图片失败: {e}")
                yield event.plain_result(
                    f"查询「{word}」时渲染 all 组图片失败。"
                )
                return
            yield event.chain_result([AstrImage.fromBytes(img_bytes)])
            return

        info = self._schema_info(schema_name)
        if not info:
            return

        if not word:
            word, _ = self._extract_reply_text(event)
            if not word:
                return

        try:
            img_bytes = self._make_image(schema_name, word, info["owner_id"], info["owner_alias"], info["max_len"], info["select_keys"], force_single=force_single)
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
        """
        用法：查询方案信息，发送
          %[词提名]
        """
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

    # ── 所有词提：列出全部词提名与来源 ─────────────────────────────────────

    @filter.command("所有词提")
    async def cmd_list_all(self, event: AstrMessageEvent):
        """
        用法：查询服务器中所有的词提，发送
          khkh所有词提
        """
        schemas = self._list_all_schemas()
        try:
            img_bytes = self._make_schemas_list_image(schemas)
        except Exception as e:
            logger.exception(f"[im_schemas] 生成所有词提图片失败: {e}")
            yield event.plain_result("生成词提列表图片失败。")
            return
        yield event.chain_result([AstrImage.fromBytes(img_bytes)])

    # ── 设置词提 ────────────────────────────────────────────────────────────

    @filter.command("设置词提")
    async def cmd_settings(self, event: AstrMessageEvent):
        """
        用法：khkh设置词提 [词提名] 选重键=...
              khkh设置词提 [词提名] 最大长度=N
              khkh设置词提 [词提名] 标点引导键=...
              khkh设置词提 [词提名] 来源=别名

        每次仅修改一项，未设置的项保持原值。
        """
        usage = (
            "用法：khkh设置词提 [词提名] 选重键=...\n"
            "      khkh设置词提 [词提名] 最大长度=N\n"
            "      khkh设置词提 [词提名] 标点引导键=...\n"
            "      khkh设置词提 [词提名] 来源=别名"
        )
        args_str = event.message_str.strip()
        if args_str.startswith("设置词提"):
            args_str = args_str[len("设置词提"):].strip()
        if not args_str:
            yield event.plain_result(usage)
            return

        tokens = args_str.split(maxsplit=1)
        if len(tokens) < 2:
            yield event.plain_result(usage)
            return
        schema_name, kv = tokens[0], tokens[1].strip()

        if "=" not in kv:
            yield event.plain_result(usage)
            return
        key, value = kv.split("=", 1)
        key = key.strip()

        owner = self._schema_owner(schema_name)
        if owner is None:
            yield event.plain_result(f"词提「{schema_name}」不存在。")
            return
        user_id = event.get_sender_id()
        if owner != user_id and not event.is_admin():
            yield event.plain_result(f"只有词提「{schema_name}」的上传者才能修改它。")
            return

        if key == "选重键":
            if not value:
                yield event.plain_result("选重键不能为空。")
                return
            self._update_schema_setting(schema_name, "select_keys", value)
            yield event.plain_result(f"词提「{schema_name}」选重键已更新为：{value}")
        elif key == "最大长度":
            try:
                max_len = int(value)
            except ValueError:
                yield event.plain_result("最大长度必须为整数。")
                return
            self._update_schema_setting(schema_name, "max_len", max_len)
            yield event.plain_result(f"词提「{schema_name}」最大长度已更新为：{max_len}")
        elif key == "标点引导键":
            self._update_schema_setting(schema_name, "punct_key", value)
            shown = value if value else "（无）"
            yield event.plain_result(f"词提「{schema_name}」标点引导键已更新为：{shown}")
        elif key == "来源":
            self._update_schema_setting(schema_name, "来源", value)
            shown = value if value else "（无）"
            yield event.plain_result(f"词提「{schema_name}」来源别名已更新为：{shown}")
        else:
            yield event.plain_result(
                f"未知设置项「{key}」。\n{usage}"
            )

    # ── 删除词提 ────────────────────────────────────────────────────────────

    @filter.command("删除词提")
    async def cmd_delete(self, event: AstrMessageEvent):
        """
        用法：删除某个词提码表，发送
          删除词提 [词提名]
        """
        schema_name = event.message_str.strip()
        if schema_name.startswith("删除词提"):
            schema_name = schema_name[len("删除词提"):].strip()
        if not schema_name:
            yield event.plain_result("用法：khkh删除词提 [词提名]")
            return
        owner = self._schema_owner(schema_name)
        if owner is None:
            yield event.plain_result(f"词提「{schema_name}」不存在。")
            return
        user_id = event.get_sender_id()
        if owner != user_id and not event.is_admin():
            yield event.plain_result(f"只有词提「{schema_name}」的上传者才能删除它。")
            return
        self._delete_schema(schema_name)
        yield event.plain_result(f"词提「{schema_name}」已删除。")
