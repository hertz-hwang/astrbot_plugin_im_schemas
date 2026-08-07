#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""astrbot_plugin_im_schemas 数据库管理工具。

提供对插件 SQLite 数据库 (schemas.db) 的交互式增删改查能力，并支持从本地
TSV 文件导入词提（码表）或词频（字词→频率）。

工作模式
--------
1. 子命令模式：``python tools/db_admin.py <子命令> [参数]``，一次执行返回结果。
2. 交互 shell：``python tools/db_admin.py shell`` 进入 REPL，连续操作同一会话。

子命令一览
----------
list    {schemas|freqs}                      列出全部词提 / 词频
show    schema <name>                        查看词提详情（含码表统计 / 前后若干条）
show    freq  <name>                         查看词频详情（含频率样例）
import  schema <name> <file> [--owner ID]    从本地 .txt 导入码表（覆盖式）
import  freq  <name> <file> [--owner ID]    从本地 .txt 导入词频
insert  code <schema> <code> <word>         单条新增 codes 条目
update  schema <name> [--set k=v ...]       修改 select_keys / max_len / punct_key / 来源 / PUA
update  freq  <name> --alias <别名>         修改 owner_alias
delete  schema <name> [-y]                  删除词提（含全部 codes）
delete  freq  <name>  [-y]                  删除词频（含全部 freq_entries）
search  code <schema> [--word 字] [--code 码] [--limit N]
                                              在码表中按字 / 码搜索
shell                                      进入交互式 REPL
stats                                      打印数据库全局统计

通用参数
--------
--db PATH      指定 SQLite 文件路径；缺省时按以下顺序自动寻找：
                  1) $ASTRBOT_DATA_DIR/plugin_data/im_data/schemas.db
                  2) <本文件>/../../../plugin_data/im_data/schemas.db
              （即 ``<AstrBot data>/plugin_data/im_data/schemas.db``）
--quiet        仅打印最终结果，抑制说明文字
--dry-run      所有写操作仅打印待执行的 SQL，不真正落库
"""
from __future__ import annotations

import argparse
import os
import shlex
import sqlite3
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Optional

# ── 常量（与 main.py 保持一致）─────────────────────────────────────────────

DEFAULT_SELECT_KEYS = "_;'4567890"
DEFAULT_MAX_LEN = 4

# 安全起见：所有可能的码表配置字段。CLI update 子命令的 --set 接受其中之一。
SCHEMA_SETTABLE = {
    "select_keys": "选重键（首位自动省略，第 2 位起依次取此串字符）",
    "max_len": "编码最大长度（整数）",
    "punct_key": "标点引导键（备用字段）",
    "owner_alias": "来源别名（展示用）",
    "custom_font": "PUA 自定义字体（fonts/ 目录下的文件名）",
}

# ── 数据库路径定位 ─────────────────────────────────────────────────────────

def _here() -> Path:
    return Path(__file__).resolve().parent


def _discover_db_path() -> Path:
    """自动发现 schemas.db 路径：优先环境变量，其次约定路径。"""
    env = os.environ.get("ASTRBOT_DATA_DIR")
    if env:
        return Path(env) / "plugin_data" / "im_data" / "schemas.db"
    # tools/db_admin.py → tools/ → <plugin>/ → plugins/ → data/
    candidate = _here().parent.parent.parent / "plugin_data" / "im_data" / "schemas.db"
    return candidate


# ── 数据库封装 ────────────────────────────────────────────────────────────

class DB:
    """SQLite 连接封装：负责表结构初始化、上下文管理与干跑支持。"""

    def __init__(self, path: Path, dry_run: bool = False):
        self.path = path
        self.dry_run = dry_run
        self._conn: Optional[sqlite3.Connection] = None

    # —— 生命周期 ——

    def __enter__(self) -> "DB":
        if self.dry_run:
            # dry-run：仍然需要打开一个连接以便 SELECT 与 prepare；
            # 但所有写操作会被拦截，不会真正落库。
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema(self._conn)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            try:
                if exc_type is None and not self.dry_run:
                    self._conn.commit()
                else:
                    self._conn.rollback()
            finally:
                self._conn.close()
                self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("DB not opened; use 'with DB(...) as db:'")
        return self._conn

    # —— 写操作：自动适配 dry-run ——

    def execute(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        if self.dry_run:
            print(f"[dry-run] SQL: {sql.strip()}  |  params={tuple(params)}")
            # 返回一个空游标；调用方通常只看 rowcount / fetchall
            return _DryRunCursor()
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq: Iterable[Iterable]) -> sqlite3.Cursor:
        rows = list(seq)
        if self.dry_run:
            print(f"[dry-run] SQL(many x{len(rows)}): {sql.strip()}")
            for i, r in enumerate(rows[:3]):
                print(f"        sample[{i}] = {tuple(r)}")
            if len(rows) > 3:
                print(f"        ... ({len(rows) - 3} more)")
            return _DryRunCursor()
        return self.conn.executemany(sql, rows)

    # —— 表结构初始化（与 main.py._init_db 保持同步） ——

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schemas (
                name         TEXT PRIMARY KEY,
                owner_id     TEXT NOT NULL,
                owner_alias  TEXT NOT NULL DEFAULT '',
                select_keys  TEXT NOT NULL DEFAULT '',
                max_len      INTEGER NOT NULL DEFAULT 4,
                punct_key    TEXT NOT NULL DEFAULT '',
                custom_font  TEXT NOT NULL DEFAULT ''
            )
        """)
        # 兼容旧库：可能缺 owner_alias / custom_font 列
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(schemas)").fetchall()
        }
        if "owner_alias" not in cols:
            conn.execute(
                "ALTER TABLE schemas ADD COLUMN owner_alias TEXT NOT NULL DEFAULT ''"
            )
        if "custom_font" not in cols:
            conn.execute(
                "ALTER TABLE schemas ADD COLUMN custom_font TEXT NOT NULL DEFAULT ''"
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_codes_by_code "
            "ON codes(schema_name, code)"
        )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS freqs (
                name        TEXT PRIMARY KEY,
                owner_id    TEXT NOT NULL,
                owner_alias TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS freq_entries (
                freq_name TEXT NOT NULL,
                word      TEXT NOT NULL,
                freq      TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_freq_entries "
            "ON freq_entries(freq_name, word)"
        )
        conn.commit()


class _DryRunCursor:
    """模拟 sqlite3.Cursor 的最小子集，仅供 dry-run 时链式调用不报错。"""

    @property
    def rowcount(self) -> int:
        return 0

    @property
    def lastrowid(self) -> int:
        return 0

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def close(self):
        pass


# ── 文件解析（与 main.py 保持一致）────────────────────────────────────────

def _read_text_smart(path: Path) -> str:
    """按 utf-8-sig / utf-8 / gbk / gb18030 顺序尝试解码。"""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "all", raw, 0, len(raw), "无法以 utf-8/gbk 解码，请转换文件编码后重试"
    )


def parse_schema_tsv(content: str) -> list[tuple[str, str]]:
    """解析码表 TSV：每行 ``code<TAB>word`` 或 ``code<TAB>w1<TAB>w2...``。

    返回 ``[(code, word), ...]``。空行 / ``#`` 注释行忽略。
    """
    entries: list[tuple[str, str]] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code = parts[0].strip()
        if not code:
            continue
        for i in range(1, len(parts)):
            word = parts[i].strip()
            if word:
                entries.append((code, word))
    return entries


def parse_freq_tsv(content: str) -> list[tuple[str, str]]:
    """解析词频 TSV：每行 ``word<TAB>freq``。注意列序与码表相反。

    重复字词保留首次。
    """
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        word = parts[0].strip()
        freq = parts[1].strip()
        if not word or not freq or word in seen:
            continue
        seen.add(word)
        entries.append((word, freq))
    return entries


# ── 输出辅助 ─────────────────────────────────────────────────────────────

def _hr(char: str = "─", width: int = 60) -> str:
    return char * width


def _print_table(rows: list[sqlite3.Row], columns: list[str]) -> None:
    """简易表格打印：列宽自适应（按显示宽度估算）。"""
    if not rows:
        print("（无数据）")
        return
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], _display_width(str(row[c] or "")))
    line = "  ".join(f"{c:<{widths[c]}}" for c in columns)
    print(line)
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(f"{_display_width(str(row[c] or '')) and str(row[c] or ''):<{widths[c]}}" for c in columns))


def _display_width(s: str) -> int:
    """粗略估算 CJK 显示宽度（每个 CJK 算 2 个 ASCII 列宽）。"""
    w = 0
    for ch in s:
        if ord(ch) > 0x2E80:  # 汉字及之后都按 2 算
            w += 2
        else:
            w += 1
    return w


# ── 业务命令 ────────────────────────────────────────────────────────────

def cmd_list(target: str, db: DB, quiet: bool) -> int:
    if target == "schemas":
        rows = db.execute(
            "SELECT name, owner_id, owner_alias, max_len, "
            "(SELECT COUNT(*) FROM codes WHERE schema_name = schemas.name) AS cnt "
            "FROM schemas ORDER BY name"
        ).fetchall()
        if not quiet:
            print(_hr())
            print(f"  词提列表（共 {len(rows)} 个）")
            print(_hr())
        _print_table(rows, ["name", "owner_id", "owner_alias", "max_len", "cnt"])
    elif target == "freqs":
        rows = db.execute(
            "SELECT name, owner_id, owner_alias, "
            "(SELECT COUNT(*) FROM freq_entries WHERE freq_name = freqs.name) AS cnt "
            "FROM freqs ORDER BY name"
        ).fetchall()
        if not quiet:
            print(_hr())
            print(f"  词频列表（共 {len(rows)} 个）")
            print(_hr())
        _print_table(rows, ["name", "owner_id", "owner_alias", "cnt"])
    else:
        print(f"未知类型: {target}（应为 schemas 或 freqs）", file=sys.stderr)
        return 2
    return 0


def cmd_show(kind: str, name: str, db: DB, quiet: bool, sample: int = 5) -> int:
    if kind == "schema":
        row = db.execute(
            "SELECT * FROM schemas WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            print(f"词提「{name}」不存在。", file=sys.stderr)
            return 1
        count = db.execute(
            "SELECT COUNT(*) FROM codes WHERE schema_name = ?", (name,)
        ).fetchone()[0]
        if not quiet:
            print(_hr())
            print(f"  词提：{row['name']}")
            print(_hr())
        for k in ("owner_id", "owner_alias", "select_keys", "max_len",
                  "punct_key", "custom_font"):
            print(f"  {k:<14} = {row[k]}")
        print(f"  {'codes 数':<14} = {count}")
        # 取前 N 条作为样例
        sample_rows = db.execute(
            "SELECT code, word FROM codes WHERE schema_name = ? "
            "ORDER BY code, word LIMIT ?", (name, sample)
        ).fetchall()
        if sample_rows:
            print()
            print(f"  样例（前 {len(sample_rows)} 条）：")
            _print_table(sample_rows, ["code", "word"])
    elif kind == "freq":
        row = db.execute(
            "SELECT * FROM freqs WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            print(f"词频「{name}」不存在。", file=sys.stderr)
            return 1
        count = db.execute(
            "SELECT COUNT(*) FROM freq_entries WHERE freq_name = ?", (name,)
        ).fetchone()[0]
        if not quiet:
            print(_hr())
            print(f"  词频：{row['name']}")
            print(_hr())
        for k in ("owner_id", "owner_alias"):
            print(f"  {k:<14} = {row[k]}")
        print(f"  {'entries 数':<14} = {count}")
        sample_rows = db.execute(
            "SELECT word, freq FROM freq_entries WHERE freq_name = ? "
            "ORDER BY rowid LIMIT ?", (name, sample)
        ).fetchall()
        if sample_rows:
            print()
            print(f"  样例（前 {len(sample_rows)} 条）：")
            _print_table(sample_rows, ["word", "freq"])
    else:
        print(f"未知 kind: {kind}", file=sys.stderr)
        return 2
    return 0


def cmd_import(
    kind: str, name: str, file: Path, owner: str, db: DB, quiet: bool
) -> int:
    """覆盖式导入：删除旧条目，写入新条目。"""
    if not file.exists():
        print(f"文件不存在: {file}", file=sys.stderr)
        return 1
    if file.stat().st_size == 0:
        print(f"文件为空: {file}", file=sys.stderr)
        return 1
    if not owner.strip():
        print("owner 不能为空（用 --owner 指定）", file=sys.stderr)
        return 1

    try:
        content = _read_text_smart(file)
    except UnicodeDecodeError as e:
        print(f"文件编码无法识别: {e}", file=sys.stderr)
        return 1

    if kind == "schema":
        entries = parse_schema_tsv(content)
        if not entries:
            print("码表文件未解析到任何条目（检查是否 TSV 格式）。", file=sys.stderr)
            return 1
        if not quiet:
            print(f"  解析到 {len(entries):,} 条码表条目，准备覆盖写入...")
        existed = db.execute(
            "SELECT 1 FROM schemas WHERE name = ?", (name,)
        ).fetchone() is not None
        db.execute(
            """
            INSERT INTO schemas(name, owner_id, select_keys, max_len, punct_key, custom_font)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET owner_id = excluded.owner_id
            """,
            (name, owner, DEFAULT_SELECT_KEYS, DEFAULT_MAX_LEN, "", ""),
        )
        db.execute("DELETE FROM codes WHERE schema_name = ?", (name,))
        db.executemany(
            "INSERT INTO codes(schema_name, code, word) VALUES (?, ?, ?)",
            [(name, code, word) for code, word in entries],
        )
        if not quiet:
            action = "覆盖更新" if existed else "新建"
            print(f"  ✓ {action}词提「{name}」成功，共 {len(entries):,} 条 codes。")
    elif kind == "freq":
        entries = parse_freq_tsv(content)
        if not entries:
            print("词频文件未解析到任何条目（检查是否 TSV 格式）。", file=sys.stderr)
            return 1
        if not quiet:
            print(f"  解析到 {len(entries):,} 条词频条目，准备覆盖写入...")
        existed = db.execute(
            "SELECT 1 FROM freqs WHERE name = ?", (name,)
        ).fetchone() is not None
        db.execute(
            """
            INSERT INTO freqs(name, owner_id, owner_alias)
            VALUES (?, ?, '')
            ON CONFLICT(name) DO UPDATE SET owner_id = excluded.owner_id
            """,
            (name, owner),
        )
        db.execute("DELETE FROM freq_entries WHERE freq_name = ?", (name,))
        db.executemany(
            "INSERT INTO freq_entries(freq_name, word, freq) VALUES (?, ?, ?)",
            [(name, word, freq) for word, freq in entries],
        )
        if not quiet:
            action = "覆盖更新" if existed else "新建"
            print(f"  ✓ {action}词频「{name}」成功，共 {len(entries):,} 条 entries。")
    else:
        print(f"未知 kind: {kind}", file=sys.stderr)
        return 2
    return 0


def cmd_insert_code(
    schema: str, code: str, word: str, db: DB, quiet: bool
) -> int:
    if not db.execute(
        "SELECT 1 FROM schemas WHERE name = ?", (schema,)
    ).fetchone():
        print(f"词提「{schema}」不存在。请先用 import 创建。", file=sys.stderr)
        return 1
    db.execute(
        "INSERT INTO codes(schema_name, code, word) VALUES (?, ?, ?)",
        (schema, code, word),
    )
    if not quiet:
        print(f"  ✓ 已添加 codes: {schema} | {code} | {word}")
    return 0


def cmd_update_schema(
    name: str, sets: list[str], db: DB, quiet: bool
) -> int:
    if not db.execute(
        "SELECT 1 FROM schemas WHERE name = ?", (name,)
    ).fetchone():
        print(f"词提「{name}」不存在。", file=sys.stderr)
        return 1
    if not sets:
        print("--set k=v 至少给一项。可写字段: " + ", ".join(SCHEMA_SETTABLE),
              file=sys.stderr)
        return 2
    for kv in sets:
        if "=" not in kv:
            print(f"参数格式错误: {kv}（应为 k=v）", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in SCHEMA_SETTABLE:
            print(f"不可写字段: {k}。可选: {list(SCHEMA_SETTABLE)}", file=sys.stderr)
            return 2
        if k == "max_len":
            try:
                v = str(int(v))
            except ValueError:
                print("max_len 必须是整数", file=sys.stderr)
                return 2
        db.execute(
            f"UPDATE schemas SET {k} = ? WHERE name = ?", (v, name)
        )
        if not quiet:
            print(f"  ✓ {name}.{k} = {v}  ({SCHEMA_SETTABLE[k]})")
    return 0


def cmd_update_freq(name: str, alias: Optional[str], db: DB, quiet: bool) -> int:
    if not db.execute(
        "SELECT 1 FROM freqs WHERE name = ?", (name,)
    ).fetchone():
        print(f"词频「{name}」不存在。", file=sys.stderr)
        return 1
    if alias is None:
        print("请通过 --alias <别名> 指定新别名。", file=sys.stderr)
        return 2
    db.execute(
        "UPDATE freqs SET owner_alias = ? WHERE name = ?", (alias, name)
    )
    if not quiet:
        print(f"  ✓ {name}.owner_alias = {alias}")
    return 0


def cmd_delete(kind: str, name: str, db: DB, yes: bool, quiet: bool) -> int:
    if kind == "schema":
        row = db.execute(
            "SELECT * FROM schemas WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            print(f"词提「{name}」不存在。", file=sys.stderr)
            return 1
        cnt = db.execute(
            "SELECT COUNT(*) FROM codes WHERE schema_name = ?", (name,)
        ).fetchone()[0]
        if not yes:
            print(f"将删除词提「{name}」及其 {cnt} 条 codes。继续？[y/N] ", end="", flush=True)
            try:
                ans = input().strip().lower()
            except EOFError:
                # 非交互环境（管道 / 重定向）默认取消
                print("\n（无 stdin，默认取消；用 -y 跳过确认）")
                return 0
            if ans != "y":
                print("已取消。")
                return 0
        db.execute("DELETE FROM codes WHERE schema_name = ?", (name,))
        db.execute("DELETE FROM schemas WHERE name = ?", (name,))
        if not quiet:
            print(f"  ✓ 已删除词提「{name}」及 {cnt} 条 codes。")
    elif kind == "freq":
        row = db.execute(
            "SELECT * FROM freqs WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            print(f"词频「{name}」不存在。", file=sys.stderr)
            return 1
        cnt = db.execute(
            "SELECT COUNT(*) FROM freq_entries WHERE freq_name = ?", (name,)
        ).fetchone()[0]
        if not yes:
            print(f"将删除词频「{name}」及其 {cnt} 条 entries。继续？[y/N] ", end="", flush=True)
            try:
                ans = input().strip().lower()
            except EOFError:
                print("\n（无 stdin，默认取消；用 -y 跳过确认）")
                return 0
            if ans != "y":
                print("已取消。")
                return 0
        db.execute("DELETE FROM freq_entries WHERE freq_name = ?", (name,))
        db.execute("DELETE FROM freqs WHERE name = ?", (name,))
        if not quiet:
            print(f"  ✓ 已删除词频「{name}」及 {cnt} 条 entries。")
    else:
        print(f"未知 kind: {kind}", file=sys.stderr)
        return 2
    return 0


def cmd_search_code(
    schema: str, word: Optional[str], code: Optional[str],
    limit: int, db: DB, quiet: bool
) -> int:
    if not db.execute(
        "SELECT 1 FROM schemas WHERE name = ?", (schema,)
    ).fetchone():
        print(f"词提「{schema}」不存在。", file=sys.stderr)
        return 1
    if not word and not code:
        print("--word / --code 至少给一项。", file=sys.stderr)
        return 2
    sql = "SELECT code, word FROM codes WHERE schema_name = ?"
    params: list = [schema]
    if word:
        sql += " AND word = ?"
        params.append(word)
    if code:
        sql += " AND code = ?"
        params.append(code)
    sql += " ORDER BY code, word LIMIT ?"
    params.append(limit)
    rows = db.execute(sql, params).fetchall()
    if not quiet:
        print(f"  命中 {len(rows)} 条（limit={limit}）：")
    _print_table(rows, ["code", "word"])
    return 0


def cmd_stats(db: DB) -> int:
    print(_hr())
    print("  数据库统计")
    print(_hr())
    n_schema = db.execute("SELECT COUNT(*) FROM schemas").fetchone()[0]
    n_code = db.execute("SELECT COUNT(*) FROM codes").fetchone()[0]
    n_freq = db.execute("SELECT COUNT(*) FROM freqs").fetchone()[0]
    n_entry = db.execute("SELECT COUNT(*) FROM freq_entries").fetchone()[0]
    print(f"  词提数           = {n_schema}")
    print(f"  codes 总条数     = {n_code:,}")
    print(f"  词频数           = {n_freq}")
    print(f"  freq_entries 总  = {n_entry:,}")
    print(f"  DB 文件          = {db.path}")
    return 0


# ── 交互式 REPL ──────────────────────────────────────────────────────────

HELP_TEXT = textwrap.dedent("""\
    可用命令（每条 :: 表示「list/show/...」；省略时按上下文补全）：

      list schemas|freqs
      show schema|freq <name>
      import schema|freq <name> <file> [--owner ID]
      insert code <schema> <code> <word>
      update schema <name> --set k=v [--set k=v ...]
      update freq  <name> --alias <别名>
      delete schema|freq <name> [-y]
      search code <schema> [--word 字] [--code 码] [--limit N]
      stats
      help
      quit / exit
    """)


class Repl:
    def __init__(self, db: DB, quiet: bool):
        self.db = db
        self.quiet = quiet
        self._stop = False

    def loop(self) -> int:
        print(_hr("═"))
        print("  astrbot_plugin_im_schemas 数据库管理工具（交互模式）")
        print(f"  DB: {self.db.path}")
        print(_hr("═"))
        print("  输入 `help` 查看命令；输入 `quit` 退出。")
        print()
        while not self._stop:
            try:
                line = input("imsdb> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self._stop = True
                break
            if not line:
                continue
            if line in ("quit", "exit", ":q"):
                self._stop = True
                break
            if line in ("help", "?"):
                print(HELP_TEXT)
                continue
            try:
                rc = self._dispatch(line)
            except SystemExit as e:
                # argparse 在错误时会 sys.exit(2)，REPL 中转为提示
                rc = e.code if isinstance(e.code, int) else 1
                print(f"（命令解析失败，rc={rc}）", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[错误] {e}", file=sys.stderr)
                rc = 1
            if rc:
                # 非 0 退出码仅展示，不终止 REPL
                pass
        print("bye.")
        return 0

    def _dispatch(self, line: str) -> int:
        """把一行文本拆成 argv，走一遍与 CLI 相同的解析路径。"""
        argv = shlex.split(line)
        # 补全主命令：直接打 `schemas` / `freqs` 当 list
        if argv and argv[0] in ("schemas", "freqs"):
            argv = ["list", *argv]
        if argv and argv[0] in ("schema", "freq") and len(argv) >= 2:
            # 形如 `schema 五笔86`：补全为 show schema ...
            if argv[1] not in ("list", "show", "import", "insert", "update",
                               "delete", "search"):
                argv = ["show", *argv]
        ns = cli_parse(argv)
        return run_cli(ns, self.db, self.quiet, in_repl=True)


# ── argparse 接线 ────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", type=Path, default=None,
                   help="SQLite 文件路径；缺省时按 $ASTRBOT_DATA_DIR 或约定路径自动发现")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="仅打印最终结果，抑制说明文字")
    p.add_argument("--dry-run", action="store_true",
                   help="所有写操作仅打印 SQL，不真正落库")


def _build_parser() -> tuple[argparse.ArgumentParser, argparse._SubParsersAction, dict]:
    """构造顶层 parser 与 subparsers，一次性注册所有子命令。

    返回 ``(parser, sub, sub_parsers)``，其中 ``sub_parsers[cmd_name]`` 是
    每个子命令对应的 ArgumentParser 实例，供 dispatch 复用。
    """
    parser = argparse.ArgumentParser(
        prog="db_admin",
        description="astrbot_plugin_im_schemas 数据库管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="提示：直接运行而不带子命令即进入交互 shell。",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<子命令>")

    # list ────────────────────────────────────────────────
    p_list = sub.add_parser("list", help="列出所有词提 / 词频")
    p_list.add_argument("target", choices=["schemas", "freqs"])

    # show ────────────────────────────────────────────────
    p_show = sub.add_parser("show", help="查看词提 / 词频详情")
    p_show.add_argument("kind", choices=["schema", "freq"])
    p_show.add_argument("name")
    p_show.add_argument("--sample", type=int, default=5,
                        help="样例条数（默认 5）")

    # import ──────────────────────────────────────────────
    p_imp = sub.add_parser("import", help="从本地 .txt 导入码表 / 词频")
    p_imp.add_argument("kind", choices=["schema", "freq"])
    p_imp.add_argument("name", help="词提 / 词频名")
    p_imp.add_argument("file", type=Path, help="TSV 文件路径")
    p_imp.add_argument("--owner", required=True, help="归属 user_id")

    # insert ──────────────────────────────────────────────
    p_ins = sub.add_parser("insert", help="插入单条 codes 条目")
    p_ins.add_argument("code", help="要插入的子命令，目前仅 code")
    p_ins.add_argument("schema")
    p_ins.add_argument("code_value", help="编码")
    p_ins.add_argument("word", help="字 / 词")

    # update schema ───────────────────────────────────────
    p_us = sub.add_parser("update", help="修改 schema / freq 字段")
    p_us.add_argument("kind", choices=["schema", "freq"])
    p_us.add_argument("name")
    p_us.add_argument("--set", dest="sets", action="append", default=[],
                      metavar="k=v", help="schema 字段赋值（可多次）")
    p_us.add_argument("--alias", help="freq 的新 owner_alias")

    # delete ──────────────────────────────────────────────
    p_del = sub.add_parser("delete", help="删除词提 / 词频（级联）")
    p_del.add_argument("kind", choices=["schema", "freq"])
    p_del.add_argument("name")
    p_del.add_argument("-y", "--yes", action="store_true",
                       help="跳过交互确认")

    # search ──────────────────────────────────────────────
    p_se = sub.add_parser("search", help="在码表中按字 / 码搜索")
    p_se.add_argument("code", help="固定子命令 code")
    p_se.add_argument("schema")
    p_se.add_argument("--word", help="匹配的字 / 词")
    p_se.add_argument("--code", dest="code_match", help="匹配的编码")
    p_se.add_argument("--limit", type=int, default=20)

    # shell ───────────────────────────────────────────────
    sub.add_parser("shell", help="进入交互式 REPL")

    # stats ───────────────────────────────────────────────
    sub.add_parser("stats", help="打印数据库全局统计")

    return parser, sub, sub.choices


# 顶层认识的「子命令」白名单（与 _build_parser 同步）
SUBCOMMAND_NAMES = {
    "list", "show", "import", "insert", "update",
    "delete", "search", "shell", "stats",
}


def _extract_subcommand(argv: list[str]) -> Optional[str]:
    """在 argv 中找出第一个「子命令名」token（忽略选项与值）。

    - ``--db PATH`` 中的 PATH 不算（因为 --db 的值不是 - 开头）
    - ``-q`` / ``--quiet`` 等布尔开关也不算
    - ``--`` 之后的所有 token 都不算（剩余位置参数留给子 parser）
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            return None
        if tok.startswith("-"):
            # 选项：若下一项不是选项形式且不带 =，则作为值消耗
            if "=" not in tok and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                # 仅对「带值」的选项消耗一项
                if tok in ("--db",):
                    i += 2
                    continue
            i += 1
            continue
        if tok in SUBCOMMAND_NAMES:
            return tok
        # 第一个非选项 token 不是子命令名：当作无子命令（让顶层报错）
        return None
    return None


def cli_parse(argv: list[str]) -> argparse.Namespace:
    """让全局参数（--db / -q / --dry-run）在子命令前/后都能用。

    思路：手动定位子命令名（不依赖 argparse 的 subparsers 派发），然后把
    ``cmd`` 之前的公共参数连同 ``cmd`` 之后的剩余参数一起交给对应子 parser
    解析。这样能避免 argparse 在 subparsers 上对全局参数的吞并冲突。
    """
    _, _, sub_parsers = _build_parser()
    cmd = _extract_subcommand(argv)
    if cmd is None or cmd not in sub_parsers:
        # 无 / 未识别子命令：交给顶层 parser 报错（会打印 usage / help）
        return _build_parser()[0].parse_args(argv if argv else ["--help"])

    # 把子命令名从 argv 中剥掉；子 parser 看到的是它自己的参数序列。
    if cmd in argv:
        argv = argv[: argv.index(cmd)] + argv[argv.index(cmd) + 1:]

    common_only = _make_common_parser()
    full = argparse.ArgumentParser(
        prog=f"db_admin {cmd}",
        # 关闭 full 自带的 -h/--help，避免与子 parser 冲突
        add_help=False,
        parents=[common_only, sub_parsers[cmd]],
    )
    ns = full.parse_args(argv)
    # 把子命令名挂回 namespace，方便后续 dispatch
    ns.cmd = cmd
    return ns


def _make_common_parser() -> argparse.ArgumentParser:
    """独立的 common parser（不挂在任何 parent 上），用于两阶段解析。"""
    p = argparse.ArgumentParser(add_help=False)
    _add_common(p)
    return p


def _open(ns: argparse.Namespace) -> DB:
    path = ns.db or _discover_db_path()
    if not ns.quiet:
        print(f"  DB: {path}{'  (dry-run)' if ns.dry_run else ''}")
    return DB(path, dry_run=ns.dry_run)


def run_cli(ns: argparse.Namespace, db: Optional[DB] = None,
            quiet: Optional[bool] = None, in_repl: bool = False) -> int:
    """执行解析后的子命令。供 CLI 入口与 REPL 复用。

    - 当 ``db`` 为 None 时：打开新连接、打印横幅、执行后关闭。
    - 当 ``db`` 不为 None 时：直接复用（REPL 场景，连接由外层管理）。
    """
    own_db = db is None
    q = ns.quiet if quiet is None else quiet
    if own_db:
        if not q:
            print(_hr())
            print("  astrbot_plugin_im_schemas 数据库管理工具")
            print(_hr())
        with _open(ns) as opened:
            return _exec_with_db(opened, ns, q, in_repl)
    return _exec_with_db(db, ns, q, in_repl)


def _exec_with_db(db: DB, ns: argparse.Namespace, q: bool, in_repl: bool) -> int:
    cmd = ns.cmd
    if cmd is None:
        # 没有任何子命令 → 进入 shell
        return Repl(db, q).loop() if not in_repl else 1
    try:
        if cmd == "list":
            return cmd_list(ns.target, db, q)
        if cmd == "show":
            return cmd_show(ns.kind, ns.name, db, q, sample=ns.sample)
        if cmd == "import":
            return cmd_import(ns.kind, ns.name, ns.file, ns.owner, db, q)
        if cmd == "insert":
            return cmd_insert_code(ns.schema, ns.code_value, ns.word, db, q)
        if cmd == "update":
            if ns.kind == "schema":
                return cmd_update_schema(ns.name, ns.sets, db, q)
            return cmd_update_freq(ns.name, ns.alias, db, q)
        if cmd == "delete":
            return cmd_delete(ns.kind, ns.name, db, ns.yes, q)
        if cmd == "search":
            code_match = getattr(ns, "code_match", None)
            return cmd_search_code(
                ns.schema, ns.word, code_match, ns.limit, db, q
            )
        if cmd == "shell":
            return Repl(db, q).loop()
        if cmd == "stats":
            return cmd_stats(db)
        print(f"未实现子命令: {cmd}", file=sys.stderr)
        return 2
    except sqlite3.IntegrityError as e:
        print(f"[IntegrityError] {e}", file=sys.stderr)
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    # 直接请求帮助时交给顶层 parser 处理（不补 shell）
    if argv and argv[0] in ("-h", "--help"):
        ns = cli_parse(argv)
        return run_cli(ns)
    # 没有任何参数 → 进入 shell
    if not argv:
        argv = ["shell"]
    else:
        # 仅在没有显式子命令时，缺省走 shell
        sub_names = {
            "list", "show", "import", "insert", "update",
            "delete", "search", "shell", "stats",
        }
        if not any(tok in sub_names for tok in argv):
            argv = ["shell", *argv]
    ns = cli_parse(argv)
    return run_cli(ns)


if __name__ == "__main__":
    sys.exit(main())
