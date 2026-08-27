"""TUI командного слоя: выбор проектов и кандидатов для публикации.

Запуск: python -m realmemory.team ui --path ./rm_data
Слева — карточки проектов, справа — кандидаты выбранного проекта с превью.
Публикация требует явного подтверждения модальным экраном со списком
конкретных текстов; never-правила блокируют строки на месте (замок).
"""
from __future__ import annotations

import time
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Static

from ..store.sqlite_store import MemoryStore
from . import registry
from .identity import resolve_identity
from .policy import ELIGIBLE, TeamPolicy, load_policy, save_policy, set_project_shareable
from .view import ProjectView, find_project, projects_view

_GLYPH_PICKED = "☑"
_GLYPH_OPEN = "◻"
_PREVIEW_LIMIT = 400


def _store_for(root):
    """Открыть базу без загрузки эмбеддера (геометрия из db_meta, как хуки)."""
    from ..config import MemoryConfig
    from ..hook_cli import _infer_dim, _load_cfg

    cfg = _load_cfg(root) or MemoryConfig(dim=_infer_dim(root))
    return MemoryStore(root / "memory.db", dim=cfg.dim)


class ConfirmPublish(ModalScreen[bool]):
    """Подтверждение: ровно то, что уйдёт в команду, списком текстов."""

    CSS: ClassVar[str] = """
    ConfirmPublish { align: center middle; }
    #box { width: 88; height: auto; max-height: 78%;
           border: thick $accent; background: $surface; padding: 1 2; }
    #hint { color: $text-muted; margin-top: 1; }
    """
    BINDINGS: ClassVar[list] = [
        ("enter", "confirm", "опубликовать"),
        ("escape", "cancel", "отмена"),
    ]

    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self._lines = lines

    def compose(self) -> ComposeResult:
        yield Static("\n".join(self._lines), id="box", markup=False)
        yield Static("Enter — опубликовать · Esc — отмена", id="hint")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class TeamApp(App):
    TITLE = "realMemory · team sharing"
    CSS: ClassVar[str] = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #left { width: 48; border-right: solid $panel; padding: 0 1; }
    #projects { height: auto; max-height: 55%; }
    #stats { padding-top: 1; color: $text-muted; }
    #candidates { height: 52%; }
    #preview { height: 1fr; border-top: solid $panel; padding: 0 1; }
    """
    BINDINGS: ClassVar[list] = [
        Binding("tab", "focus_next", "панель", show=True),
        Binding("space", "toggle_pick", "выбрать", show=True),
        Binding("s", "toggle_shareable", "шарить вкл/выкл", show=True),
        Binding("p", "publish_picked", "опубликовать", show=True),
        Binding("u", "retract_row", "отозвать", show=True),
        Binding("r", "reload", "обновить", show=False),
        Binding("q", "quit", "выход", show=True),
    ]

    def __init__(self, root, policy_path=None) -> None:
        super().__init__()
        self._root = root
        self._policy_path = policy_path
        self.policy: TeamPolicy = load_policy(policy_path)
        if not self.policy.identity:
            self.policy.identity = resolve_identity()
        self.store = _store_for(root)
        self.views: list[ProjectView] = []
        self.current: str | None = None
        self.picked: set[int] = set()
        self.row_owner: dict[str, int] = {}       # строка таблицы -> trace_id
        self.picked_rows: dict[int, object] = {}  # trace_id -> row_key

    # -- компоновка ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield DataTable(id="projects", cursor_type="row",
                                zebra_stripes=True)
                yield Static("", id="stats", markup=True)
            with Vertical():
                yield DataTable(id="candidates", cursor_type="row",
                                zebra_stripes=True)
                yield Static("(нет выбора)", id="preview")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#projects", DataTable).add_columns(
            "", "проект", "кандид.", "публ.")
        self.query_one("#candidates", DataTable).add_columns(
            "", "#", "kind", "реп.", "факт")
        self.reload()

    # -- данные -------------------------------------------------------------

    def reload(self) -> None:
        self.views = projects_view(self.store, self.policy)
        ptable = self.query_one("#projects", DataTable)
        ptable.clear()
        for v in self.views:
            mark = "✓" if self.policy.is_shareable_project(v.name) else "·"
            ptable.add_row(mark, v.name, f"{v.eligible}/{len(v.records)}",
                           str(v.published), key=v.name)
        target = self.current or (self.views[0].name if self.views else None)
        self.show_project(target)

    def show_project(self, name: str | None) -> None:
        view = find_project(self.views, name) if name else None
        self.current = name
        table = self.query_one("#candidates", DataTable)
        table.clear()
        self.row_owner.clear()
        self.picked.clear()
        self.picked_rows.clear()
        if view is None:
            self.query_one("#stats", Static).update("")
            self._set_preview("")
            return
        for rec, status, reason in view.records:
            if status == ELIGIBLE:
                glyph = _GLYPH_OPEN
            else:
                short = reason.replace("\n", " ")[:16]
                glyph = f"🔒{short}"
            key = table.add_row(glyph, str(rec.id), rec.kind,
                                str(rec.reinforced_count),
                                (rec.text or "")[:64], key=str(rec.id))
            self.row_owner[str(key.value)] = int(rec.id or 0)
        self.query_one("#stats", Static).update(
            f"▸ [b]{view.name}[/b] · активных {view.total} · "
            f"кандидатов {view.eligible} · never-block {view.blocked} · "
            f"в команде {view.published}")

    def _set_preview(self, text: str) -> None:
        if len(text) > _PREVIEW_LIMIT:
            text = text[:_PREVIEW_LIMIT] + "…"
        self.query_one("#preview", Static).update(text or "(нет выбора)")

    def _view_records(self):
        view = find_project(self.views, self.current or "")
        return view.records if view else []

    # -- события ------------------------------------------------------------

    def on_data_table_row_highlighted(self, event) -> None:
        if event.data_table.id != "candidates":
            return
        tid = self.row_owner.get(str(getattr(event.row_key, "value",
                                             event.row_key)))
        rec = self.store.get(tid) if tid is not None else None
        if rec is None:
            self._set_preview("")
            return
        meta_line = ", ".join(f"{k}={v!r}" for k, v in (rec.meta or {}).items())
        author = rec.author or "(без автора)"
        self._set_preview(f"#{tid} [{rec.scope}] автор: {author}\n"
                          f"{rec.text}\n{meta_line}")

    # -- действия -----------------------------------------------------------

    def action_toggle_pick(self) -> None:
        table = self.query_one("#candidates", DataTable)
        if not table.rows:
            return
        rk = list(table.rows)[min(table.cursor_row, len(table.rows) - 1)]
        kstr = str(getattr(rk, "value", rk))
        tid = self.row_owner.get(kstr)
        if tid is None:
            return
        record_status = next(((s, r) for r_, s, r in self._view_records()
                              if int(r_.id or -1) == tid), ("", ""))
        if record_status[0] != ELIGIBLE:
            self.notify(f"заблокировано: {record_status[1]}",
                        severity="warning")
            return
        col0 = next(iter(table.columns))
        if tid in self.picked:
            self.picked.discard(tid)
            self.picked_rows.pop(tid, None)
            table.update_cell(rk, col0, _GLYPH_OPEN)
        else:
            self.picked.add(tid)
            self.picked_rows[tid] = rk
            table.update_cell(rk, col0, _GLYPH_PICKED)

    def action_toggle_shareable(self) -> None:
        if not self.current:
            return
        want = not self.policy.is_shareable_project(self.current)
        if set_project_shareable(self.policy, self.current, want):
            save_policy(self.policy, self._policy_path)
            self.notify(f"shareable[{self.current}] = {want}")
        else:
            self.notify("политика не изменилась", severity="warning")
        self.reload()

    def action_publish_picked(self) -> None:
        ids = sorted(self.picked)
        if not ids:
            self.notify("ничего не выбрано (space на кандидате)",
                        severity="warning")
            return
        lines = ["Публикация в команду — уйдут следующие записи:", ""]
        for tid in ids[:8]:
            rec = self.store.get(tid)
            lines.append(f"• #{tid} {(rec.text if rec else '?')[:66]}")
        if len(ids) > 8:
            lines.append(f"…и ещё {len(ids) - 8}")
        lines.append(f"\nитого записей: {len(ids)}")

        def do_it(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                rows = registry.publish(self.store, ids, policy=self.policy,
                                        now=time.time(),
                                        identity=resolve_identity())
                self.notify(f"опубликовано локально: {len(rows)} "
                            "(сетевая синхронизация появится позже)")
                self.reload()
            except Exception as exc:  # noqa: BLE001 - показать оператору
                self.notify(str(exc), severity="error")

        self.push_screen(ConfirmPublish(lines), do_it)

    def action_retract_row(self) -> None:
        table = self.query_one("#candidates", DataTable)
        if not table.rows:
            return
        rk = list(table.rows)[min(table.cursor_row, len(table.rows) - 1)]
        tid = self.row_owner.get(str(getattr(rk, "value", rk)))
        if tid is None:
            return
        n = registry.retract(self.store, now=time.time(), trace_ids=[tid])
        self.notify(f"отозвано публикаций: {n}")
        self.reload()


def run_app(root, policy_path=None) -> None:  # pragma: no cover - интерактив
    TeamApp(root, policy_path=policy_path).run()
