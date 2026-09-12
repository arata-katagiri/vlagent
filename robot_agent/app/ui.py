"""Terminal rendering. Kept out of main.py because the REPL and the demo
script both render the same things."""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from ..actions.schema import ActionCall, Safety

console = Console()

BADGE = {
    Safety.SAFE: ("safe", "bold green"),
    Safety.CAUTION: ("caution", "bold yellow"),
    Safety.IRREVERSIBLE: ("irreversible", "bold red"),
}


def badge(level: Safety) -> Text:
    label, style = BADGE[level]
    return Text(label, style=style)


def banner(backend: str, model: str) -> None:
    console.print(
        Panel(
            Text.from_markup(
                "[bold]Language-to-action robot arm[/bold]\n"
                "Type a request in plain English. The plan is shown before anything moves.\n"
                f"backend [cyan]{backend}[/cyan]   planner [cyan]{model}[/cyan]   "
                "[dim]Ctrl-D or 'quit' to exit[/dim]"
            ),
            border_style="cyan",
        )
    )


def scene(state: dict) -> None:
    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("object")
    table.add_column("position (m)")
    table.add_column("notes")
    for name, obj in state["objects"].items():
        notes = []
        if obj["fragile"]:
            notes.append("[red]fragile[/red]")
        if not obj["graspable"]:
            notes.append("[dim]fixed[/dim]")
        if obj["on_top_of"]:
            notes.append(f"on {obj['on_top_of']}")
        if obj["supporting"]:
            notes.append("holds " + ", ".join(obj["supporting"]))
        if obj["near_table_edge"]:
            notes.append("[yellow]near edge[/yellow]")
        if obj["held"]:
            notes.append("[cyan]held[/cyan]")
        p = obj["position_m"]
        table.add_row(name, f"{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}", "  ".join(notes))
    holding = state["gripper"]["holding"] or "nothing"
    console.print(
        Panel(
            Group(table, Text.from_markup(f"\ngripper: holding [cyan]{holding}[/cyan]")),
            title="scene",
            border_style="dim",
        )
    )


def plan(steps: list[ActionCall], levels: list[tuple[Safety, str]], summary: str) -> None:
    table = Table(show_header=True, header_style="dim", border_style="dim")
    table.add_column("#", width=2, justify="right")
    table.add_column("action")
    table.add_column("arguments")
    table.add_column("why")
    table.add_column("safety")
    for i, (step, (level, _)) in enumerate(zip(steps, levels), start=1):
        args = ", ".join(f"{k}={v}" for k, v in step.args.items())
        table.add_row(str(i), step.name, args, step.rationale, badge(level))
    console.print(Panel(table, title=summary or "proposed plan", border_style="cyan"))


def rejected(steps: list[ActionCall], problems: list[str]) -> None:
    body = Text()
    body.append("proposed: ", style="dim")
    body.append(" -> ".join(s.name for s in steps) + "\n")
    for p in problems:
        body.append("refused: ", style="bold red")
        body.append(p + "\n")
    console.print(Panel(body, title="plan rejected", border_style="red"))


def confirm(step_number: int, step: ActionCall, reason: str) -> bool:
    """Ask for a per-step yes. The default is no, and it is never reused."""
    console.print(
        Panel(
            Text.from_markup(
                f"[bold red]Step {step_number} is irreversible.[/bold red]\n\n"
                f"[bold]{step.name}[/bold]  "
                + ", ".join(f"{k}={v}" for k, v in step.args.items())
                + f"\n\n{reason}"
            ),
            border_style="red",
            title="confirmation required",
        )
    )
    try:
        answer = Prompt.ask("  Type [bold]yes[/bold] to allow this one step", default="no")
    except (EOFError, KeyboardInterrupt):
        # No input available (piped, or the user bailed out). The default is no,
        # and an irreversible step must never proceed on an unanswered prompt.
        console.print()
        return False
    return answer.strip().lower() == "yes"


def step_result(i: int, step: ActionCall, ok: bool, reason: str) -> None:
    mark = "[green]OK[/green]" if ok else "[red]FAILED[/red]"
    console.print(f"  {i}. [bold]{step.name}[/bold] {mark} [dim]{reason}[/dim]")


def question(text: str) -> None:
    console.print(Panel(Text(text), title="the agent needs a clarification", border_style="yellow"))


def answer(text: str) -> None:
    console.print(Panel(Text(text), title="answer", border_style="cyan"))


def report(text: str, ok: bool = True) -> None:
    console.print(Panel(Text(text), border_style="green" if ok else "red", title="result"))


def note(text: str) -> None:
    console.print(f"[dim]{text}[/dim]")


__all__ = ["console", "banner", "scene", "plan", "rejected", "confirm",
           "step_result", "question", "answer", "report", "note", "badge"]
