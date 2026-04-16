"""
CLI demo for the AI Act Risk Classifier.

Usage:
  python scripts/demo.py "A system that screens CVs and ranks job candidates automatically"
  python scripts/demo.py --interactive
  python scripts/demo.py --run-examples
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.rule import Rule
from rich.padding import Padding
from rich import box

from classifier import classify
from models.schemas import RiskLevel

console = Console()

# ── Risk level config ─────────────────────────────────────────────────────────

RISK_CONFIG = {
    RiskLevel.PROHIBITED: {
        "label": "PROHIBITED",
        "color": "white",
        "bg": "red",
        "icon": "X",
        "border": "red",
    },
    RiskLevel.HIGH: {
        "label": "HIGH RISK",
        "color": "white",
        "bg": "dark_orange",
        "icon": "!",
        "border": "dark_orange",
    },
    RiskLevel.LIMITED: {
        "label": "LIMITED RISK",
        "color": "black",
        "bg": "yellow3",
        "icon": "~",
        "border": "yellow3",
    },
    RiskLevel.MINIMAL: {
        "label": "MINIMAL RISK",
        "color": "white",
        "bg": "dark_green",
        "icon": "OK",
        "border": "dark_green",
    },
    RiskLevel.UNCLEAR: {
        "label": "UNCLEAR",
        "color": "white",
        "bg": "grey50",
        "icon": "?",
        "border": "grey50",
    },
}

CONFIDENCE_BARS = {
    "high":   ("[########]", "green"),
    "medium": ("[#####---]", "yellow"),
    "low":    ("[##------]", "red"),
}

EXAMPLE_CASES = [
    "A bank uses an AI model to automatically decide whether to approve or reject mortgage loan applications based on applicant financial data and credit history.",
    "A municipality deploys cameras with real-time facial recognition in a public square to identify suspects in an ongoing criminal investigation.",
    "An e-commerce platform uses AI to recommend products to users based on their browsing history.",
    "An HR software company offers a tool that scores candidates' CVs and ranks them before a human recruiter reviews the shortlist.",
    "A mental health app uses AI to detect emotional states from users' text messages and suggest coping strategies.",
]


# ── Header ────────────────────────────────────────────────────────────────────

def print_header() -> None:
    console.print()
    title = Text()
    title.append("  AI ACT  ", style="bold white on blue")
    title.append("  RISK CLASSIFIER  ", style="bold blue on white")
    console.print(Padding(title, (0, 0)))
    console.print(
        "[dim]Regulation (EU) 2024/1689  |  Rules + RAG + LLM pipeline[/dim]"
    )
    console.print()


# ── Risk banner ───────────────────────────────────────────────────────────────

def print_risk_banner(result) -> None:
    cfg = RISK_CONFIG.get(result.risk_level, RISK_CONFIG[RiskLevel.UNCLEAR])
    bar, bar_color = CONFIDENCE_BARS.get(result.confidence, ("██░░░░░░", "white"))

    label_text = Text()
    label_text.append(f"  [{cfg['icon']}]  {cfg['label']}  ", style=f"bold {cfg['color']} on {cfg['bg']}")

    conf_text = Text()
    conf_text.append(f"  Confidence  ")
    conf_text.append(bar, style=bar_color)
    conf_text.append(f"  {result.confidence.upper()}", style=f"bold {bar_color}")

    console.print(label_text)
    console.print(conf_text)
    console.print()


# ── Use case panel ────────────────────────────────────────────────────────────

def print_use_case(use_case: str) -> None:
    console.print(Panel(
        f"[italic]{use_case}[/italic]",
        title="[bold dim]USE CASE[/bold dim]",
        border_style="dim",
        padding=(0, 1),
    ))


# ── Citations table ───────────────────────────────────────────────────────────

def print_citations(result) -> None:
    if not result.primary_citations:
        return

    cfg = RISK_CONFIG.get(result.risk_level, RISK_CONFIG[RiskLevel.UNCLEAR])
    table = Table(
        box=box.SIMPLE_HEAD,
        header_style=f"bold {cfg['border']}",
        show_edge=False,
        padding=(0, 1),
    )
    table.add_column("PROVISION", style=f"bold {cfg['border']}", no_wrap=True, min_width=22)
    table.add_column("LEGAL BASIS", style="white")

    for citation in result.primary_citations:
        table.add_row(citation.article, citation.summary)

    console.print(Panel(
        table,
        title="[bold dim]LEGAL BASIS[/bold dim]",
        border_style=cfg["border"],
        padding=(0, 0),
    ))


# ── Reasoning ─────────────────────────────────────────────────────────────────

def print_reasoning(result) -> None:
    console.print(Panel(
        result.reasoning,
        title="[bold dim]LEGAL REASONING[/bold dim]",
        border_style="dim",
        padding=(0, 1),
    ))


# ── Ambiguities ───────────────────────────────────────────────────────────────

def print_ambiguities(result) -> None:
    if not result.ambiguities:
        return

    lines = Text()
    for amb in result.ambiguities:
        lines.append("  ?  ", style="bold yellow on black")
        lines.append(f"  {amb}\n", style="yellow")

    console.print(Panel(
        lines,
        title="[bold yellow]REQUIRES HUMAN REVIEW[/bold yellow]",
        border_style="yellow",
        padding=(0, 0),
    ))


# ── Disclaimer ────────────────────────────────────────────────────────────────

def print_disclaimer(result) -> None:
    console.print(
        f"\n[dim]{result.disclaimer}[/dim]\n"
    )


# ── Full result ───────────────────────────────────────────────────────────────

def print_result(result) -> None:
    print_use_case(result.use_case)
    print_risk_banner(result)
    print_citations(result)
    print_reasoning(result)
    print_ambiguities(result)
    print_disclaimer(result)


# ── Pipeline status ───────────────────────────────────────────────────────────

def classify_with_status(use_case: str):
    """Run classify() showing live pipeline steps."""
    result = None

    steps = [
        ("Rule pre-filter", 0.3),
        ("RAG retrieval (ChromaDB)", 1.0),
        ("LLM classification", None),  # None = unknown duration
    ]

    with console.status("[bold cyan]Running pipeline...[/bold cyan]") as status:
        for i, (step, _) in enumerate(steps[:-1], 1):
            status.update(f"[bold cyan]Step {i}/3:[/bold cyan] {step}")
            time.sleep(0.3)

        status.update("[bold cyan]Step 3/3:[/bold cyan] LLM classification — this may take a moment")
        result = classify(use_case)

    return result


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_interactive() -> None:
    print_header()
    console.print("[bold]Interactive Mode[/bold] — describe an AI system use case.")
    console.print("[dim]Type 'quit' to exit.[/dim]\n")

    while True:
        try:
            use_case = console.input("[bold cyan]> Use case:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if use_case.lower() in ("quit", "exit", "q", ""):
            break

        result = classify_with_status(use_case)
        print_result(result)
        console.print(Rule(style="dim"))


def run_examples() -> None:
    print_header()
    console.print(f"[bold]Running {len(EXAMPLE_CASES)} example use cases[/bold]\n")

    for i, case in enumerate(EXAMPLE_CASES, 1):
        console.print(Rule(f"[dim]Example {i} of {len(EXAMPLE_CASES)}[/dim]", style="dim"))
        result = classify_with_status(case)
        print_result(result)

    console.print(Rule("[dim]Done[/dim]", style="dim"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Act Risk Classifier — EU Regulation 2024/1689"
    )
    parser.add_argument("use_case", nargs="?", help="Use case description to classify")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--run-examples", action="store_true", help="Run built-in examples")
    parser.add_argument("--json", action="store_true", help="Output raw JSON (pipe-friendly)")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.run_examples:
        run_examples()
    elif args.use_case:
        if not args.json:
            print_header()
        result = classify_with_status(args.use_case)
        if args.json:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print_result(result)
    else:
        print_header()
        parser.print_help()


if __name__ == "__main__":
    main()
