"""
CLI demo for the AI Act Risk Classifier.

Usage:
  python scripts/demo.py "A system that screens CVs and ranks job candidates automatically"
  python scripts/demo.py --interactive
  python scripts/demo.py --run-examples
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from classifier import classify
from models.schemas import RiskLevel

console = Console()

RISK_COLORS = {
    RiskLevel.PROHIBITED: "bold red",
    RiskLevel.HIGH: "bold orange1",
    RiskLevel.LIMITED: "bold yellow",
    RiskLevel.MINIMAL: "bold green",
    RiskLevel.UNCLEAR: "bold white",
}

RISK_LABELS = {
    RiskLevel.PROHIBITED: "PROHIBITED",
    RiskLevel.HIGH: "HIGH RISK",
    RiskLevel.LIMITED: "LIMITED RISK",
    RiskLevel.MINIMAL: "MINIMAL RISK",
    RiskLevel.UNCLEAR: "UNCLEAR",
}

EXAMPLE_CASES = [
    "A bank uses an AI model to automatically decide whether to approve or reject mortgage loan applications based on applicant financial data and credit history.",
    "A municipality deploys cameras with real-time facial recognition in a public square to identify suspects in an ongoing criminal investigation.",
    "An e-commerce platform uses AI to recommend products to users based on their browsing history.",
    "An HR software company offers a tool that scores candidates' CVs and ranks them before a human recruiter reviews the shortlist.",
    "A mental health app uses AI to detect emotional states from users' text messages and suggest coping strategies.",
]


def print_result(result) -> None:
    color = RISK_COLORS.get(result.risk_level, "white")
    label = RISK_LABELS.get(result.risk_level, result.risk_level.value.upper())

    console.print()
    console.print(
        Panel(
            f"[{color}]{label}[/{color}]  [dim](confidence: {result.confidence})[/dim]",
            title="[bold]Classification Result[/bold]",
            expand=False,
        )
    )

    # Citations table
    if result.primary_citations:
        table = Table(title="Legal Basis", show_header=True, header_style="bold cyan")
        table.add_column("Provision", style="cyan", no_wrap=True)
        table.add_column("Relevance", style="white")
        for citation in result.primary_citations:
            table.add_row(citation.article, citation.summary)
        console.print(table)

    # Reasoning
    console.print(Panel(result.reasoning, title="Legal Reasoning", border_style="dim"))

    # Ambiguities
    if result.ambiguities:
        console.print("[bold yellow]Ambiguities requiring human review:[/bold yellow]")
        for amb in result.ambiguities:
            console.print(f"  • {amb}")

    # Disclaimer
    console.print(f"\n[dim italic]{result.disclaimer}[/dim italic]\n")


def run_interactive():
    console.print("[bold]AI Act Risk Classifier[/bold] — Interactive Mode")
    console.print("Type your AI system use case description. Enter 'quit' to exit.\n")

    while True:
        use_case = console.input("[bold cyan]Use case:[/bold cyan] ").strip()
        if use_case.lower() in ("quit", "exit", "q"):
            break
        if not use_case:
            continue

        with console.status("Classifying..."):
            result = classify(use_case)
        print_result(result)


def run_examples():
    console.print("[bold]Running example use cases...[/bold]\n")
    for i, case in enumerate(EXAMPLE_CASES, 1):
        console.print(f"[bold]Example {i}:[/bold] {case}")
        with console.status("Classifying..."):
            result = classify(case)
        print_result(result)
        console.rule()


def main():
    parser = argparse.ArgumentParser(description="AI Act Risk Classifier")
    parser.add_argument("use_case", nargs="?", help="Use case description to classify")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--run-examples", action="store_true", help="Run built-in example cases")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.run_examples:
        run_examples()
    elif args.use_case:
        with console.status("Classifying..."):
            result = classify(args.use_case)
        if args.json:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            print_result(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
