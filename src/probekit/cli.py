"""Command-line interface for the ProbeKit pipeline."""

import typer

app = typer.Typer(no_args_is_help=True, help="Train and deploy GPT-2 probes.")


@app.callback()
def main() -> None:
    """Run an individual pipeline stage."""
