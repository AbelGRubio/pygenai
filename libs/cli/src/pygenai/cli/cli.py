"""PyGenAI command-line interface launcher."""

import typer

from .commands.front import app as front_app
from .launcher import run_cli

app = typer.Typer(help="PyGenAI CLI Launcher", add_completion=True, invoke_without_command=True)


@app.callback()
def callback_command(ctx: typer.Context) -> None:
    """Callback function for the CLI."""
    if ctx.invoked_subcommand is None:
        typer.echo("Welcome to PyGenAI CLI Launcher! Use --help for available commands.")
        typer.echo(ctx.get_help())


app.command()(callback_command)
app.command("run")(run_cli)
app.add_typer(front_app, name="front", help="Commands related to the PyGenAI front-end.")
