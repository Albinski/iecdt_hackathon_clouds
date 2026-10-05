import argparse
import os
import re
import time
from pathlib import Path

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.measure import Measurement
from rich.table import Table
from rich.text import Text

ROOT = os.environ.get("IECDT_ROOT", "/gws/ssde/j25b/iecdt/modis_hackathon")
DEFAULT_PATH = os.environ.get("IECDT_LEADERBOARD", f"{ROOT}/leaderboard/leaderboard.md")

#: A column is right-aligned when every value in it looks like a number: a
#: score, optionally followed by that task's rank in brackets, or a dash for a
#: task the submission was not scored on.
NUMERIC = re.compile(r"^(?:[-+]?\d+(?:\.\d+)?(?:\s*\(\d+\))?|-|—)$")

#: Markdown escapes a literal pipe inside a cell; a team name may contain one.
UNESCAPE = re.compile(r"\\\|")


def parse_tables(text):
    """Every markdown table in `text`, as a list of rows of cells.

    The separator row is dropped, so row 0 of each table is its header.
    """
    tables, current = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if current:
                tables.append(current)
                current = []
            continue
        cells = [
            UNESCAPE.sub("|", c).strip().strip("`")
            for c in re.split(r"(?<!\\)\|", line)[1:-1]
        ]
        if all(set(c) <= set("-: ") and c for c in cells):
            continue  # the header separator
        current.append(cells)
    if current:
        tables.append(current)
    return tables


def build_table(rows, title=None):
    """One parsed markdown table as a `rich` table.

    Numeric columns are right-aligned so the decimal points line up, which is
    most of what makes a column of scores readable. The first column is left
    alone: it is a rank or a team name.
    """
    header, *body = rows
    table = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        header_style="bold",
        title_style="bold",
        title_justify="left",
        pad_edge=False,
    )
    for i, name in enumerate(header):
        numeric = i and all(NUMERIC.match(r[i]) for r in body)
        table.add_column(
            name,
            justify="right" if numeric else "left",
            style="cyan" if i == 0 else None,
            no_wrap=i == 0,
        )
    for row in body:
        # The leader is worth being able to find at a glance.
        first = Text(row[0], style="bold") if row[0] == "1" else row[0]
        table.add_row(first, *row[1:])
    return table


def build_blocks(rows):
    """The per-task table as one small table per submission.

    For a terminal too narrow for the wide form. Letting rich shrink eleven
    columns into 80 characters truncates every heading and every value to three
    characters and a dot, which is worse than useless; a block per submission
    keeps the row-per-team reading and fits anywhere. Laid out side by side
    where there is room for it.
    """
    header, *body = rows
    blocks = []
    for row in body:
        block = Table(
            title=row[0],
            box=box.SIMPLE_HEAD,
            title_style="bold cyan",
            title_justify="left",
            pad_edge=False,
        )
        block.add_column("task", style="dim")
        block.add_column("score", justify="right")
        for name, value in zip(header[1:], row[1:]):
            block.add_row(name, value)
        blocks.append(block)
    return Columns(blocks, padding=(0, 3))


def fits(console, renderable):
    """Does it render at its natural width without being squeezed?

    Measured against **unclamped** options: `console.options` already carries
    the console's width, so measuring with those reports at most that width and
    the answer is always yes.
    """
    options = console.options.update(max_width=10_000)
    return Measurement.get(console, options, renderable).maximum <= console.width


def age(path):
    """How long ago the file was written, in words."""
    seconds = max(0, time.time() - path.stat().st_mtime)
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago"


def read_tables(path):
    """The tables in the leaderboard file, or an actionable exit."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise SystemExit(
            f"{path} does not exist.\n"
            f"Either nothing has been scored yet, or the path is wrong -- the "
            f"one announced at the kickoff goes in IECDT_LEADERBOARD, or pass "
            f"it as an argument."
        )
    except PermissionError:
        raise SystemExit(f"{path} is not readable by you. Ask the organisers.")
    tables = parse_tables(text)
    if not tables:
        raise SystemExit(f"{path} holds no tables. Is it the leaderboard file?")
    return tables


def show(console, path, tables, standings_only=False):
    """Render the leaderboard to `console`."""
    n = len(tables[0]) - 1
    console.print()
    console.print(
        f"[bold]IECDT leaderboard[/bold]  [dim]{n} "
        f"submission{'' if n == 1 else 's'} · written "
        f"{age(path)}[/dim]"
    )
    console.print(build_table(tables[0]))
    if standings_only or len(tables) < 2:
        return
    wide = build_table(tables[1], title="Per task")
    if fits(console, wide):
        console.print(wide)
    else:
        console.print("[bold]Per task[/bold]")
        console.print(build_blocks(tables[1]))


def main(argv=None):
    """`argv` is taken from the command line unless a caller passes one."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_PATH,
        help="The leaderboard file (default: %(default)s)",
    )
    p.add_argument(
        "--standings",
        action="store_true",
        help="Only the standings, not the per-task detail",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Render at this width instead of the terminal's. The "
        "per-task table is wide; rich shrinks it to fit, so a "
        "larger width here is how to see it in full when "
        "piping to a file or a pager.",
    )
    args = p.parse_args(argv)

    path = Path(args.path)
    show(
        Console(width=args.width),
        path,
        read_tables(path),
        standings_only=args.standings,
    )


if __name__ == "__main__":
    main()
