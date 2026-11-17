"""Rebuild the bundled Chinese font from a locally downloaded Noto Sans SC TTF.

Run with --source pointing to the upstream font. Retains existing bundled
characters and adds characters from both web interfaces, overwriting only the
bundled font. Download the source from google/fonts/ofl/notosanssc; its OFL
license is distributed beside the output. No network requests are made here.
"""

import argparse
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    console = root / "src/core/app/console/static"
    output = console / "fonts/eva-task-cjk.ttf"
    with TTFont(output) as previous:
        characters = set(previous.getBestCmap())

    # Include interface labels without depending on the host's installed fonts
    for directory in (console, root / "tools/datasets"):
        for path in directory.rglob("*"):
            if path.suffix in {".html", ".js", ".css"}:
                characters.update(map(ord, path.read_text()))
    with TTFont(args.source) as font:
        subsetter = subset.Subsetter()
        subsetter.populate(unicodes=characters)
        subsetter.subset(font)
        font.save(output)
    print(f"Updated {output}: {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
