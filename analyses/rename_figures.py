"""
Create alphabetically-labelled copies of every image referenced in figure_paths.md

Example output:
    Figure1a.tif
    Figure1b.tif
    ...
    Figure9i.tif
"""

import re
import shutil
from pathlib import Path
from string import ascii_lowercase

MD_FILE = Path("figure_paths.md")
OUT_DIR = Path("../submission")

# --- regex helpers ----------------------------------------------------------
fig_header = re.compile(r"^#\s*Figure\s+(\d+)")
img_line   = re.compile(r"!\[\]\(([^)]+\.tif)\)", re.IGNORECASE)

# --- walk through the markdown ---------------------------------------------
figure_num = None
letter_idx = 0           # position inside the current figure

for line in MD_FILE.read_text().splitlines():

    # 1) new figure section?
    if (m := fig_header.match(line)):
        figure_num = int(m.group(1))
        letter_idx = 0
        continue

    # 2) image reference inside a figure section?
    if figure_num is None:
        continue                            # haven't reached first header yet

    if (m := img_line.search(line)):
        src_path = Path("..") / m.group(1)
        if not src_path.exists():
            print(f"WARNING: source image not found → {src_path}")
            continue

        # prepare destination name
        try:
            letter = ascii_lowercase[letter_idx]
        except IndexError:
            raise ValueError(
                f"Figure {figure_num} has more than 26 panels; "
                f"extend the script to handle double letters."
            )
        dst_path = OUT_DIR / f"Figure{figure_num}{letter}.tif"

        # copy with metadata preserved
        shutil.copy2(src_path, dst_path)
        print(f"Copied {src_path}  →  {dst_path}")
        letter_idx += 1

