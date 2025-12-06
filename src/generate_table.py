import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Reuse LaTeX helpers
from utils import printLatexTable, getTableHeaders

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "report" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "report" / "output"


def read_table_lines(name: str):
    fpath = TABLES_DIR / f"{name}.txt"
    if not fpath.exists():
        print(f"[WARN] Missing table data: {fpath}")
        return []
    with open(fpath, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines


def save_latex_source(name: str, table_type: str, lines: list[str]):
    # Join raw lines into a LaTeX body (they should already end with \\)
    body = "\n".join(lines)
    latex = printLatexTable(body, table_type)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = OUTPUT_DIR / f"{name}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[OK] Wrote LaTeX: {tex_path}")


def parse_lines_to_matrix(lines: list[str]):
    rows = []
    for ln in lines:
        # remove trailing LaTeX line break if present
        ln = ln.rstrip()
        if ln.endswith("\\\\"):
            ln = ln[:-2].rstrip()
        # split on ' & ' but tolerate different spacing
        parts = [p.strip() for p in ln.split("&")]
        rows.append(parts)
    return rows


def clamp_headers(headers: list[str], n_cols: int):
    if len(headers) == n_cols:
        return headers
    if n_cols < len(headers):
        return headers[:n_cols]
    # pad generic headers
    extra = [f"C{i}" for i in range(len(headers)+1, n_cols+1)]
    return headers + extra


def render_png_table(name: str, table_type: str, lines: list[str]):
    if not lines:
        return
    data = parse_lines_to_matrix(lines)
    # align column count
    n_cols = max(len(r) for r in data)
    headers = clamp_headers(getTableHeaders(table_type), n_cols)

    fig, ax = plt.subplots(figsize=(min(12, 1.5*n_cols), 1 + 0.5*len(data)))
    ax.axis('off')

    table = ax.table(cellText=data, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{name}.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Wrote PNG: {out_path}")


def main():
    # camera pose errors
    cam_lines = read_table_lines("cameraPoseErrors")
    save_latex_source("cameraPoseErrors", "cameraPoseErrors", cam_lines)
    render_png_table("cameraPoseErrors", "cameraPoseErrors", cam_lines)

    # point cloud errors
    pc_lines = read_table_lines("pointCloudErrors")
    save_latex_source("pointCloudErrors", "pointCloudErrors", pc_lines)
    render_png_table("pointCloudErrors", "pointCloudErrors", pc_lines)

    # projection errors
    proj_lines = read_table_lines("projectionErrors")
    save_latex_source("projectionErrors", "projectionErrors", proj_lines)
    render_png_table("projectionErrors", "projectionErrors", proj_lines)

    print("[DONE] Generated LaTeX sources and PNGs for tables.")


if __name__ == "__main__":
    main()
