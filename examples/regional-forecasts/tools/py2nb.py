"""Percent-format .py -> .ipynb, then execute in place.

jupytext is not installed in the accord-chc env and the workspace rule says not
to add packages to a shared runtime env without noting it, so this does the
round-trip with nbformat + nbclient, which are already there.

usage: py2nb.py SOURCE.py TARGET.ipynb [--no-exec] [--timeout SECONDS]
"""
import sys
import re
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

CELL_RE = re.compile(r"^# %%(.*)$")


def split_cells(src):
    """Yield (kind, body) for each percent cell.

    `# %% [markdown]` -> markdown; `# %% [hide]` -> a code cell whose SOURCE is
    collapsed in the rendered notebook (plotting helpers and other machinery the
    reader does not need in their face), while its output still shows.
    """
    cells, kind, buf = [], None, []
    for line in src.splitlines():
        m = CELL_RE.match(line)
        if m:
            if kind is not None:
                cells.append((kind, buf))
            suffix = m.group(1)
            kind = ("markdown" if "[markdown]" in suffix
                    else "hide" if "[hide]" in suffix else "code")
            buf = []
        else:
            if kind is None:            # preamble before the first marker
                kind, buf = "code", []
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    return cells


def demarkdown(lines):
    """Strip the leading '# ' comment prefix from a markdown cell's lines."""
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.strip() == "#":
            out.append("")
        else:
            out.append(ln)
    return out


def build(src_path):
    src = open(src_path).read()
    nb = new_notebook()
    for kind, lines in split_cells(src):
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue
        body = "\n".join(demarkdown(lines) if kind == "markdown" else lines)
        if kind == "markdown":
            nb.cells.append(new_markdown_cell(body))
        else:
            cell = new_code_cell(body)
            if kind == "hide":
                cell.metadata.update({"jupyter": {"source_hidden": True},
                                      "tags": ["hide-input"]})
            nb.cells.append(cell)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python (accord-chc)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13",
                          "file_extension": ".py", "mimetype": "text/x-python",
                          "nbconvert_exporter": "python",
                          "pygments_lexer": "ipython3"},
    })
    return nb


if __name__ == "__main__":
    src_path, out_path = sys.argv[1], sys.argv[2]
    timeout = 14400
    if "--timeout" in sys.argv:
        timeout = int(sys.argv[sys.argv.index("--timeout") + 1])
    nb = build(src_path)
    print(f"built {len(nb.cells)} cells from {src_path}", flush=True)
    if "--no-exec" in sys.argv:
        nbformat.write(nb, out_path)
        print(f"wrote (unexecuted) {out_path}")
        sys.exit(0)

    import os
    from nbclient import NotebookClient
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(src_path))}},
        allow_errors=False,
    )
    try:
        client.execute()
    finally:
        nbformat.write(nb, out_path)
        print(f"wrote {out_path}", flush=True)
