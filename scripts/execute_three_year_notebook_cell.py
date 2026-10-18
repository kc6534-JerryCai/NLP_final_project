"""Execute and persist the final three-year analysis cell without nbconvert."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "macro_event_surprise_dataframe_3year.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell = notebook["cells"][-1]
    if cell.get("cell_type") != "code" or "THREE_YEAR_DIR" not in "".join(cell.get("source", [])):
        raise RuntimeError("The final notebook cell is not the expected three-year analysis cell.")

    outputs: list[dict[str, Any]] = []

    class NotebookStdout(io.TextIOBase):
        def write(self, value: str) -> int:
            if value:
                if outputs and outputs[-1].get("output_type") == "stream":
                    outputs[-1]["text"] += value
                else:
                    outputs.append({"name": "stdout", "output_type": "stream", "text": value})
            return len(value)

        def flush(self) -> None:
            return None

    def notebook_display(value: Any) -> None:
        data: dict[str, Any] = {"text/plain": repr(value)}
        if hasattr(value, "to_html"):
            data["text/html"] = value.to_html()
        outputs.append({"data": data, "metadata": {}, "output_type": "display_data"})

    def notebook_show(*_args: Any, **_kwargs: Any) -> None:
        figure = plt.gcf()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        outputs.append(
            {
                "data": {
                    "image/png": encoded,
                    "text/plain": f"<Figure size {figure.get_figwidth() * figure.dpi:.0f}x{figure.get_figheight() * figure.dpi:.0f}>",
                },
                "metadata": {},
                "output_type": "display_data",
            }
        )
        plt.close(figure)

    namespace = {
        "Path": Path,
        "ROOT": ROOT,
        "pd": pd,
        "np": np,
        "plt": plt,
        "display": notebook_display,
        "EXT_EVENT_ORDER": ["CPI_RELEASE", "EMPLOYMENT_RELEASE", "PCE_RELEASE", "FOMC_RATE_DECISION"],
        "EXT_EVENT_TITLES": {
            "CPI_RELEASE": "CPI release",
            "EMPLOYMENT_RELEASE": "Employment release",
            "PCE_RELEASE": "PCE release",
            "FOMC_RATE_DECISION": "FOMC rate decision",
        },
        "EXT_LABEL_NAMES": {"利好": "Bullish", "利空": "Bearish", "中性": "Neutral"},
        "EXT_LABEL_COLORS": {"利好": "#009E73", "利空": "#D55E00", "中性": "#6B7280"},
    }

    original_show = plt.show
    plt.show = notebook_show
    stdout = NotebookStdout()
    try:
        import contextlib

        with contextlib.redirect_stdout(stdout):
            exec(compile("".join(cell["source"]), str(NOTEBOOK_PATH), "exec"), namespace)
    finally:
        plt.show = original_show

    cell["outputs"] = outputs
    cell["execution_count"] = max(
        (item.get("execution_count") or 0 for item in notebook["cells"] if item.get("cell_type") == "code"),
        default=0,
    ) + 1
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NOTEBOOK_PATH} with {len(outputs)} persisted outputs.")


if __name__ == "__main__":
    main()
