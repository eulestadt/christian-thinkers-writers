#!/usr/bin/env python3
"""Step 5 (Colab GPU) lives in the 3-arm notebook, not this stub.

Open colab/cai_3way_experiment_and_eval.ipynb on an L4/T4 runtime.
This path exists so the protocol's scripts/05_colab_train.py name is not dangling.
"""

from __future__ import annotations

NOTEBOOK = "colab/cai_3way_experiment_and_eval.ipynb"

if __name__ == "__main__":
    raise SystemExit(
        f"Use {NOTEBOOK} for the 3-arm CAI train+eval harness "
        "(Base / Control / Experimental, Unsloth QLoRA, adapter hot-swap). "
        "Do not train from this stub."
    )
