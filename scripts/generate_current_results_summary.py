from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("generate_final_results_summary.py")
    cmd = [sys.executable, str(script_path), *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
