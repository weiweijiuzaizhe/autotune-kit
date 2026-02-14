import argparse
import sys

from atk.run import run_from_config_path, rerun_from_run_dir
from atk.bootstrap import bootstrap as bootstrap_deps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atk", description="ATK minimal CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a new training pipeline")
    p_run.add_argument("--config", required=True, help="Path to atk YAML config")

    p_rerun = sub.add_parser("rerun", help="Rerun using a previous run_dir/config.effective.yaml")
    p_rerun.add_argument("run_dir", help="Path to previous run_dir")

    p_boot = sub.add_parser("bootstrap", help="Install runtime deps (torch + LF stack)")
    p_boot.add_argument("--cuda", default="cu126", choices=["cu126", "cpu"], help="PyTorch wheel index selector")
    p_boot.add_argument("--skip-torch", action="store_true", help="Do not install torch")
    p_boot.add_argument("--torch-spec", default=None, help="Override torch spec (e.g. torch==2.7.0+cu126)")

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        return run_from_config_path(args.config)
    if args.cmd == "bootstrap":
        return bootstrap_deps(cuda=args.cuda, install_torch=(not args.skip_torch), torch_spec=args.torch_spec)
    if args.cmd == "rerun":
        return rerun_from_run_dir(args.run_dir)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
