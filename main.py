from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.dataset_config import (
    get_first_available,
    get_config_int,
    get_config_scalar,
    load_dataset_config,
    print_resolved_config_summary,
    validate_script_config,
)


STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

DATASET_CHOICES = ("physionet", "mimic")
STAGE_SEQUENCE = [
    "graph",
    "majority_vote",
    "mortality_prediction",
    "matching",
    "cate_estimation",
    "analyze_cate_results",
    "permutations_test",
]


@dataclass
class StageRecord:
    name: str
    script_path: str
    log_path: str
    output_paths: dict[str, str] = field(default_factory=dict)
    required_output_keys: list[str] = field(default_factory=list)
    primary_output_key: str | None = None
    background: bool = False
    status: str = STATUS_PENDING
    command: list[str] = field(default_factory=list)
    cwd: str | None = None
    return_code: int | None = None
    error: str | None = None
    failure_kind: str | None = None
    skip_reason: str | None = None
    started_at: str | None = None
    ended_at: str | None = None


@dataclass
class RunContext:
    repo_root: Path
    output_root: Path
    logs_dir: Path
    stage_dirs: dict[str, Path]
    dataset: str
    dataset_config_csv: Path
    config: dict[str, object]
    model_type: str
    trials: int
    latent_tags_dir: Path
    dataset_pkl_path: Path
    stages: dict[str, StageRecord]
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the post-preprocessing causal inference experiment pipeline."
    )
    parser.add_argument(
        "--latent-tags-dir",
        default=None,
        help="Directory containing the latent-tag voter CSV files.",
    )
    parser.add_argument(
        "--dataset-pkl-path",
        default=None,
        help="Path to the processed dataset pickle.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_CHOICES,
        help="Dataset identifier for graph selection and downstream script defaults.",
    )
    parser.add_argument(
        "--dataset-config-csv",
        default=None,
        help=(
            "Path to the dataset global-variables CSV. If omitted, use the default "
            "config for --dataset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root directory for the orchestrated run. Default: current working directory.",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and child commands, then exit.",
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def resolve_directory_path(path_like: str, field_name: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{field_name} is not a directory: {path}")
    return path


def resolve_file_path(path_like: str, field_name: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} is not a file: {path}")
    return path


def resolve_output_root(path_like: str | None) -> Path:
    if path_like is None:
        return Path.cwd().resolve()
    path = Path(path_like).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output-dir is not a directory: {path}")
    return path


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_stage_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Stage output path is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_run_context(args: argparse.Namespace) -> RunContext:
    repo_root = Path(__file__).resolve().parent
    config = load_dataset_config(args.dataset, args.dataset_config_csv)
    dataset_config_csv = Path(str(config["__config_csv_path__"])).resolve()
    model_type = str(get_config_scalar(config, "MODEL_TYPE", "CausalForest"))
    trials = get_config_int(config, "TRIALS", 10)
    if trials is None:
        trials = 10
    output_root = resolve_output_root(args.output_dir)
    logs_dir = output_root / "logs"
    stage_dirs = {
        "graph": output_root / "graph",
        "majority_vote": output_root / "majority_vote",
        "mortality_prediction": output_root / "mortality_prediction",
        "matching": output_root / "matching",
        "cate_estimation": output_root / "cate_estimation",
        "analyze_cate_results": output_root / "analyze_cate_results",
        "permutations_test": output_root / "permutations_test",
    }

    graph_script = (
        repo_root / "src" / "physionet2012_causal_graph.py"
        if args.dataset == "physionet"
        else repo_root / "src" / "mimiciii_causal_graph.py"
    )
    stage_scripts = {
        "graph": graph_script,
        "majority_vote": repo_root / "src" / "majority_vote_latents.py",
        "mortality_prediction": repo_root / "src" / "mortality_prediction_using_latents.py",
        "matching": repo_root / "src" / "matching_causal_effect.py",
        "cate_estimation": repo_root / "src" / "cate_estimation.py",
        "analyze_cate_results": repo_root / "src" / "analyze_cate_results.py",
        "permutations_test": repo_root / "src" / "permutations_test.py",
    }

    graph_pkl_path = stage_dirs["graph"] / f"{args.dataset}_causal_graph.pkl"
    graph_png_path = stage_dirs["graph"] / f"{args.dataset}_causal_dag.png"
    majority_vote_csv = stage_dirs["majority_vote"] / "latent_tags_majority_vote.csv"
    mortality_results_txt = (
        stage_dirs["mortality_prediction"] / "mortality_prediction_results.txt"
    )
    matching_summary_csv = stage_dirs["matching"] / "global_summary.csv"
    cate_summary_csv = stage_dirs["cate_estimation"] / "global_summary.csv"
    cate_control_csv = (
        stage_dirs["cate_estimation"] / "control_messages_cate_estimation.csv"
    )
    analyze_summary_csv = stage_dirs["analyze_cate_results"] / "benchmark_summary.csv"
    analyze_control_csv = (
        stage_dirs["analyze_cate_results"]
        / "control_messages_analyze_cate_results.csv"
    )
    treatment_perm_csv = (
        stage_dirs["permutations_test"] / "treatment_permutation_results.csv"
    )
    outcome_perm_csv = (
        stage_dirs["permutations_test"] / "outcome_permutation_results.csv"
    )

    stages = {
        "graph": StageRecord(
            name="graph",
            script_path=str(stage_scripts["graph"]),
            log_path=str(logs_dir / "01_graph.log"),
            output_paths={
                "graph_dir": str(stage_dirs["graph"]),
                "graph_pkl_path": str(graph_pkl_path),
                "graph_png_path": str(graph_png_path),
            },
            required_output_keys=["graph_pkl_path", "graph_png_path"],
            primary_output_key="graph_pkl_path",
        ),
        "majority_vote": StageRecord(
            name="majority_vote",
            script_path=str(stage_scripts["majority_vote"]),
            log_path=str(logs_dir / "02_majority_vote.log"),
            output_paths={
                "output_dir": str(stage_dirs["majority_vote"]),
                "majority_vote_csv": str(majority_vote_csv),
            },
            required_output_keys=["majority_vote_csv"],
            primary_output_key="majority_vote_csv",
        ),
        "mortality_prediction": StageRecord(
            name="mortality_prediction",
            script_path=str(stage_scripts["mortality_prediction"]),
            log_path=str(logs_dir / "03_mortality_prediction.log"),
            output_paths={
                "output_dir": str(stage_dirs["mortality_prediction"]),
                "results_txt": str(mortality_results_txt),
            },
            required_output_keys=["results_txt"],
            primary_output_key="results_txt",
            background=True,
        ),
        "matching": StageRecord(
            name="matching",
            script_path=str(stage_scripts["matching"]),
            log_path=str(logs_dir / "04_matching.log"),
            output_paths={
                "output_dir": str(stage_dirs["matching"]),
                "global_summary_csv": str(matching_summary_csv),
            },
            required_output_keys=["global_summary_csv"],
            primary_output_key="output_dir",
            background=True,
        ),
        "cate_estimation": StageRecord(
            name="cate_estimation",
            script_path=str(stage_scripts["cate_estimation"]),
            log_path=str(logs_dir / "05_cate_estimation.log"),
            output_paths={
                "output_dir": str(stage_dirs["cate_estimation"]),
                "global_summary_csv": str(cate_summary_csv),
                "control_messages_csv": str(cate_control_csv),
            },
            required_output_keys=["global_summary_csv", "control_messages_csv"],
            primary_output_key="output_dir",
        ),
        "analyze_cate_results": StageRecord(
            name="analyze_cate_results",
            script_path=str(stage_scripts["analyze_cate_results"]),
            log_path=str(logs_dir / "06_analyze_cate_results.log"),
            output_paths={
                "output_dir": str(stage_dirs["analyze_cate_results"]),
                "benchmark_summary_csv": str(analyze_summary_csv),
                "control_messages_csv": str(analyze_control_csv),
            },
            required_output_keys=["benchmark_summary_csv", "control_messages_csv"],
            primary_output_key="output_dir",
        ),
        "permutations_test": StageRecord(
            name="permutations_test",
            script_path=str(stage_scripts["permutations_test"]),
            log_path=str(logs_dir / "07_permutations_test.log"),
            output_paths={
                "output_dir": str(stage_dirs["permutations_test"]),
                "treatment_results_csv": str(treatment_perm_csv),
                "outcome_results_csv": str(outcome_perm_csv),
            },
            required_output_keys=["treatment_results_csv", "outcome_results_csv"],
            primary_output_key="output_dir",
        ),
    }

    if args.latent_tags_dir is None:
        if args.validate_config_only:
            latent_tags_dir = Path(".").resolve()
        else:
            raise ValueError("Provide --latent-tags-dir for a full pipeline run.")
    else:
        latent_tags_dir = Path(args.latent_tags_dir).expanduser().resolve()

    dataset_pkl_value = args.dataset_pkl_path
    if dataset_pkl_value is None and args.validate_config_only:
        dataset_pkl_value = str(
            get_first_available(
                config,
                ["DATASET_PKL_PATH", "PHYSIONET_PKL_PATH"],
                "dataset.pkl",
            )
        )
    if dataset_pkl_value is None:
        raise ValueError("Provide --dataset-pkl-path for a full pipeline run.")

    return RunContext(
        repo_root=repo_root,
        output_root=output_root,
        logs_dir=logs_dir,
        stage_dirs=stage_dirs,
        dataset=args.dataset,
        dataset_config_csv=dataset_config_csv,
        config=config,
        model_type=model_type,
        trials=trials,
        latent_tags_dir=latent_tags_dir,
        dataset_pkl_path=Path(dataset_pkl_value).expanduser().resolve(),
        stages=stages,
        summary_path=output_root / "run_summary.json",
    )


def validate_config_only(args: argparse.Namespace, context: RunContext) -> None:
    resolved = validate_script_config("main.py", context.config)
    print_resolved_config_summary("main.py", context.config, resolved)

    if args.latent_tags_dir is not None:
        resolve_directory_path(str(context.latent_tags_dir), "latent-tags dir")
    if args.dataset_pkl_path is not None:
        resolve_file_path(str(context.dataset_pkl_path), "dataset pickle")

    print("  child_commands:")
    for stage_name, builder in [
        ("graph", build_graph_command),
        ("majority_vote", build_majority_vote_command),
        ("mortality_prediction", build_mortality_command),
        ("matching", build_matching_command),
        ("cate_estimation", build_cate_command),
        ("analyze_cate_results", build_analyze_command),
        ("permutations_test", build_permutations_command),
    ]:
        command = builder(context)
        print(f"    {stage_name}: {format_command(command)}")


def prepare_output_directories(context: RunContext) -> None:
    ensure_directory(context.output_root)
    ensure_directory(context.logs_dir)
    for stage_dir in context.stage_dirs.values():
        ensure_directory(stage_dir)


def validate_context(context: RunContext) -> None:
    resolve_directory_path(str(context.latent_tags_dir), "latent-tags dir")
    resolve_file_path(str(context.dataset_pkl_path), "dataset pickle")

    if not sys.executable:
        raise RuntimeError("sys.executable is empty; cannot launch child scripts.")

    python_path = Path(sys.executable).expanduser().resolve()
    if not python_path.exists():
        raise FileNotFoundError(f"Python executable does not exist: {python_path}")
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable is not a file: {python_path}")

    for stage_name in STAGE_SEQUENCE:
        script_path = Path(context.stages[stage_name].script_path)
        if not script_path.exists():
            raise FileNotFoundError(
                f"Required script for stage '{stage_name}' does not exist: {script_path}"
            )
        if not script_path.is_file():
            raise FileNotFoundError(
                f"Required script for stage '{stage_name}' is not a file: {script_path}"
            )


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return env


def set_stage_fields(
    stage: StageRecord,
    lock: threading.Lock,
    **fields: object,
) -> None:
    with lock:
        for key, value in fields.items():
            setattr(stage, key, value)


def verify_required_outputs(stage: StageRecord) -> None:
    for output_key in stage.required_output_keys:
        output_path = Path(stage.output_paths[output_key])
        if not output_path.exists():
            raise FileNotFoundError(
                f"expected output '{output_key}' was not created: {output_path}"
            )
        if not output_path.is_file():
            raise FileNotFoundError(
                f"expected output '{output_key}' is not a file: {output_path}"
            )


def append_log(log_path: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message)


def run_stage_subprocess(
    stage: StageRecord,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    lock: threading.Lock,
) -> bool:
    stage.started_at = utc_timestamp()
    set_stage_fields(
        stage,
        lock,
        command=list(command),
        cwd=str(cwd),
        status=STATUS_RUNNING,
        return_code=None,
        error=None,
        failure_kind=None,
        skip_reason=None,
        started_at=stage.started_at,
        ended_at=None,
    )

    log_header = (
        f"Stage: {stage.name}\n"
        f"Command: {format_command(command)}\n"
        f"Cwd: {cwd}\n"
        f"Started at (UTC): {stage.started_at}\n\n"
    )

    try:
        with open(stage.log_path, "w", encoding="utf-8") as log_file:
            log_file.write(log_header)
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    except Exception as exc:
        ended_at = utc_timestamp()
        error_text = (
            f"[main] Orchestration failure during {stage.name} stage: "
            f"failed to launch child process ({type(exc).__name__}: {exc})"
        )
        set_stage_fields(
            stage,
            lock,
            status=STATUS_FAILED,
            failure_kind="main",
            error=error_text,
            ended_at=ended_at,
        )
        append_log(
            stage.log_path,
            "\nLauncher exception:\n" + traceback.format_exc(),
        )
        print(error_text)
        return False

    ended_at = utc_timestamp()
    set_stage_fields(
        stage,
        lock,
        return_code=completed.returncode,
        ended_at=ended_at,
    )

    if completed.returncode != 0:
        script_name = Path(stage.script_path).name
        error_text = (
            f"[child failure] {script_name} exited with code {completed.returncode}. "
            f"See log: {stage.log_path}"
        )
        set_stage_fields(
            stage,
            lock,
            status=STATUS_FAILED,
            failure_kind="child",
            error=error_text,
        )
        append_log(
            stage.log_path,
            f"\nExit code: {completed.returncode}\nFinished at (UTC): {ended_at}\n",
        )
        print(error_text)
        return False

    try:
        verify_required_outputs(stage)
    except Exception as exc:
        error_text = (
            f"[main] Orchestration failure during {stage.name} stage: {exc}"
        )
        set_stage_fields(
            stage,
            lock,
            status=STATUS_FAILED,
            failure_kind="main",
            error=error_text,
        )
        append_log(
            stage.log_path,
            "\nVerification failure:\n"
            f"{type(exc).__name__}: {exc}\n"
            f"Finished at (UTC): {ended_at}\n",
        )
        print(error_text)
        return False

    set_stage_fields(
        stage,
        lock,
        status=STATUS_SUCCESS,
    )
    append_log(
        stage.log_path,
        f"\nExit code: 0\nFinished at (UTC): {ended_at}\n",
    )
    return True


def mark_stage_skipped(stage: StageRecord, lock: threading.Lock, reason: str) -> None:
    if stage.status != STATUS_PENDING:
        return
    set_stage_fields(
        stage,
        lock,
        status=STATUS_SKIPPED,
        skip_reason=reason,
        ended_at=utc_timestamp(),
    )


def mark_stages_skipped(
    context: RunContext,
    stage_names: list[str],
    lock: threading.Lock,
    reason: str,
) -> None:
    for stage_name in stage_names:
        mark_stage_skipped(context.stages[stage_name], lock, reason)


def build_graph_command(context: RunContext) -> list[str]:
    stage = context.stages["graph"]
    return [
        sys.executable,
        stage.script_path,
        "--dataset-config-csv",
        str(context.dataset_config_csv),
        "--graph-pkl-path",
        stage.output_paths["graph_pkl_path"],
        "--graph-png-path",
        stage.output_paths["graph_png_path"],
    ]


def build_majority_vote_command(context: RunContext) -> list[str]:
    stage = context.stages["majority_vote"]
    return [
        sys.executable,
        stage.script_path,
        "--input-dir",
        str(context.latent_tags_dir),
        "--output-path",
        stage.output_paths["majority_vote_csv"],
    ]


def build_mortality_command(context: RunContext) -> list[str]:
    stage = context.stages["mortality_prediction"]
    return [
        sys.executable,
        stage.script_path,
        "--model",
        context.dataset,
        "--dataset-config-csv",
        str(context.dataset_config_csv),
        "--latent-tags-path",
        context.stages["majority_vote"].output_paths["majority_vote_csv"],
        "--physionet-pkl-path",
        str(context.dataset_pkl_path),
        "--results-txt-path",
        stage.output_paths["results_txt"],
    ]


def build_matching_command(context: RunContext) -> list[str]:
    stage = context.stages["matching"]
    return [
        sys.executable,
        stage.script_path,
        "--model",
        context.dataset,
        "--dataset-config-csv",
        str(context.dataset_config_csv),
        "--latent-tags-path",
        context.stages["majority_vote"].output_paths["majority_vote_csv"],
        "--physionet-pkl-path",
        str(context.dataset_pkl_path),
        "--graph-pkl-path",
        context.stages["graph"].output_paths["graph_pkl_path"],
        "--output-dir",
        stage.output_paths["output_dir"],
    ]


def build_cate_command(context: RunContext) -> list[str]:
    stage = context.stages["cate_estimation"]
    return [
        sys.executable,
        stage.script_path,
        "--model",
        context.dataset,
        "--dataset-config-csv",
        str(context.dataset_config_csv),
        "--latent-tags-path",
        context.stages["majority_vote"].output_paths["majority_vote_csv"],
        "--physionet-pkl-path",
        str(context.dataset_pkl_path),
        "--graph-pkl-path",
        context.stages["graph"].output_paths["graph_pkl_path"],
        "--output-dir",
        stage.output_paths["output_dir"],
        "--model-type",
        context.model_type,
    ]


def build_analyze_command(context: RunContext) -> list[str]:
    stage = context.stages["analyze_cate_results"]
    return [
        sys.executable,
        stage.script_path,
        "--model",
        context.dataset,
        "--dataset-config-csv",
        str(context.dataset_config_csv),
        "--latent-tags-path",
        context.stages["majority_vote"].output_paths["majority_vote_csv"],
        "--physionet-pkl-path",
        str(context.dataset_pkl_path),
        "--results-dir",
        context.stages["cate_estimation"].output_paths["output_dir"],
        "--output-dir",
        stage.output_paths["output_dir"],
    ]


def build_permutations_command(context: RunContext) -> list[str]:
    stage = context.stages["permutations_test"]
    return [
        sys.executable,
        stage.script_path,
        "--model",
        context.dataset,
        "--dataset-config-csv",
        str(context.dataset_config_csv),
        "--trials",
        str(context.trials),
        "--experiment-dir",
        stage.output_paths["output_dir"],
        "--latent-tags-path",
        context.stages["majority_vote"].output_paths["majority_vote_csv"],
        "--physionet-pkl-path",
        str(context.dataset_pkl_path),
        "--graph-pkl-path",
        context.stages["graph"].output_paths["graph_pkl_path"],
        "--model-type",
        context.model_type,
    ]


def primary_output_path(stage: StageRecord) -> str | None:
    if stage.primary_output_key is None:
        return None
    return stage.output_paths.get(stage.primary_output_key)


def background_stage_runner(
    context: RunContext,
    stage_name: str,
    command: list[str],
    env: dict[str, str],
    lock: threading.Lock,
) -> None:
    stage = context.stages[stage_name]
    success = run_stage_subprocess(stage, command, context.repo_root, env, lock)
    if success:
        output_path = primary_output_path(stage)
        if stage_name == "mortality_prediction":
            print(
                "mortality_prediction_using_latents.py completed successfully. "
                f"Output: {output_path}"
            )
        elif stage_name == "matching":
            print(
                "matching_causal_effect.py completed successfully. "
                f"Output dir: {output_path}"
            )


def write_run_summary(
    context: RunContext,
    overall_status: str,
    overall_error: str | None,
) -> None:
    summary_payload = {
        "overall_status": overall_status,
        "overall_error": overall_error,
        "generated_at_utc": utc_timestamp(),
        "repo_root": str(context.repo_root),
        "output_root": str(context.output_root),
        "dataset": context.dataset,
        "dataset_config_csv": str(context.dataset_config_csv),
        "latent_tags_dir": str(context.latent_tags_dir),
        "dataset_pkl_path": str(context.dataset_pkl_path),
        "stages": {
            stage_name: asdict(context.stages[stage_name])
            for stage_name in STAGE_SEQUENCE
        },
    }
    ensure_directory(context.summary_path.parent)
    with open(context.summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, indent=2, ensure_ascii=False)


def print_final_summary(context: RunContext, overall_status: str) -> None:
    print(f"Final status: {overall_status.upper()}")
    for stage_name in STAGE_SEQUENCE:
        stage = context.stages[stage_name]
        summary_line = (
            f"- {stage_name}: {stage.status} | log: {stage.log_path}"
        )
        output_path = primary_output_path(stage)
        if output_path is not None:
            summary_line += f" | output: {output_path}"
        if stage.status == STATUS_SKIPPED and stage.skip_reason:
            summary_line += f" | reason: {stage.skip_reason}"
        if stage.status == STATUS_FAILED and stage.error:
            summary_line += f" | error: {stage.error}"
        print(summary_line)
    print(f"Run summary JSON: {context.summary_path}")


def orchestrate(context: RunContext) -> tuple[bool, str | None]:
    env = build_env()
    lock = threading.Lock()
    background_threads: dict[str, threading.Thread] = {}
    foreground_ok = True
    overall_error: str | None = None

    try:
        reset_stage_directory(context.stage_dirs["graph"])
        print(f"[1/7] Building causal graph for dataset={context.dataset}")
        if not run_stage_subprocess(
            context.stages["graph"],
            build_graph_command(context),
            context.repo_root,
            env,
            lock,
        ):
            foreground_ok = False
            overall_error = context.stages["graph"].error
            mark_stages_skipped(
                context,
                [
                    "majority_vote",
                    "mortality_prediction",
                    "matching",
                    "cate_estimation",
                    "analyze_cate_results",
                    "permutations_test",
                ],
                lock,
                "Skipped because the graph stage failed.",
            )
        else:
            print(
                "Graph script completed successfully. Graph pickle: "
                f"{context.stages['graph'].output_paths['graph_pkl_path']}"
            )

        if foreground_ok:
            reset_stage_directory(context.stage_dirs["majority_vote"])
            print("[2/7] Creating majority-vote latent CSV")
            if not run_stage_subprocess(
                context.stages["majority_vote"],
                build_majority_vote_command(context),
                context.repo_root,
                env,
                lock,
            ):
                foreground_ok = False
                overall_error = context.stages["majority_vote"].error
                mark_stages_skipped(
                    context,
                    [
                        "mortality_prediction",
                        "matching",
                        "cate_estimation",
                        "analyze_cate_results",
                        "permutations_test",
                    ],
                    lock,
                    "Skipped because the majority-vote stage failed.",
                )
            else:
                print(
                    "Majority vote completed. Output: "
                    f"{context.stages['majority_vote'].output_paths['majority_vote_csv']}"
                )

        if foreground_ok:
            print("[3/7] Starting background validation tasks")
            for stage_name, command_builder in (
                ("mortality_prediction", build_mortality_command),
                ("matching", build_matching_command),
            ):
                reset_stage_directory(context.stage_dirs[stage_name])
                thread = threading.Thread(
                    target=background_stage_runner,
                    args=(context, stage_name, command_builder(context), env, lock),
                    name=f"{stage_name}_thread",
                    daemon=False,
                )
                background_threads[stage_name] = thread
                thread.start()
                print(
                    f"Started {stage_name} thread. Log: "
                    f"{context.stages[stage_name].log_path}"
                )

            reset_stage_directory(context.stage_dirs["cate_estimation"])
            print("[4/7] Running cate_estimation.py")
            if not run_stage_subprocess(
                context.stages["cate_estimation"],
                build_cate_command(context),
                context.repo_root,
                env,
                lock,
            ):
                foreground_ok = False
                overall_error = context.stages["cate_estimation"].error
                mark_stages_skipped(
                    context,
                    ["analyze_cate_results", "permutations_test"],
                    lock,
                    "Skipped because cate_estimation.py failed.",
                )
            else:
                print("cate_estimation.py completed successfully")

        if foreground_ok:
            reset_stage_directory(context.stage_dirs["analyze_cate_results"])
            print("[5/7] Running analyze_cate_results.py")
            if not run_stage_subprocess(
                context.stages["analyze_cate_results"],
                build_analyze_command(context),
                context.repo_root,
                env,
                lock,
            ):
                foreground_ok = False
                overall_error = context.stages["analyze_cate_results"].error
                mark_stages_skipped(
                    context,
                    ["permutations_test"],
                    lock,
                    "Skipped because analyze_cate_results.py failed.",
                )
            else:
                print("analyze_cate_results.py completed successfully")

        if foreground_ok:
            reset_stage_directory(context.stage_dirs["permutations_test"])
            print(f"[6/7] Running permutations_test.py with {context.trials} trials")
            if not run_stage_subprocess(
                context.stages["permutations_test"],
                build_permutations_command(context),
                context.repo_root,
                env,
                lock,
            ):
                foreground_ok = False
                overall_error = context.stages["permutations_test"].error
            else:
                print("permutations_test.py completed successfully")
    except Exception as exc:
        overall_error = f"[main] Orchestration failure: {type(exc).__name__}: {exc}"
        print(overall_error)
        append_log(
            str(context.logs_dir / "00_main_orchestration_failure.log"),
            overall_error + "\n\n" + traceback.format_exc(),
        )
        pending_stages = [
            stage_name
            for stage_name in STAGE_SEQUENCE
            if context.stages[stage_name].status == STATUS_PENDING
        ]
        mark_stages_skipped(
            context,
            pending_stages,
            lock,
            "Skipped because main.py orchestration failed.",
        )
        foreground_ok = False
    finally:
        if background_threads:
            print("[7/7] Waiting for background tasks to finish")
            for stage_name in ("mortality_prediction", "matching"):
                thread = background_threads.get(stage_name)
                if thread is not None:
                    thread.join()

    overall_success = foreground_ok
    for stage_name in STAGE_SEQUENCE:
        if context.stages[stage_name].status not in {STATUS_SUCCESS, STATUS_SKIPPED}:
            overall_success = False
        if context.stages[stage_name].status == STATUS_FAILED and overall_error is None:
            overall_error = context.stages[stage_name].error

    if any(context.stages[name].status == STATUS_FAILED for name in STAGE_SEQUENCE):
        overall_success = False

    return overall_success, overall_error


def main() -> int:
    args = parse_args()
    try:
        context = build_run_context(args)
    except Exception as exc:
        print(f"[main] Validation failure: {exc}")
        return 1

    if args.validate_config_only:
        try:
            validate_config_only(args, context)
        except Exception as exc:
            print(f"[main] Validation failure: {exc}")
            return 1
        return 0

    overall_status = "failed"
    overall_error: str | None = None

    try:
        prepare_output_directories(context)
        validate_context(context)
    except Exception as exc:
        overall_error = f"[main] Validation failure: {exc}"
        for stage_name in STAGE_SEQUENCE:
            if context.stages[stage_name].status == STATUS_PENDING:
                context.stages[stage_name].status = STATUS_SKIPPED
                context.stages[stage_name].skip_reason = (
                    "Skipped because main.py validation/setup failed."
                )
                context.stages[stage_name].ended_at = utc_timestamp()
        print(overall_error)
        if context.logs_dir.exists():
            setup_log = context.logs_dir / "00_main_setup_failure.log"
            with open(setup_log, "w", encoding="utf-8") as log_file:
                log_file.write(overall_error + "\n\n")
                log_file.write(traceback.format_exc())
    else:
        overall_success, overall_error = orchestrate(context)
        overall_status = "success" if overall_success else "failed"

    try:
        write_run_summary(context, overall_status, overall_error)
    except Exception as exc:
        print(
            f"[main] Failed to write run summary JSON: {type(exc).__name__}: {exc}"
        )

    print_final_summary(context, overall_status)
    return 0 if overall_status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
