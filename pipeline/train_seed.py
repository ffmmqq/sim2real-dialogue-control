#!/usr/bin/env python3
"""Portable launcher for learner-seed runs with HybridSimulator seed fixed at 42."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "training.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_provenance(root: Path) -> Dict[str, Any]:
    """Create a deterministic digest of the executable source/config snapshot."""
    included_suffixes = {".py", ".json", ".sh", ".sbatch"}
    excluded_parts = {
        ".cache", ".git", ".venv", "__pycache__", "cluster_runs",
        "runs", "smoke_runs", "slurm_logs", "training_logs",
    }
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in included_suffixes:
            continue
        relative = path.relative_to(root)
        if any(part in excluded_parts for part in relative.parts):
            continue
        files.append({
            "path": relative.as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    digest = hashlib.sha256()
    for entry in files:
        digest.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def runtime_provenance_from_env(required: bool) -> Dict[str, Any] | None:
    value = os.environ.get("IUI27_RUNTIME_PROVENANCE")
    if not value:
        if required:
            raise RuntimeError(
                "Slurm local-Llama run is missing IUI27_RUNTIME_PROVENANCE; "
                "use slurm/run_local_llama_seed_job.sh"
            )
        return None
    path = Path(value)
    if not path.is_file():
        raise RuntimeError(f"Runtime provenance file does not exist: {path}")
    return read_json(path)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def resolve_from_bundle(value: str) -> Path:
    """Resolve a configured path relative to this uploaded bundle."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def local_llm_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def check_local_llm(base_url: str, model: str, api_key: str) -> Dict[str, Any]:
    """Make one tiny request so a batch job fails before a 500-episode run."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply only with OK."}],
        "temperature": 0.0,
        "max_tokens": 4,
    }).encode("utf-8")
    request = urllib.request.Request(
        local_llm_chat_url(base_url), data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Local Llama endpoint is not ready at {local_llm_chat_url(base_url)}: {exc}"
        ) from exc
    if not data.get("choices"):
        raise RuntimeError(f"Local Llama response has no choices: {data}")
    return {"endpoint": local_llm_chat_url(base_url), "model": model, "status": "ok"}


def simulator_seed_audit(project_root: Path) -> Dict[str, Any]:
    """Prove that the training loop leaves the factory's seed at default 42."""
    factory_path = project_root / "simulator" / "__init__.py"
    loop_path = project_root / "pipeline" / "training" / "training_loop.py"
    factory_tree = ast.parse(factory_path.read_text(encoding="utf-8"))
    default = None
    for node in factory_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_simulator":
            positional = node.args.args
            defaults = node.args.defaults
            mapping = dict(zip(positional[-len(defaults):], defaults)) if defaults else {}
            seed_default = mapping.get(next((arg for arg in positional if arg.arg == "seed"), None))
            if isinstance(seed_default, ast.Constant):
                default = seed_default.value
            break

    loop_tree = ast.parse(loop_path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(loop_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_simulator":
            calls.append({
                "line": node.lineno,
                "keywords": [keyword.arg for keyword in node.keywords],
            })
    if default != 42:
        raise RuntimeError(f"Simulator factory default seed changed: expected 42, found {default}")
    if not calls or any("seed" in call["keywords"] for call in calls):
        raise RuntimeError("Training loop now passes a simulator seed; fixed-seed assumption is no longer valid")
    return {
        "factory_seed_default": default,
        "training_get_simulator_calls": calls,
        "conclusion": "HybridSimulator seed remains fixed at factory default 42",
    }


def latest_resume_pair(run_dir: Path) -> Tuple[Path, Path, int]:
    candidates = []
    for model in (run_dir / "checkpoints").glob("checkpoint_ep*_model.pt"):
        try:
            episode = int(model.stem.split("_ep", 1)[1].split("_", 1)[0])
        except (IndexError, ValueError):
            continue
        metrics = model.with_name(model.name.replace("_model.pt", "_metrics.json"))
        if metrics.is_file():
            candidates.append((episode, model, metrics))
    if not candidates:
        raise RuntimeError(f"No resumable checkpoint/metrics pair under {run_dir / 'checkpoints'}")
    episode, model, metrics = max(candidates)
    if episode >= 500:
        raise RuntimeError(f"seed run already has episode-{episode} checkpoint")
    return model, metrics, episode


def training_command(
    *, project_root: Path, run_dir: Path, seed: int, device: str,
    config: Dict[str, Any], resume: bool,
) -> Tuple[List[str], int | None]:
    reward = config["reward"]
    command = [
        sys.executable, "-m", "pipeline.train",
        "--variant", "h1",
        "--simulator", "hybrid",
        "--stochasticity", str(config["simulator_stochasticity"]),
        "--reward_mode", reward["mode"],
        "--episodes", str(config["episodes"]),
        "--turns", str(config["max_turns"]),
        "--lr", str(config["learning_rate"]),
        "--gamma", str(config["gamma"]),
        "--seed", str(seed),
        "--device", device,
        "--name", f"{config['experiment_name']}_learner_seed{seed}",
        "--checkpoint-interval", str(config["checkpoint_interval"]),
        "--map-interval", str(config["map_interval"]),
        "--centred-engagement",
        "--broadened-novelty",
        "--response-type-reward",
        "--alpha-new", str(reward["alpha_new"]),
        "--alpha-rep", str(reward["alpha_rep"]),
        "--alpha-clar", str(reward["alpha_clar"]),
        "--alpha-ask", str(reward["alpha_ask"]),
        "--alpha-stale", str(reward["alpha_stale"]),
        "--alpha-transition", str(reward["alpha_transition"]),
        "--action-repeat-penalty", str(reward["action_repeat_penalty"]),
        "--action-repeat-threshold", str(reward["action_repeat_threshold"]),
        "--w-responsiveness", str(reward["w_responsiveness"]),
        "--w-conclude", str(reward["w_conclude"]),
        "--w-response-type", str(reward["w_response_type"]),
    ]
    resume_episode = None
    if resume:
        model, metrics, resume_episode = latest_resume_pair(run_dir)
        command.extend([
            "--resume-experiment-dir", str(run_dir),
            "--resume-checkpoint", str(model),
            "--resume-metrics", str(metrics),
        ])
    else:
        command.extend(["--experiment-dir", str(run_dir)])
    return command, resume_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="Config file; relative paths are resolved from this bundle",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--run-root",
        help="Override the configured output root (useful for isolated smoke jobs)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config_path = resolve_from_bundle(args.config)
    config = read_json(config_path)
    agent_temperature = config.get("agent_llm_temperature", 0.3)
    simulator_temperature = config.get("simulator_llm_temperature", 0.6)
    agent_max_tokens = config.get("agent_llm_max_tokens", 300)
    simulator_max_tokens = config.get("simulator_llm_max_tokens", 150)
    local_top_p = config.get("local_llm_top_p", 0.9)
    project_root = resolve_from_bundle(config["project_root"])
    run_root = resolve_from_bundle(args.run_root or config.get("run_root", "runs"))
    run_dir = run_root / f"seed_{args.seed}"
    device = args.device or config["default_device"]
    audit = simulator_seed_audit(project_root)

    if not project_root.joinpath("pipeline", "train.py").is_file():
        raise RuntimeError(
            f"Portable source bundle is incomplete: {project_root / 'pipeline' / 'train.py'} not found"
        )
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"{run_dir} is not empty; refuse to overwrite. Use --resume if interrupted.")
    if args.resume and not run_dir.is_dir():
        raise RuntimeError(f"Cannot resume: {run_dir} does not exist")

    command, resume_episode = training_command(
        project_root=project_root, run_dir=run_dir, seed=args.seed,
        device=device, config=config, resume=args.resume,
    )
    print(json.dumps({
        "learner_seed": args.seed,
        "simulator_seed": 42,
        "device": device,
        "resume_from_episode": resume_episode,
        "output": str(run_dir),
        "working_directory": str(project_root),
        "simulator_seed_audit": audit,
        "llm": {
            "backend": config["llm_backend"],
            "model": config["llm_model"],
            "agent_temperature": agent_temperature,
            "agent_max_tokens": agent_max_tokens,
            "simulator_temperature": simulator_temperature,
            "simulator_max_tokens": simulator_max_tokens,
            "top_p": local_top_p,
            "request_seed": config["local_llm_seed"],
            "request_seed_policy": "fixed across learner seeds",
        },
        "command": shlex.join(command),
        "config": str(config_path),
    }, indent=2))
    if args.dry_run:
        return 0

    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but PyTorch reports CUDA unavailable")
    llm_backend = config["llm_backend"]
    local_llm_check = None
    if llm_backend == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("Set OPENROUTER_API_KEY before starting OpenRouter training")
    elif llm_backend == "local_openai":
        base_url = os.environ.get("LOCAL_LLM_BASE_URL", config["local_llm_base_url"])
        api_key = os.environ.get("LOCAL_LLM_API_KEY", "EMPTY")
        local_llm_check = check_local_llm(base_url, config["llm_model"], api_key)
    elif llm_backend != "huggingface":
        raise RuntimeError(f"Unsupported cluster LLM backend: {llm_backend}")

    if args.preflight_only:
        print(json.dumps({"status": "preflight_ok", "llm": local_llm_check}, indent=2))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    default_hf_cache = str(HERE / ".cache" / "huggingface")
    env.update({
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": os.environ.get(
            "MPLCONFIGDIR", str(HERE / ".cache" / "matplotlib")
        ),
        "PYTHONPYCACHEPREFIX": os.environ.get(
            "PYTHONPYCACHEPREFIX", str(HERE / ".cache" / "pycache")
        ),
        "HF_HOME": os.environ.get("HF_HOME", default_hf_cache),
        "TRANSFORMERS_CACHE": os.environ.get(
            "TRANSFORMERS_CACHE", os.environ.get("HF_HOME", default_hf_cache)
        ),
        "HRL_LLM_BACKEND": config["llm_backend"],
        "HRL_LLM_MODEL": config["llm_model"],
        "HRL_AGENT_TEMPERATURE": str(agent_temperature),
        "HRL_SIMULATOR_TEMPERATURE": str(simulator_temperature),
        "HRL_AGENT_MAX_TOKENS": str(agent_max_tokens),
        "HRL_SIMULATOR_MAX_TOKENS": str(simulator_max_tokens),
        "HRL_BERT_DEVICE": os.environ.get("HRL_BERT_DEVICE", "cpu"),
    })
    if llm_backend == "openrouter":
        env.update({
            "OPENROUTER_PROVIDER_ORDER": config["openrouter_provider_order"],
            "OPENROUTER_ALLOW_FALLBACKS": "true",
            "OPENROUTER_REQUIRE_PARAMETERS": "true",
        })
    elif llm_backend == "local_openai":
        env.update({
            "LOCAL_LLM_BASE_URL": os.environ.get(
                "LOCAL_LLM_BASE_URL", config["local_llm_base_url"]
            ),
            "LOCAL_LLM_API_KEY": os.environ.get("LOCAL_LLM_API_KEY", "EMPTY"),
            "LOCAL_LLM_SEED": str(config["local_llm_seed"]),
            "LOCAL_LLM_TOP_P": str(local_top_p),
        })
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["PYTHONPYCACHEPREFIX"]).mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "run_manifest.json"
    runtime_provenance = runtime_provenance_from_env(
        required=bool(os.environ.get("SLURM_JOB_ID") and llm_backend == "local_openai")
    )
    code_provenance = source_tree_provenance(project_root)
    previous = read_json(manifest_path) if manifest_path.exists() else {}
    events = previous.get("events", [])
    events.append({
        "type": "resume" if args.resume else "start",
        "at": now(),
        "resume_from_episode": resume_episode,
        "command": command,
    })
    manifest = {
        "status": "running",
        "learner_seed": args.seed,
        "simulator_seed": 42,
        "simulator_seed_policy": "fixed across learner seeds",
        "device": device,
        "project_root": str(project_root),
        "project_git_commit": git_commit(project_root),
        "code_provenance": code_provenance,
        "config_snapshot": config,
        "simulator_seed_audit": audit,
        "llm_preflight": local_llm_check,
        "runtime_provenance": runtime_provenance,
        "events": events,
    }
    write_json(manifest_path, manifest)
    (run_dir / "run_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")

    log_mode = "a" if args.resume else "w"
    log_path = run_dir / "train_stdout.log"
    with log_path.open(log_mode, encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command, cwd=project_root, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()

    manifest["status"] = "complete" if return_code == 0 else "failed"
    manifest["finished_at"] = now()
    manifest["return_code"] = return_code
    final_model = run_dir / "models" / "trained_agent.pt"
    if final_model.is_file():
        manifest["final_model"] = str(final_model)
        manifest["final_model_sha256"] = sha256(final_model)
    write_json(manifest_path, manifest)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
