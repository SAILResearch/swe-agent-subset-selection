"""Stage 3: sanitized trajectories -> trajectory embeddings (paper Section 4.3).

    python pipeline/embed.py --dataset multi_model [--input SANITIZED_DIR] [--output VECTOR_DIR]
                             [--files FILE ...] [--device mps|cpu]

Every step is embedded with Nomic Embed v1.5 (768 dimensions) from the text
``"Action: <action> | Result: <observation>"``. Two variants are written per trajectory:

    VECTOR_DIR/timeseries/<run>/<name>_ts.npy     float32 matrix, steps x 768
    VECTOR_DIR/pooled/<run>/<name>_pooled.npy     float32 vector of 5 x 768 = 3,840 values:
                                                  first step, middle step, last step, mean, standard deviation

``<name>`` is ``<OUTCOME>_<repository>_<trajectory id>`` with OUTCOME = SUCCESS or FAIL. Later stages read
the pass/fail outcome of a run from this prefix, so the naming must be kept.

Steps longer than the model's context are split into overlapping token windows whose embeddings are
averaged. ``--files`` restricts the run to the given sanitized files, so any subset can be embedded.

The vectors used in the paper are published with the package; embedding is slow and hardware-dependent,
so the later stages run on the published vectors and ``verify_embeddings.py`` checks this script against
them on a sample.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "e5cf08aadaa33385f5990def41f7a23405aec398"
MODEL_CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"   # nomic-ai/nomic-bert-2048 (model code)
MAX_SEQUENCE_TOKENS = 8192
DIRECT_ENCODE_MARGIN = 20        # steps up to MAX_SEQUENCE_TOKENS - 20 tokens are encoded in one pass
WINDOW_TOKENS = MAX_SEQUENCE_TOKENS - 100
WINDOW_OVERLAP = 200
MAX_CHARACTERS = 30000           # longer step texts keep their first 15,000 and last 10,000 characters
ACCELERATOR_MEMORY_LIMIT_GB = 12.0


def default_device():
    """Apple-silicon GPU when available, otherwise CPU (the devices used for the published vectors)."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device):
    """Load the pinned embedding model on a device."""
    torch.manual_seed(config.SEED)
    model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True, revision=MODEL_REVISION,
                                model_kwargs={"code_revision": MODEL_CODE_REVISION},
                                config_kwargs={"code_revision": MODEL_CODE_REVISION})
    model.max_seq_length = MAX_SEQUENCE_TOKENS
    return model


def release_accelerator_memory(device):
    """Free cached accelerator memory when the allocation exceeds the limit."""
    if device == "mps" and torch.mps.current_allocated_memory() / 1024 ** 3 > ACCELERATOR_MEMORY_LIMIT_GB:
        torch.mps.empty_cache()


def encode(model, texts):
    """Normalised embeddings of a list of texts on the model's current device."""
    with torch.no_grad():
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)


def encode_on_cpu(model, texts, device):
    """Encode on the CPU and move the model back to its device."""
    model.to("cpu")
    try:
        return encode(model, texts)
    finally:
        model.to(device)


def encode_with_cpu_fallback(model, text, device):
    """Encode one text on the device; if the accelerator runs out of memory, encode it on the CPU."""
    try:
        return encode(model, [text])[0]
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        return encode_on_cpu(model, [text], device)[0]


def token_windows(model, tokens):
    """Overlapping token windows of a long step, decoded back to text."""
    windows, start = [], 0
    while True:
        end = min(start + WINDOW_TOKENS, len(tokens))
        windows.append(model.tokenizer.decode(tokens[start:end]))
        if end >= len(tokens):
            return windows
        start += WINDOW_TOKENS - WINDOW_OVERLAP


def embed_text(model, text, device):
    """Embedding of one step text; long texts are windowed, the windows encoded on the CPU and their embeddings averaged."""
    release_accelerator_memory(device)
    if len(text) > MAX_CHARACTERS:
        text = text[:15000] + " ... " + text[-10000:]
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= MAX_SEQUENCE_TOKENS - DIRECT_ENCODE_MARGIN:
        return encode_with_cpu_fallback(model, text, device)
    return np.mean(encode_on_cpu(model, token_windows(model, tokens), device), axis=0)


def step_text(step):
    """Embedding input of one step: its sanitized action and observation."""
    return f"Action: {step.get('action', '')} | Result: {str(step.get('observation', ''))}"


def pooled_vector(series):
    """Pooled vector of a time-series matrix (steps x 768): first, middle and last step, mean and standard deviation."""
    spread = np.std(series, axis=0) if len(series) > 1 else np.zeros_like(series[0])
    return np.concatenate([series[0], series[len(series) // 2], series[-1], np.mean(series, axis=0), spread])


def embed_trajectory(record, model, device):
    """(time-series matrix, pooled vector) of a sanitized trajectory, both float32."""
    series = np.array([embed_text(model, step_text(step), device) for step in record["steps"]], dtype=np.float32)
    return series, pooled_vector(series)


def output_name(record, fallback_id):
    """File-name stem ``<OUTCOME>_<repository>_<trajectory id>`` of a trajectory's vectors."""
    trajectory_id = record.get("trajectory_id", fallback_id).lower()
    repository = record.get("repo_name", "unknown_repo").lower().replace("/", "_").replace(" ", "_")
    return f"{'SUCCESS' if record.get('final_outcome') else 'FAIL'}_{repository}_{trajectory_id}"


def embed_files(files, input_dir, output_dir, model, device):
    """Embed the given sanitized files; return the number of trajectories written."""
    written = 0
    for index, path in enumerate(files, start=1):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("steps"):
            print(f"  [{index}/{len(files)}] {path.name}: no steps, nothing written")
            continue
        started = time.time()
        series, pooled = embed_trajectory(record, model, device)
        run, name = path.relative_to(input_dir).parent, output_name(record, path.stem)
        for variant, suffix, array in (("timeseries", "ts", series), ("pooled", "pooled", pooled)):
            target = output_dir / variant / run / f"{name}_{suffix}.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, array)
        written += 1
        print(f"  [{index}/{len(files)}] {path.name}: {len(series)} steps, {time.time() - started:.1f} s")
    return written


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(config.DATASETS))
    parser.add_argument("--input", type=Path, default=None, help="sanitized folder (default: data/<dataset>/sanitized)")
    parser.add_argument("--output", type=Path, default=None, help="vector folder (default: data/<dataset>/vectors)")
    parser.add_argument("--files", type=Path, nargs="+", default=None, help="sanitized files below the input folder to embed (default: all)")
    parser.add_argument("--device", default=None, choices=("mps", "cpu"), help="default: mps when available, else cpu")
    arguments = parser.parse_args()
    input_dir = (arguments.input or config.stage_dir("sanitized", arguments.dataset)).resolve()
    output_dir = arguments.output or config.stage_dir("vectors", arguments.dataset)
    files = [path.resolve() for path in arguments.files] if arguments.files else sorted(input_dir.rglob("unified_*.json"))
    outside = [path for path in files if input_dir not in path.parents]
    if outside:
        raise SystemExit(f"embed: --files must lie below the sanitized folder {input_dir}: {outside[0]}")
    if not files:
        raise SystemExit(f"No sanitized files found in {input_dir}")
    device = arguments.device or default_device()
    print(f"embed: {len(files)} sanitized files from {input_dir}; model {MODEL_NAME}@{MODEL_REVISION[:8]} on {device}")
    written = embed_files(files, input_dir, output_dir, load_model(device), device)
    print(f"embed: wrote {written} time-series matrices and {written} pooled vectors to {output_dir}")


if __name__ == "__main__":
    main()
