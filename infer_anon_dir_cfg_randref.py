import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import soundfile

from inference import infer_tool_anon as infer_tool
from inference.infer_tool_anon import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")


def list_audio_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    suffixes = {f".{ext.lower().lstrip('.')}" for ext in extensions}
    if not suffixes:
        suffixes = {".wav"}
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def resolve_search_roots(input_dir: Path, subfolders: List[str]) -> List[Path]:
    if not subfolders:
        return [input_dir]
    roots = []
    for entry in subfolders:
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = (input_dir / candidate).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Subfolder '{candidate}' does not exist.")
        if not candidate.is_dir():
            raise NotADirectoryError(f"Subfolder '{candidate}' is not a directory.")
        try:
            candidate.relative_to(input_dir)
        except ValueError as err:
            raise ValueError(
                f"Subfolder '{candidate}' is not inside input_dir '{input_dir}'."
            ) from err
        roots.append(candidate)
    return roots


def write_wav_scp(entries: List[Tuple[str, Path]], output_dir: Path) -> None:
    scp_base = output_dir.parent if output_dir.name == "wav" else output_dir
    wav_scp_path = output_dir / "wav.scp"
    with wav_scp_path.open("w", encoding="utf-8") as handle:
        for utt_id, wav_path in sorted(entries, key=lambda item: item[0]):
            try:
                rel_path = wav_path.relative_to(scp_base)
                wav_ref = rel_path.as_posix()
            except ValueError:
                wav_ref = str(wav_path)
            handle.write(f"{utt_id} {wav_ref}\n")


def copy_reference_metadata(reference_dir: Path, output_dir: Path) -> None:
    for name in (
        "spk2gender",
        "spk2utt",
        "text",
        "trials",
        "utt2spk",
        "spk2fold",
        "utt2emo",
    ):
        src_path = reference_dir / name
        if not src_path.exists():
            logging.warning("Reference metadata missing: %s", src_path)
            continue
        shutil.copy2(src_path, output_dir / name)


def resolve_group_dir(output_dir: Path, relative: Path) -> Path:
    if output_dir.name == "wav":
        return output_dir.parent
    if len(relative.parts) > 1 and relative.parts[0] != "wav":
        return output_dir / relative.parts[0]
    return output_dir


def resolve_reference_dir(
    reference_root: Path,
    group_dir: Path,
    multi_group: bool,
) -> Optional[Path]:
    candidate = reference_root / group_dir.name
    if candidate.exists():
        return candidate
    has_root_files = any(
        (reference_root / name).exists()
        for name in (
            "spk2gender",
            "spk2utt",
            "text",
            "trials",
            "utt2spk",
            "spk2fold",
            "utt2emo",
        )
    )
    if has_root_files and not multi_group:
        return reference_root
    if has_root_files and multi_group:
        logging.warning(
            "Reference root %s does not contain %s; skipping metadata copy.",
            reference_root,
            group_dir.name,
        )
        return None
    logging.warning("Reference directory missing: %s", candidate)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description='ns2vc anonymous inference for a directory of wav files (CFG, random anon refs, FreeVC)'
    )
    parser.add_argument('-m', '--model_path', type=str, default="logs/model-127.pt",
                        help='Path to the model.')
    parser.add_argument('-c', '--config_path', type=str, default="config.json",
                        help='Path to the configuration file.')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory containing wav files to anonymize.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory where anonymized utterances are saved.')
    parser.add_argument('-wf', '--wav_format', type=str, default='wav',
                        help='Output format.')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device used for inference. None means auto selecting.')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,
                        help='F0 Filtering threshold, valid only when f0_mean_pooling is enabled.')
    parser.add_argument('--mode', type=str, default="null",
                        help='Controls which conditioning signals are used (null/resynt/anon/anon_pool/random/etc).')
    parser.add_argument(
        '-e', '--extensions',
        nargs='+',
        default=['wav'],
        help='List of file extensions to process (default: wav).',
    )
    parser.add_argument(
        '-s', '--subfolders',
        nargs='*',
        default=[],
        help='Optional list of subfolders (relative to input_dir) to process.',
    )
    parser.add_argument(
        '--reference_dir',
        type=str,
        default="exp/data/voiceprivacy",
        help='Root directory containing reference metadata to copy.',
    )
    parser.add_argument(
        '--anon_ref_root',
        type=str,
        default="exp/data/libritts_all_anon",
        help='Root directory containing anonymized reference wavs.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1337,
        help='Random seed for reproducibility.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip inference if the output file already exists.',
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anon_ref_root = Path(args.anon_ref_root)
    if not anon_ref_root.exists():
        raise FileNotFoundError(f"Anon reference root '{anon_ref_root}' does not exist.")

    if args.mode == "anon_pool" or args.mode == "anon_pool_null":
        pooled_files = list(anon_ref_root.rglob("*.freevcs.pt"))
        if not pooled_files:
            raise RuntimeError(f"No pooled embeddings found under {anon_ref_root}")
        anon_refs = []
    elif args.mode == "random" or args.mode == "random_null" or args.mode == "null_both":
        pass 
    else:
        anon_refs = list_audio_files(anon_ref_root, ['wav'])
        if not anon_refs:
            raise RuntimeError(f"No .wav files found under {anon_ref_root}")

    rng = random.Random(args.seed)

    roots = resolve_search_roots(input_dir, args.subfolders)
    audio_files = []
    seen = set()
    for root in roots:
        for path in list_audio_files(root, args.extensions):
            if path in seen:
                continue
            seen.add(path)
            audio_files.append(path)
    if not audio_files:
        logging.warning("No audio files found under %s", input_dir)
        return

    svc_model = Svc(args.model_path, args.config_path, args.device)
    cr_threshold = args.f0_filter_threshold
    wav_scp_entries: dict[Path, List[Tuple[str, Path]]] = {}

    for src in audio_files:
        raw_audio_path = str(src)
        relative = src.relative_to(input_dir)
        res_path = (output_dir / relative).with_suffix(f'.{args.wav_format}')
        if args.skip_existing and res_path.exists():
            group_dir = resolve_group_dir(output_dir, relative)
            wav_scp_entries.setdefault(group_dir, []).append((res_path.stem, res_path))
            continue
        infer_tool.format_wav(raw_audio_path)
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        info = soundfile.info(wav_path)
        duration = round(info.frames / info.samplerate, 3)
        print(f'#=====segment start, {duration}s======')

        if args.mode == "null":
            use_cond = [1, 0]
        elif args.mode == "resynt":
            use_cond = [1, 1]
        elif args.mode == "null_both":
            use_cond = [0, 0]
        elif args.mode == "anon":
            use_cond = [1, 1]
        elif args.mode == "anon_pool":
            use_cond = [1, 1]
        elif args.mode == "anon_pool_null":
            use_cond = [0,1]
        elif args.mode == "null_prosody":
            use_cond = [0, 1]
        elif args.mode == "random":
            use_cond = [1, 1]
        elif args.mode == "random_null":
            use_cond = [0, 1]
        else:
            use_cond = [1, 1]

        if args.mode == "anon_pool" or args.mode == "anon_pool_null":
            refer_wav_path = anon_ref_root
        elif args.mode == "resynt":
            refer_wav_path = wav_path
        elif args.mode in {"random","random_null","null_both"}:
            refer_wav_path = None
        else:
            ref_path = rng.choice(anon_refs)
            refer_wav_path = ref_path
 
        if refer_wav_path is not None and args.mode != "anon_pool" and args.mode != "anon_pool_null":
            infer_tool.format_wav(str(refer_wav_path))
            refer_wav_path = Path(str(refer_wav_path)).with_suffix('.wav')

        try:
            out_audio, out_sr = svc_model.infer_anon(
                wav_path,
                refer_wav_path,
                cr_threshold=cr_threshold,
                sampling='ddim_cfg',
                use_cond=use_cond,
                cfg_scales=[0.8,0],
            )
        except Exception as err:
            logging.exception("Inference failed for %s: %s", src, err)
            continue

        _audio = out_audio.cpu().numpy()
        res_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(res_path, _audio, svc_model.target_sample, format=args.wav_format)
        svc_model.clear_empty()
        group_dir = resolve_group_dir(output_dir, relative)
        wav_scp_entries.setdefault(group_dir, []).append((res_path.stem, res_path))

    if not wav_scp_entries:
        return

    multi_group = len(wav_scp_entries) > 1
    reference_root = Path(args.reference_dir)
    for group_dir in sorted(wav_scp_entries.keys()):
        write_wav_scp(wav_scp_entries[group_dir], group_dir)
        reference_dir = resolve_reference_dir(reference_root, group_dir, multi_group)
        if reference_dir is not None:
            copy_reference_metadata(reference_dir, group_dir)


if __name__ == '__main__':
    main()
