"""The pipeline: ordered stages, run per song, with selectable ranges and a run manifest.

    run(RunOptions(...))            one song (or every file in input/ with options.all_songs)

Stages, in order (``STEP_KEYS``): separate, melody, chords, acoustic, profile, arrange,
render, mix, master. ``start`` / ``stop`` / ``only`` choose a range; everything a stage needs
is read back from the song's working directory, so ``render`` or ``mix`` can be re-run alone.

Mood detection (lyrics + web context) needs a human/LLM in the loop and cannot be scripted:
if analysis/mood.json exists for this song (written by ``mood set``) it is used, otherwise
the profile falls back to an acoustic guess automatically.

Mastering only runs if a reference is given. Results are copied to output/<song>/ together
with a manifest.json (versions, settings, seed, stage timings).
"""
import dataclasses
import json
import platform
import shutil
import time
from pathlib import Path

from . import __version__, feedback, hardware, logs, memory
from . import config as cfg
from .analysis import acoustic, chords, melody, profile, score, separate
from .arrange import arranger
from .render import master, mix, stems

STEP_KEYS = ("separate", "melody", "chords", "acoustic", "score", "profile", "arrange", "render", "mix", "master")
AUDIO_ONLY = ("separate", "melody", "chords", "acoustic")   # replaced by `score` when the input is sheet music / MIDI
LABELS = {
    "separate": "1  Stem separation",
    "melody": "2  Melody extraction (pYIN)",
    "chords": "3c Chord + beat + key detection",
    "acoustic": "3  Acoustic analysis",
    "score": "1-4 Read the score (melody, chords, tempo, meter, key)",
    "profile": "4  Understand song + choose instrumentation",
    "arrange": "5  Arrange: generate -> score -> refine",
    "render": "8  Render (VST3 / SFZ / SF2 / fluidsynth)",
    "mix": "9  Mix",
    "master": "10 Mastering",
}


@dataclasses.dataclass
class RunOptions:
    input_file: str | None = None
    all_songs: bool = False
    reference: str | None = None
    band: str | None = None            # None = let the song decide, "solo" or "trio"
    lead: str | None = None            # None = let the song decide
    seed: int | None = None
    force: bool = False
    meter: int | None = None
    tempo_scale: float | None = None
    quality: str | None = None
    pop_size: int | None = None
    generations: int | None = None
    jobs: int | None = None
    device: str | None = None
    start: str | None = None
    stop: str | None = None
    only: str | None = None
    output_dir: str | None = None
    dry_run: bool = False
    score_part: str | None = None      # sheet-music input: melody part (name fragment or index)
    feedback: bool | None = None       # ask the after-run questions: None = only on an interactive terminal
    memory: bool = True                # use and update the memory (remembered settings, preferences, similar songs)


def select_steps(start=None, stop=None, only=None, reference=None, kind="audio"):
    """Stage keys to run. Master runs only when a reference is given (or is asked for explicitly). `kind` is
    "audio" (stem separation, melody, chords, acoustic) or "score" (one `score` stage instead)."""
    for key in (start, stop, only):
        if key is not None and key not in STEP_KEYS:
            raise ValueError(f"unknown stage {key!r}; choose from {', '.join(STEP_KEYS)}")
    if only:
        return [only]
    i = STEP_KEYS.index(start) if start else 0
    j = STEP_KEYS.index(stop) if stop else len(STEP_KEYS) - 1
    if i > j:
        raise ValueError(f"--from {start} comes after --to {stop}")
    chosen = [k for k in STEP_KEYS[i:j + 1] if (k != "score" if kind == "audio" else k not in AUDIO_ONLY)]
    if "master" in chosen and not reference and stop != "master":
        chosen.remove("master")
    return chosen


def apply_options(opts: RunOptions) -> None:
    """Push the options into config (seed, overrides, quality, jobs, device)."""
    if opts.seed is not None:
        cfg.set_seed(opts.seed)
    if opts.quality:
        cfg.set_quality(opts.quality)
    for key in ("pop_size", "generations", "jobs", "device"):
        value = getattr(opts, key)
        if value is not None:
            cfg.SETTINGS[key] = value
    cfg.SETTINGS["lead"] = opts.lead
    cfg.SETTINGS["score_part"] = opts.score_part
    cfg.SETTINGS["memory"] = opts.memory
    cfg.OVERRIDES["meter"] = opts.meter if opts.meter is not None else cfg.OVERRIDES["meter"]
    cfg.OVERRIDES["tempo_scale"] = opts.tempo_scale if opts.tempo_scale is not None else cfg.OVERRIDES["tempo_scale"]


def _band_is_trio(forced):
    if forced:
        return forced == "trio"
    report = cfg.ANALYSIS_DIR / "arrangement_report.json"
    if not report.exists():
        raise FileNotFoundError(f"{report} missing - run the arrange stage first")
    return json.loads(report.read_text())["instrumentation"]["band"] == "trio"


def collect_outputs(audio, mastered=False, output_dir=None, timings=None, opts=None):
    """Copy this song's final mix, arrangement MIDI, reports and a manifest to output/<song>/."""
    dest = Path(output_dir) if output_dir else cfg.OUTPUT_DIR / audio.stem
    dest.mkdir(parents=True, exist_ok=True)
    src = cfg.MIX_DIR / ("final_master.wav" if mastered else "rough_mix.wav")
    final = dest / f"{audio.stem} - jazz.wav"
    shutil.copyfile(src, final)
    for name in ("arrangement_report.json", "song_profile.json", "chord_estimate.json"):
        if (cfg.ANALYSIS_DIR / name).exists():
            shutil.copyfile(cfg.ANALYSIS_DIR / name, dest / name)
    for name in ("melody_lead", "chords", "bass", "drums"):
        if (cfg.MIDI_DIR / f"{name}.mid").exists():
            shutil.copyfile(cfg.MIDI_DIR / f"{name}.mid", dest / f"{name}.mid")
    manifest = {
        "song": audio.stem, "version": __version__, "python": platform.python_version(),
        "seed": cfg.SEED, "quality": cfg.SETTINGS["quality"], "demucs_model": cfg.DEMUCS_MODEL,
        "device": cfg.resolve_device(), "device_reason": cfg.device_reason(), "jobs": cfg.n_jobs(),
        "hardware": hardware.fingerprint(hardware.detect(), cfg.resolve_device()),
        "settings": {k: cfg.setting(k) for k in ("demucs_shifts", "pop_size", "generations", "sfizz_quality")},
        "overrides": {k: v for k, v in cfg.OVERRIDES.items() if k != "input"},
        "options": dataclasses.asdict(opts) if opts else {},
        "stage_seconds": {k: round(v, 2) for k, v in (timings or {}).items()},
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return final


def run_song(audio, opts: RunOptions):
    """Run the selected stages for one audio file; returns the final wav (or None if it was not produced)."""
    score.check_supported(audio)
    cfg.OVERRIDES["input"] = str(audio)
    cfg.use_song(Path(audio).stem)
    if memory.enabled() and Path(audio).exists():
        applied = memory.apply_song_settings(opts, cfg.file_sha256(Path(audio)))
        if applied:
            apply_options(opts)
            print("Remembered settings for this recording: " + ", ".join(f"{k}={v}" for k, v in applied.items()))
            logs.event("remembered_settings", **applied)
    cfg.ensure_dirs()
    if opts.force and cfg.CACHE_FILE.exists():
        cfg.CACHE_FILE.unlink()
    steps = select_steps(opts.start, opts.stop, opts.only, opts.reference, "score" if score.is_score(audio) else "audio")
    if opts.dry_run:
        print(f"{Path(audio).name}: would run {', '.join(steps)}")
        return None

    band = opts.band
    timings = {}
    logs.context(song=Path(audio).stem)
    for key in steps:
        t0 = time.time()
        logs.context(stage=key)
        logs.event("stage_start", stage_key=key)
        print(f"\n=== {LABELS[key]} ===")
        if key == "separate":
            separate.run()
        elif key == "melody":
            melody.run()
        elif key == "chords":
            chords.run()
        elif key == "acoustic":
            acoustic.run()
        elif key == "score":
            score.run()
        elif key == "profile":
            profile.run()
        elif key == "arrange":
            band = arranger.run(forced_band=band, forced_lead=opts.lead)["instrumentation"]["band"]
        elif key == "render":
            stems.run(full_band=_band_is_trio(band))
        elif key == "mix":
            mix.run(full_band=_band_is_trio(band))
        elif key == "master":
            if not opts.reference:
                raise ValueError("the master stage needs --reference")
            master.run(opts.reference)
        timings[key] = time.time() - t0
        logs.event("stage_end", stage_key=key, seconds=round(timings[key], 2))
        print(f"    ({timings[key]:.1f}s)")
    logs.context(stage=None)

    if "mix" in steps or "master" in steps:
        final = collect_outputs(Path(audio), mastered="master" in steps, output_dir=opts.output_dir,
                                timings=timings, opts=opts)
        print(f"\nDone: {final}  (total {sum(timings.values()):.1f}s)")
        _learn(Path(audio), timings, opts)
        return final
    return None


def _learn(audio, timings, opts):
    """Record the finished run in the memory and, on an interactive terminal, ask the after-run questions."""
    if not memory.enabled():
        return
    try:
        profile = json.loads((cfg.ANALYSIS_DIR / "song_profile.json").read_text())
        report = json.loads((cfg.ANALYSIS_DIR / "arrangement_report.json").read_text())
    except (OSError, ValueError):
        return
    row = memory.record_run(audio, profile, report, timings, opts, hardware.fingerprint(hardware.detect(), cfg.resolve_device()))
    logs.event("run_recorded", run_id=row["run_id"], fitness=row["fitness"].get("score"))
    if feedback.interactive_allowed(opts.feedback):
        try:
            feedback.ask(row)
        except (EOFError, KeyboardInterrupt):
            print("(feedback skipped)")


def run(opts: RunOptions):
    """Run for one song, or for every file in input/ when `all_songs` (a failure does not stop the batch)."""
    apply_options(opts)
    cfg.ensure_dirs()
    if logs.current_run() is None:
        logs.new_run()
    hw = hardware.detect()
    device = cfg.resolve_device()
    logs.event("run_start", options=dataclasses.asdict(opts), device=device, device_reason=cfg.device_reason(),
               hardware=hardware.summary(hw, device, cfg.device_reason()), version=__version__)
    started = time.time()
    status = "failed"
    try:
        result = _run_all(opts)
        status = "ok"
        return result
    finally:
        logs.event("run_end", status=status, seconds=round(time.time() - started, 2))


def _run_all(opts: RunOptions):
    if opts.all_songs:
        results = {}
        for song in cfg.list_input_audio():
            print(f"\n########## {song.name} ##########")
            try:
                results[song.name] = str(run_song(song, opts))
            except Exception as exc:  # noqa: BLE001 - keep going: one bad song must not stop the batch
                print(f"FAILED {song.name}: {exc!r}")
                results[song.name] = f"FAILED: {exc!r}"
        print("\n=== Batch summary ===")
        for name, res in results.items():
            print(f"  {name}: {res}")
        return results
    audio = Path(opts.input_file) if opts.input_file else cfg.find_input_audio()
    if not audio.exists():
        raise FileNotFoundError(f"Input file not found: {audio}")
    return run_song(audio, opts)
