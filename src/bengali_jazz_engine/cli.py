"""Command line interface: ``bengali-jazz-engine <command> [options]``.

    run       whole pipeline (the default: ``bengali-jazz-engine --solo`` == ``bengali-jazz-engine run --solo``)
    analyze   stages 1-4 (stems, melody, chords, acoustic, profile)
    arrange   stage 5 only
    render    stage 8 only        mix   stage 9 only        master   stage 10 only (needs --reference)
    mood      set / show the human mood sign-off for a song
    rate      listening tests: prepare candidate arrangements, record which you prefer
    vst       list / inspect plugins, check the role -> backend map
    corpus    refit the jazz statistics / fitness weights
    doctor    check the environment (dependencies, external tools, data files)
    info      workspace, songs and per-song status
    clean     delete caches / working files / stems / outputs
    lyrics    optional Whisper lyrics transcription of the vocal stem

Global options: ``--workdir DIR`` (workspace root), ``--version``, ``-v/--verbose``, ``-q/--quiet``.
Exit status: 0 ok, 1 runtime error, 2 usage error.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

from . import __version__, pipeline
from . import config as cfg

COMMANDS = ("run", "analyze", "arrange", "render", "mix", "master", "mood", "rate", "vst", "corpus", "doctor",
            "info", "clean", "lyrics")
QUALITIES = tuple(cfg.QUALITY_PRESETS)
LEADS = ("piano", "tenor_sax", "alto_sax", "soprano_sax")


# ---- argument groups ---------------------------------------------------------

def _add_input(p):
    g = p.add_argument_group("input")
    g.add_argument("--input", metavar="FILE", help="audio file to process (default: the only file in input/)")
    g.add_argument("--all", dest="all_songs", action="store_true",
                   help="process every audio file in input/, one after another (a failure does not stop the batch)")


def _add_analysis(p):
    g = p.add_argument_group("analysis")
    g.add_argument("--meter", type=int, choices=(2, 3, 4, 6), help="force beats per bar (default: tracked and checked)")
    g.add_argument("--tempo-scale", type=float, metavar="X",
                   help="0.5 = the song is felt at half the tracked tempo (slow ballad), 2 = double time")
    g.add_argument("--force", action="store_true", help="ignore cached stems / melody / chords and recompute")


def _add_arrange(p):
    g = p.add_argument_group("arrangement")
    band = g.add_mutually_exclusive_group()
    band.add_argument("--full-band", dest="band", action="store_const", const="trio",
                      help="force piano + bass + drums (default: decided from the song)")
    band.add_argument("--solo", dest="band", action="store_const", const="solo",
                      help="force a solo (lead + piano comping only)")
    g.add_argument("--lead", choices=LEADS, help="force the lead instrument (default: decided from the song)")
    g.add_argument("--seed", type=int, help="RNG seed for a reproducible arrangement (default 0)")
    g.add_argument("--pop-size", type=int, metavar="N", help="genomes per generation (default from --quality)")
    g.add_argument("--generations", type=int, metavar="N", help="search generations (default from --quality)")


def _add_speed(p):
    g = p.add_argument_group("speed / accuracy")
    g.add_argument("--quality", choices=QUALITIES,
                   help="fast = single htdemucs model + small search; balanced (default); best = 2 shifts + bigger search")
    g.add_argument("--jobs", type=int, metavar="N", help="parallel render / mix workers (default: min(4, CPUs))")
    g.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="device for Demucs and beat_this (default auto)")


def _add_output(p):
    g = p.add_argument_group("output")
    g.add_argument("--output-dir", metavar="DIR", help="copy results here instead of output/<song>/ (one song only)")
    g.add_argument("--reference", metavar="WAV", help="reference track: also master the mix against it (matchering)")
    g.add_argument("--json", action="store_true", help="print the result as JSON on stdout (progress goes to stderr)")


def _add_range(p):
    g = p.add_argument_group("stage range")
    g.add_argument("--from", dest="start", choices=pipeline.STEP_KEYS, metavar="STAGE", help="first stage to run")
    g.add_argument("--to", dest="stop", choices=pipeline.STEP_KEYS, metavar="STAGE", help="last stage to run")
    g.add_argument("--only", choices=pipeline.STEP_KEYS, metavar="STAGE", help="run just this stage")
    g.add_argument("--dry-run", action="store_true", help="show which stages would run and exit")


def build_parser():
    ap = argparse.ArgumentParser(prog="bengali-jazz-engine", description=__doc__.split("\n\n")[0],
                                 epilog="Run 'bengali-jazz-engine <command> -h' for the options of a command.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--workdir", metavar="DIR", help="workspace root (input/, stems/, work/, output/, soundfonts/, tools/, data/)")
    ap.add_argument("--no-per-song", action="store_true", help="use one shared working directory instead of work/<song>/")
    noise = ap.add_mutually_exclusive_group()
    noise.add_argument("-v", "--verbose", action="store_true", help="show tracebacks")
    noise.add_argument("-q", "--quiet", action="store_true", help="only print the final result")
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    def stage_cmd(name, help_, *groups, only=None):
        p = sub.add_parser(name, help=help_, description=help_)
        for g in groups:
            g(p)
        p.set_defaults(_only=only)
        return p

    stage_cmd("run", "run the whole pipeline (default command)", _add_input, _add_analysis, _add_arrange,
              _add_speed, _add_range, _add_output)
    p = stage_cmd("analyze", "stages 1-4: stems, melody, chords/beats/key, acoustic, song profile", _add_input,
                  _add_analysis, _add_speed)
    p.set_defaults(_stop="profile")
    stage_cmd("arrange", "stage 5: generate, score and refine the arrangement", _add_input, _add_arrange, _add_speed,
              only="arrange")
    stage_cmd("render", "stage 8: render the arrangement MIDI to stems", _add_input, _add_speed, only="render")
    stage_cmd("mix", "stage 9: mix the rendered stems to mix/rough_mix.wav", _add_input, _add_speed, _add_output,
              only="mix")
    p = stage_cmd("master", "stage 10: master the mix against a reference track", _add_input, _add_output, only="master")

    p = sub.add_parser("mood", help="set / show the human mood sign-off for a song")
    m = p.add_subparsers(dest="mood_cmd", metavar="<action>")
    s = m.add_parser("set", help="save a mood keyword for a song")
    s.add_argument("keyword", choices=cfg.MOOD_KEYWORDS)
    s.add_argument("why", help="one-line justification")
    s.add_argument("--song", help="song name (default: the file in input/)")
    m.add_parser("show", help="show the saved mood").add_argument("--song")
    m.add_parser("list", help="list the allowed mood keywords")

    p = sub.add_parser("rate", help="listening tests: candidate arrangements and pairwise preferences")
    t = p.add_subparsers(dest="rate_cmd", metavar="<action>")
    s = t.add_parser("prepare", help="generate candidate arrangements of a song (needs a finished analysis)")
    s.add_argument("--song", help="song name (default: the file in input/)")
    s.add_argument("--n", type=int, default=4, help="number of candidates (default 4)")
    s.add_argument("--render", action="store_true", help="also render each candidate to a wav to listen to")
    s.add_argument("--seed", type=int)
    s = t.add_parser("add", help="record a preference: candidate A vs candidate B")
    s.add_argument("a", type=int)
    s.add_argument("b", type=int)
    s.add_argument("winner", choices=("a", "b", "tie"), help="which one you preferred")
    s.add_argument("--song")
    s.add_argument("--note", default="")
    t.add_parser("status", help="how many ratings are recorded")

    p = sub.add_parser("vst", help="plugin discovery and the role -> backend map")
    v = p.add_subparsers(dest="vst_cmd", metavar="<action>")
    v.add_parser("list", help="list installed VST3 plugins")
    i = v.add_parser("inspect", help="is this plugin an instrument, and its parameters")
    i.add_argument("path")
    i.add_argument("--plugin-name")
    v.add_parser("check", help="resolve every role to its backend and check the files exist")

    p = sub.add_parser("corpus", help="refit jazz statistics / fitness weights from the corpora")
    c = p.add_subparsers(dest="corpus_cmd", metavar="<action>")
    c.add_parser("fit-stats", help="refit data/empirical.json").add_argument("--download", action="store_true")
    c.add_parser("fit-weights", help="refit the harmony fitness weights")
    fr = c.add_parser("fit-ratings", help="fit the fitness weights from recorded listener ratings")
    fr.add_argument("--min", type=int, default=30, help="minimum number of ratings (default 30)")
    fr.add_argument("--force", action="store_true", help="fit even with fewer ratings")
    c.add_parser("fit-jtd", help="fit rhythm-section statistics from the Jazz Trio Database annotations").add_argument(
        "--download", action="store_true")

    d = sub.add_parser("doctor", help="check dependencies, external tools and data files")
    d.add_argument("--json", action="store_true")

    n = sub.add_parser("info", help="workspace, songs and per-song status")
    n.add_argument("--json", action="store_true")
    n.add_argument("--song", help="show one song in detail")

    c = sub.add_parser("clean", help="delete caches / working files / stems / outputs (dry run unless --yes)")
    for flag, text in (("--cache", "stage cache and snapshots"), ("--work", "work/<song>/ working files"),
                       ("--stems", "separated stems (slow to recreate!)"), ("--output", "output/ results"),
                       ("--all", "everything above")):
        c.add_argument(flag, action="store_true", help=text)
    c.add_argument("--song", help="only this song's work/stems/output")
    c.add_argument("--yes", action="store_true", help="really delete")

    sub.add_parser("lyrics", help="Whisper transcription of the vocal stem (needs the 'lyrics' extra)")
    return ap


# ---- command implementations ---------------------------------------------------

def _options(args, only=None, stop=None):
    return pipeline.RunOptions(
        input_file=getattr(args, "input", None), all_songs=getattr(args, "all_songs", False),
        reference=getattr(args, "reference", None), band=getattr(args, "band", None),
        lead=getattr(args, "lead", None), seed=getattr(args, "seed", None), force=getattr(args, "force", False),
        meter=getattr(args, "meter", None), tempo_scale=getattr(args, "tempo_scale", None),
        quality=getattr(args, "quality", None), pop_size=getattr(args, "pop_size", None),
        generations=getattr(args, "generations", None), jobs=getattr(args, "jobs", None),
        device=getattr(args, "device", None), start=getattr(args, "start", None),
        stop=getattr(args, "stop", None) or stop, only=getattr(args, "only", None) or only,
        output_dir=getattr(args, "output_dir", None), dry_run=getattr(args, "dry_run", False))


def cmd_pipeline(args):
    opts = _options(args, only=getattr(args, "_only", None), stop=getattr(args, "_stop", None))
    if opts.all_songs and opts.output_dir:
        raise ValueError("--output-dir cannot be combined with --all")
    if opts.only == "master" and not opts.reference:
        raise ValueError("master needs --reference WAV")
    result = pipeline.run(opts)
    return {"result": result if isinstance(result, dict) else (str(result) if result else None)}


def cmd_mood(args):
    from . import mood

    if args.mood_cmd == "set":
        mood.run(args.keyword, args.why, args.song)
        return {"mood": args.keyword}
    if args.mood_cmd == "show":
        data = mood.show(args.song)
        print(json.dumps(data, indent=2) if data else "no mood saved for this song")
        return {"mood": data}
    if args.mood_cmd == "list":
        print("\n".join(cfg.MOOD_KEYWORDS))
        return {"moods": list(cfg.MOOD_KEYWORDS)}
    raise ValueError("mood needs an action: set | show | list")


def cmd_rate(args):
    from . import rate

    if args.rate_cmd == "status":
        info = rate.status()
        print(f"{info['ratings']} ratings ({info['decisive']} decisive) from {len(info['songs'])} song(s) in {info['file']}")
        return info
    if args.rate_cmd in ("prepare", "add"):
        cfg.use_song(args.song or cfg.find_input_audio().stem)
        if args.rate_cmd == "prepare":
            return {"candidates": rate.prepare(args.n, args.render, args.seed)}
        return {"recorded": rate.add(args.a, args.b, args.winner, args.note)}
    raise ValueError("rate needs an action: prepare | add | status")


def cmd_vst(args):
    from .render import vst

    if args.vst_cmd == "list":
        found = vst.discover()
        print(f"Config: {vst.config_path()} ({'found' if vst.config_path().exists() else 'missing'})")
        for p in found:
            print(f"  {p['name']:40s} {p['path']}")
        if not found:
            print("  (no VST3 plugins found)")
        return {"plugins": found}
    if args.vst_cmd == "inspect":
        info = vst.inspect(args.path, args.plugin_name)
        print(f"{info['path']}\n  instrument: {info['is_instrument']}\n  parameters: {', '.join(info['parameters'])}")
        return info
    if args.vst_cmd == "check":
        rows = {}
        for role in ("piano", "comp", "tenor_sax", "alto_sax", "soprano_sax", "bass", "drums"):
            spec = vst.plugin_for(role)
            kind = "GM soundfont (default)" if not spec else ("SFZ" if "sfz" in spec else "SF2" if "sf2" in spec else "VST3")
            rows[role] = {"backend": kind, "spec": spec}
            print(f"  {role:12s} {kind}" + (f"  {spec.get('sfz') or spec.get('sf2') or spec.get('path')}" if spec else ""))
        return rows
    raise ValueError("vst needs an action: list | inspect | check")


def cmd_corpus(args):
    if args.corpus_cmd == "fit-stats":
        from .corpus import fit_stats

        fit_stats.run(do_download=args.download)
        return {"wrote": str(cfg.PACKAGE_DATA / "empirical.json")}
    if args.corpus_cmd == "fit-weights":
        from .corpus import fit_weights

        fit_weights.run()
        return {"wrote": str(cfg.PACKAGE_DATA / "fitted_weights.json")}
    if args.corpus_cmd == "fit-ratings":
        from .corpus import ratings

        return {"fit": ratings.run(min_ratings=args.min, force=args.force)}
    if args.corpus_cmd == "fit-jtd":
        from .corpus import fit_jtd

        fit_jtd.run(do_download=args.download)
        return {"wrote": str(cfg.PACKAGE_DATA / "jtd.json")}
    raise ValueError("corpus needs an action: fit-stats | fit-weights | fit-jtd | fit-ratings")


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def doctor_report():
    """[{name, status: ok|warn|missing, detail, required}] - what is installed and usable."""
    from .render import vst

    rows = []

    def add(name, ok, detail="", required=True):
        rows.append({"name": name, "status": "ok" if ok else ("missing" if required else "warn"),
                     "detail": str(detail), "required": required})

    add("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0])
    for mod in ("numpy", "scipy", "librosa", "pretty_midi", "mido", "pedalboard"):
        add(f"python: {mod}", _has(mod))
    for mod, why in (("demucs", "stem separation"), ("beat_this", "beat/downbeat tracking (else librosa fallback)"),
                     ("matchering", "mastering"), ("pandas", "corpus fitting"), ("music21", "key fallback"),
                     ("mir_eval", "melody tests")):
        add(f"python: {mod}", _has(mod), why, required=mod in ("demucs",))
    add("ffmpeg", shutil.which("ffmpeg"), shutil.which("ffmpeg") or "needed to decode mp3/m4a", required=False)
    try:
        fs = cfg.require_fluidsynth()
    except FileNotFoundError as exc:
        fs, err = None, str(exc)
    add("fluidsynth", fs, fs or err)
    try:
        sfz = vst.find_sfizz_render()
    except FileNotFoundError:
        sfz = None
    add("sfizz_render", sfz, sfz or "only needed for SFZ roles", required=False)
    add("soundfont", cfg.SOUNDFONT.exists(), cfg.SOUNDFONT)
    add("empirical.json", (cfg.PACKAGE_DATA / "empirical.json").exists(), cfg.PACKAGE_DATA / "empirical.json", required=False)
    add("fitted_weights.json", (cfg.PACKAGE_DATA / "fitted_weights.json").exists(), "", required=False)
    add("vst.json", vst.config_path().exists(), vst.config_path(), required=False)
    add("compute device", True, cfg.resolve_device())
    return rows


def cmd_doctor(args):
    rows = doctor_report()
    if not args.json:
        for r in rows:
            mark = {"ok": "ok     ", "warn": "warn   ", "missing": "MISSING"}[r["status"]]
            print(f"  [{mark}] {r['name']:24s} {r['detail']}")
    bad = [r for r in rows if r["status"] == "missing"]
    if not args.json:
        print("\nAll required components found." if not bad else f"\n{len(bad)} required component(s) missing.")
    return {"checks": rows, "ok": not bad}, (0 if not bad else 1)


def song_status(name):
    base = cfg.WORK_DIR / name
    stem_dir = cfg.STEMS_DIR / cfg.DEMUCS_MODEL / name
    flags = {
        "stems": all((stem_dir / f"{s}.wav").exists() for s in ("vocals", "bass", "drums", "other")),
        "melody": (base / "midi" / "melody_raw_expressive.mid").exists(),
        "chords": (base / "analysis" / "chord_estimate.json").exists(),
        "profile": (base / "analysis" / "song_profile.json").exists(),
        "arrangement": (base / "analysis" / "arrangement_report.json").exists(),
        "render": (base / "render" / "melody.wav").exists(),
        "mix": (base / "mix" / "rough_mix.wav").exists(),
        "output": (cfg.OUTPUT_DIR / name).exists(),
    }
    detail = {}
    prof = base / "analysis" / "song_profile.json"
    if prof.exists():
        p = json.loads(prof.read_text())
        detail.update(tempo_bpm=round(p["tempo_bpm"], 1), beats_per_bar=p["beats_per_bar"], key=p["key"],
                      mood=p["mood"], instrumentation=p["instrumentation"]["lead"] + "/" + p["instrumentation"]["band"])
    rep = base / "analysis" / "arrangement_report.json"
    if rep.exists():
        detail["fitness"] = json.loads(rep.read_text())["fitness"]["score"]
    return {"stages": flags, **detail}


def cmd_info(args):
    songs = [p.stem for p in cfg.list_input_audio()]
    if cfg.WORK_DIR.exists():
        songs += [d.name for d in cfg.WORK_DIR.iterdir() if d.is_dir() and d.name not in songs]
    songs = [args.song] if args.song else sorted(songs)
    data = {"version": __version__, "workspace": str(cfg.ROOT), "quality": cfg.SETTINGS["quality"],
            "demucs_model": cfg.DEMUCS_MODEL, "device": cfg.resolve_device(), "seed": cfg.SEED,
            "songs": {s: song_status(s) for s in songs}}
    if not args.json:
        print(f"bengali-jazz-engine {__version__}   workspace {cfg.ROOT}")
        print(f"quality {data['quality']}  demucs {data['demucs_model']}  device {data['device']}  seed {cfg.SEED}")
        for s, st in data["songs"].items():
            done = [k for k, v in st["stages"].items() if v]
            extra = "  " + ", ".join(f"{k}={v}" for k, v in st.items() if k != "stages")
            print(f"  {s}: {', '.join(done) or 'nothing yet'}{extra if len(st) > 1 else ''}")
        if not songs:
            print(f"  (no songs: put audio in {cfg.INPUT_DIR})")
    return data


def cmd_clean(args):
    root = cfg.ROOT.resolve()
    targets = []
    song = args.song
    if args.cache or args.all:
        targets += [cfg.CACHE_FILE, cfg.BASE_ANALYSIS_DIR / ".cache_files"]
    if args.work or args.all:
        targets.append(cfg.WORK_DIR / song if song else cfg.WORK_DIR)
    if args.stems or args.all:
        targets.append(cfg.STEMS_DIR / cfg.DEMUCS_MODEL / song if song else cfg.STEMS_DIR)
    if args.output or args.all:
        targets.append(cfg.OUTPUT_DIR / song if song else cfg.OUTPUT_DIR)
    if not targets:
        raise ValueError("nothing selected: pass --cache, --work, --stems, --output or --all")
    removed = []
    for t in targets:
        t = Path(t)
        if not t.exists():
            continue
        if root not in t.resolve().parents:
            raise ValueError(f"refusing to delete outside the workspace: {t}")
        print(f"{'deleting' if args.yes else 'would delete'} {t}")
        if args.yes:
            shutil.rmtree(t) if t.is_dir() else t.unlink()
        removed.append(str(t))
    if not args.yes and removed:
        print("(dry run - pass --yes to delete)")
    return {"removed" if args.yes else "would_remove": removed}


def cmd_lyrics(args):
    from .analysis import lyrics

    return {"transcript": str(lyrics.run())}


DISPATCH = {"run": cmd_pipeline, "analyze": cmd_pipeline, "arrange": cmd_pipeline, "render": cmd_pipeline,
            "mix": cmd_pipeline, "master": cmd_pipeline, "mood": cmd_mood, "rate": cmd_rate, "vst": cmd_vst, "corpus": cmd_corpus,
            "doctor": cmd_doctor, "info": cmd_info, "clean": cmd_clean, "lyrics": cmd_lyrics}


def normalize_argv(argv):
    """`bengali-jazz-engine --solo` (no command) means `run --solo`; global options may come first."""
    argv = list(argv)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in COMMANDS:
            return argv
        if a in ("-h", "--help", "--version"):
            return argv
        if a == "--workdir":
            i += 2
        elif a.startswith("--workdir=") or a in ("--no-per-song", "-v", "--verbose", "-q", "--quiet"):
            i += 1
        else:
            break
    return argv[:i] + ["run"] + argv[i:]


def main(argv=None):
    argv = normalize_argv(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.command is None:
        ap.print_help()
        return 2
    if args.workdir:
        cfg.set_root(args.workdir)
    if args.no_per_song:
        cfg.set_per_song(False)
    want_json = getattr(args, "json", False)
    code = 0
    try:
        stream = sys.stderr if want_json else (io.StringIO() if args.quiet else sys.stdout)
        with contextlib.redirect_stdout(stream):
            out = DISPATCH[args.command](args)
        if isinstance(out, tuple):
            out, code = out
        if want_json:
            print(json.dumps(out, indent=2, default=str))
        elif args.quiet and out and out.get("result"):
            print(out["result"])
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2 if isinstance(exc, ValueError) else 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return code


if __name__ == "__main__":
    sys.exit(main())
