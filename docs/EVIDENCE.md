# Evidence ledger

Where each arranger parameter comes from. **Fitted** = derived from data by
`fit_corpus.py`. **Published** = a measurement in the cited work (read via search
summaries/abstracts where the full text was inaccessible). **Assumption** = standard
pedagogy or a judgement call that could not be verified; these are the first things to
tune by ear.

| Parameter | Value | Status | Source |
|---|---|---|---|
| Melody-note cost over a chord (non-chord tones) | 1.5 x -ln(P/Pmax) on strong beats, per chord quality | Fitted | Weimar Jazz Database, 456 solos (`data/empirical.json`) |
| Soloist eighth-note ratio | 0.86-1.14 by tempo bin, median 1.1 : 1 | Fitted | WJazzD, 22.5k eighth pairs. Consistent with Corcoran & Frieler (median ~1.3) |
| Chord-change rate target | slow tunes ~1.6 changes/bar, medium/up ~1.2-1.3 (band +-0.35) | Fitted | WJazzD chords per bar by tempo class |
| Chord-transition cost | -ln P(next chord \| previous) relative to the likeliest move | Fitted | iReal-derived JazzStandards.json (~47k transitions) |
| Ride / comping swing ratio | 3.3:1 near 125 bpm falling linearly to ~1.15:1 at 280 bpm, capped at 3.3 | Published | Dittmar, Pfleiderer & Mueller 2015 (921 ride excerpts); Friberg & Sundstrom. Ratio below ~120 bpm is not measured; the cap is a conservative choice |
| Soloist downbeat delay | ~30 ms on downbeats only, off-beats in sync | Published | Nature Comms Physics 2022 (s42005-022-00995-z) |
| Bass vs ride timing | within a few ms, no fixed lag | Published | Jazz trio timing study (PMC5706983): cymbal-bass +2.1 ms |
| Hi-hat pedal | ~15 ms ahead of the bass | Published | same study (7-26 ms) |
| Walking bass: root on chord changes | 68% | Published | FiloBass (arXiv 2311.02023) |
| Walking bass: bars of four quarter notes | 63% | Published | FiloBass |
| Approach notes | semitone from above 26.8%, from below 21.0%, whole step from below 11.9% | Published | FiloBass (renormalised among these three) |
| Sax vibrato rate | tenor 5.0 Hz, alto/soprano 6.0 Hz | Published (pedagogy) | Luckey via Pimentel: tenor 4.3-6, alto/soprano 5-6.7 Hz |
| Sax vibrato depth 35 cents, onset 0.35 s | | Assumption | general instrument vibrato +-50 cents; onset unsourced |
| Sax scoop / fall frequency, sizes | 50%, -150 / -250 cents | Assumption | no published rates found |
| Rootless voicing shapes (type A 3-5-7-9, type B 7-9-3-5) | | Published (pedagogy) | thejazzpianosite / piano.org |
| Voicing register 50-72 (MIDI) | | Assumption | qualitative sources give E3-G4 / C4-C5 thumb note |
| Comping hits per bar, Charleston frequency | | Assumption | no data found; the search chooses among styles per song |
| Fraction of chords reharmonised | target 15-40% | Assumption | no corpus study found; tuned only by the "interest" fitness term |
| Drum velocities, ghost-note rate (12%), feathered kick (70%) | | Assumption | qualitative sources only |
| Sax ranges (tenor Ab2-E5, alto Db3-Ab5 concert) | | Assumption | general knowledge |
| Fitness weights: consonance / plausibility / change-rate | 0.216 / 0.139 / 0.005 (a 0.36 pool split by fitted importances 0.60 / 0.385 / 0.013) | Fitted | Logistic regression, real WJazzD solos vs corrupted (melody shifted a semitone, chords shuffled); held-out AUC 0.93 (`fit_weights.py`). Chord-change-rate deviation carries almost no signal |
| Fitness weights: voice-leading 0.14, faithfulness 0.22, dynamics 0.10, texture 0.10, interest 0.08 | | Assumption | No ground truth exists for these; not fitted to listener ratings |
| Chord-smoothing strength | grid 0.01-0.2 chosen per song by fit to the sung melody | Data-driven | independent evidence: the vocal is not in the harmonic stem |
| Melody prior on chord scores | 0.04 per unit clash | Assumption | tie-breaker only |
| Meter | beats/bar = bar interval / beat interval; auto-switch needs 1.2x better bar-line harmonic contrast | Heuristic | validated on synthetic chroma only |
| Beat-grid regularisation | sparse stretches subdivided to the median beat | Heuristic | fixes beat_this switching density inside a song (seen on a real recording) |

## How to improve the weakest rows

1. **Remaining fitness weights** (voice-leading, faithfulness, dynamics, texture, interest):
   the harmony terms are now fitted (see above); these need listener ratings or a corpus of
   real arrangements (not just lead sheets) to fit.
2. **Bass / comping / drums statistics**: the Jazz Trio Database (bass and piano onsets vs
   the beat) and Filosax (sax note-level timing, vibrato, ornaments) contain the numbers;
   both need restricted downloads (Zenodo access request). Use their Lite/annotation
   releases to replace the assumption rows.
3. **Ballad swing (< 100 bpm)**: no measurements were found; the soloist fit has only 0.86
   at slow tempi, so straight-ish is the safest default.

## Datasets and downloads

| Dataset | URL | License |
|---|---|---|
| Weimar Jazz Database | https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db | ODbL 1.0 |
| JazzStandards (iReal-derived) | https://raw.githubusercontent.com/mikeoliphant/JazzStandards/master/JazzStandards.json | unstated; private statistics only |
| FiloBass | https://arxiv.org/abs/2311.02023 | paper (statistics quoted) |
| Jazz Trio Database | https://github.com/HuwCheston/Jazz-Trio-Database | annotations open |
| Filosax | https://github.com/dave-foster/filosax | non-commercial, restricted |

The raw databases are not committed (`data/*.db`, `data/JazzStandards.json` are
gitignored); only the small derived `data/empirical.json` is.
