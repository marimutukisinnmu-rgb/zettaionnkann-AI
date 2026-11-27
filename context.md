# CONTEXT EXPORT

**User-facing context — 2026-11-27**

This document records user-shared preferences, development environment details, project context, prior results, and the current work state. Some hardware/software details may be outdated or may refer to different machines. Historical benchmark results are attributed to the reports in which they were shared and are not newly verified.

## 1. User profile and preferences

### General
- Preferred name: ぴか
- Role: Student
- Likes programming and technical experimentation.
- Primarily uses Windows desktop and Chrome.
- Prefers CLI over GUI.
- Frequently works with Python, JavaScript, PowerShell, MIDI, HTML, and GitHub.
- Enjoys experimenting with unusual edge cases, performance limits, and intentionally strange program behavior.

### Response preferences
- Use Japanese by default.
- English technical identifiers and code are fine.
- Use normal fenced code blocks for code, not WritingBlocks, unless explicitly requested.
- 「ぜんこーどおねがい」 means 「フルコードお願い」.
- The term ump means Ultralight MIDIPlayer, not MIDI 2.0.
- Do not suggest PowerShell -ErrorAction SilentlyContinue unless explicitly permitted.
- Do not suggest forced shutdown using /f, because OBS recordings or unsaved work may be running.
- Do not generate images unless explicitly requested.
- Preferred assistant persona name: ちゃっぴー.
- Likes confident, enthusiastic responses and occasional humor.

### Recurring interests
- Black MIDI
- Touhou music
- Scratch / TurboWarp
- MIDI processing and playback
- Audio analysis and DSP
- Performance testing
- Python tooling
- GitHub automation
- HTML/JavaScript applications

## 2. Computer and development environment

These are details mentioned in prior conversations. Some may be outdated or refer to different machines.

### Hardware
- RAM: 16 GB reported in more recent logs.
- GPU references:
  - NVIDIA GeForce GTX 960
  - AMD Radeon Vega 8 Graphics
- CPU references:
  - Intel Core i7-6700 @ 3.40 GHz
  - AMD Ryzen 5 with Radeon Vega Mobile Graphics
- Storage:
  - C: SSD
  - E: HDD
- Large MIDI and video collections stored on E:.

### Operating systems
- Windows 10 Home 22H2 (build 19045) mentioned.
- Windows 11 upgrade experiments mentioned.
- WSL with Ubuntu 22.04.5 LTS mentioned.
- WSL initially used version 1.

### Languages and tools
- Python: versions 3.12.3 and 3.14.6 mentioned.
- Node.js / JavaScript
- Java with LWJGL
- C#
- PowerShell
- HTML / CSS
- Git and GitHub
- OBS Studio
- FFmpeg
- MuseScore
- Polyphone
- CapCut
- Canva
- TurboWarp

### Python libraries mentioned
- mido
- pyftpdlib
- Pillow
- pygame
- requests
- obsws-python
- pytesseract
- numpy
- librosa
- scikit-learn
- joblib

### FFmpeg
- FFmpeg 8.0 essentials build mentioned.
- NVIDIA encoding experiments: hevc_nvenc and av1_nvenc.
- CUDA version 12.6 appeared in an FFmpeg environment log.

### Other technical context
- OBS Studio and obs-websocket 5.7.3 used heavily.
- Recording and process management are recurring concerns.
- Multiple OBS process instances and crash logs have been investigated.
- Storage and partition troubleshooting has included TestDisk, NTFS conversion, CrystalDiskInfo, and boot configuration.

# 3. MAIN CURRENT PROJECT: Absolute Pitch AI

**Status:** Active project

Repository: marimutukisinnmu-rgb/zettaionnkann-AI

Branch: main

Goal: Build an absolute-pitch identification AI that identifies a single note from an audio WAV file, supporting MIDI note numbers 0–127.

## 3.1 Version history

### v0.2
A previous benchmark report gave:
- 8,836 correct predictions
- 16,256 total files
- 54.36% accuracy

A separate earlier user log mentioned 54.71%, but that dataset was not included in the Claude analysis discussed later.

### v0.3
Claude's supplied report gave:
- 9,278 correct predictions
- 16,256 total files
- 57.07% accuracy across all files
- 14,255 valid audio files
- 65.09% accuracy on valid audio only
- 2,001 invalid audio files

Range-specific accuracy from that report:

| MIDI range | Accuracy |
|---|---:|
| 1–23 | 49.80% |
| 24–95 | 70.61% |
| 96–127 | 63.00% |

Important:
- These WAVs were rendered MIDI instruments, not acoustic recordings of real instruments.
- Some instruments (e.g. 014, 016, 031) were discussed as possible octave/label mismatch cases.
- Those hypotheses have not been independently confirmed by listening to their source WAVs.

### v0.4.0
The project was redesigned with:
- DSP candidate generation
- Candidate verification
- Machine-learning candidate reranking
- Adaptive analysis tiers
- Explicit low-confidence classification

The v0.3 model must remain separate and untouched.

## 3.2 v0.4 design

### DSP methods
- FFT
- Harmonic templates
- YIN
- Autocorrelation
- CQT
- Attack analysis
- Sustain analysis

### ML
- ExtraTrees candidate reranker
- No deep learning
- No PyTorch/CNN dependency
- CPU-based operation

### Adaptive analysis
- Tier 0: Fast initial analysis
- Tier 1: Additional verification
- Tier 2: More extensive candidate analysis

### Result classifications
- OK
- NG
- LOW_CONF
- SKIP
- ERROR

Invalid or silent audio should be classified as SKIP, with a user-facing indication such as 「有効な音声ではありません」.

### Confidence
LOW_CONF uses a relative candidate-score margin. This is not a calibrated probability.

### MIDI ranges
- MIDI 1–23: Prioritize octave-error handling.
- MIDI 24–95: Highest practical priority.
- MIDI 96–127: Supported, but lower priority.

## 3.3 v0.4 repository structure

The following files were deployed to the repository's main branch in a previous operation:

    zettaionnkann-AI/
    ├── main.py
    ├── pitch_dsp_v040.py
    ├── pitch_model_v040.py
    ├── pitch_runtime_v040.py
    ├── benchmark.py
    └── README.md

Responsibilities:

| File | Responsibility |
|---|---|
| main.py | Main entry point / wrapper |
| pitch_dsp_v040.py | DSP processing and candidate generation |
| pitch_model_v040.py | ExtraTrees model and training |
| pitch_runtime_v040.py | Runtime prediction |
| benchmark.py | Benchmark execution and reporting |
| README.md | Documentation |

Previously reported commit:

d1a6ee6cc2ebba4f512c9c5b5770c539898574d9

This is a previously reported commit, not a guarantee that it is still the current repository HEAD.

## 3.4 Model files

v0.4 model: models/pitch_reranker_v040.pkl

v0.3 model: models/pitch_ai.pkl

The v0.3 model must not be overwritten.

## 3.5 Local verification already performed

The following checks were previously performed in a local test environment:
- Python compilation succeeded.
- Silent WAV returned SKIP.
- Synthesized C4 WAV returned OK C4 MIDI 60.
- ExtraTrees training succeeded.
- 768 synthetic training samples.
- 54,702 candidate rows.
- 23 features per candidate.
- 767/768 synthetic candidate coverage (99.87%).
- 64 trees.
- Training time approximately 19.48 seconds in that environment.

A small benchmark with two files also succeeded:

| Result | Count |
|---|---:|
| OK | 1 |
| SKIP | 1 |
| ERROR | 0 |

This does not establish real-world accuracy or 16,256-file benchmark accuracy.

## 3.6 Benchmark details

Default input directory:

    E:\rise-\temp\midi音\wav

Benchmark output directory: bench_runs/

Each run creates a timestamped directory:

    bench_runs/
    └── v0.4.0_YYYYMMDD_HHMMSS/

Old benchmark results should never be overwritten.

Benchmark records:
- Results JSONL
- TXT report
- Summary JSON
- Input manifest CSV

Timing statistics include:
- Mean
- Median
- P95
- Total wall time
- CPU time

The first three files are excluded from per-file timing statistics as warm-up.

The current manifest includes filenames, expected notes, file sizes, and modification times. It does not yet implement the originally proposed SHA1-based manifest.

Other features discussed in the ideal design but not fully implemented include:
- Instrument holdout validation
- Bootstrap confidence intervals
- Dedicated external validation dataset

Do not claim these have been implemented.

## 3.7 Important known implementation details

A previous bug in pitch_model_v040.py was fixed by adding these aliases:

    _prepare_audio = dsp._prepare_audio
    _analyze_tier0 = dsp._analyze_tier0
    _generate_candidates = dsp._generate_candidates
    _candidate_feature_matrix = dsp._candidate_feature_matrix

A remaining design issue:
- The wrapper can load/train the model before checking whether the supplied input file is useful.
- Therefore, a silent WAV might still cause training first if no model exists.
- This could be optimized later.

Another issue to monitor:
- Tier 0 early-exit conditions require strong agreement among YIN, FFT, and harmonic estimates.
- This may be strict for some instruments.

# 4. CURRENT ERROR: benchmark.py KeyError

**Status:** Fix identified; GitHub update was not yet completed at the time this context was exported.

The user ran the v0.4 benchmark and supplied this output:

    Result: LOW_CONF
    Best candidate: A#-1
    MIDI: 10
    Frequency: 14.568 Hz
    Status: LOW_CONF
    Confidence score: 0.0720
    Analysis tier: 2
    Method: Candidate reranker

Individual analyzer results:

    FFT fundamental     : A#0   MIDI=22
    Harmonic template   : A#0   MIDI=22
    Autocorrelation     : A#-1  MIDI=10
    YIN                 : A#-1  MIDI=10
    CQT                 : A#0   MIDI=22
    Attack analysis     : A#0   MIDI=22
    Sustain analysis    : A#0   MIDI=22

Top candidates:

    1. A#-1  MIDI=10 Score=0.96740
    2. A#0   MIDI=22 Score=0.89774
    3. B0    MIDI=23 Score=0.28607
    4. A0    MIDI=21 Score=0.22188
    5. F1    MIDI=29 Score=0.11947

Timing:

    Tier0=11.1ms
    Tier1=65.2ms
    Tier2=27.5ms
    Total=111.8ms

The expected and predicted note matched:

    Expected: A#-1 | Predicted: A#-1 | LOW_CONF

However, the benchmark crashed afterward with:

    KeyError: 'B'

### Root cause

The benchmark.py file fetched from GitHub main had this mapping:

    NOTE_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10}

But its regular expression already included B:

    NOTE_RE = re.compile(r"^(C#|D#|F#|G#|A#|C|D|E|F|G|A|B)(-?\d+)$")

So a note such as B0 is accepted by the regex, but then NOTE_PC[pitch] raises KeyError: 'B'.

The fix is:

    NOTE_PC = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

Expected mapping:
- A#-1 → MIDI 10
- B-1 → MIDI 11
- B0 → MIDI 23
- C0 → MIDI 12

### GitHub verification performed

The benchmark.py file was fetched from:

marimutukisinnmu-rgb/zettaionnkann-AI

Branch: main

Fetched blob SHA:

94319af26b3cd46080d2e013550d5033b31baa7d

File URL:

https://github.com/marimutukisinnmu-rgb/zettaionnkann-AI/blob/main/benchmark.py

The fix was identified and provided, but it had not yet been committed to GitHub at the time of this export.

# 5. Other programming projects

## 5.1 MIDI → SB3 Generator

Project previously developed in:

    E:\rise-\temp\midi-to-sb3

Python script: sb3_builder.py

Features discussed:
- Convert MIDI files into Scratch SB3 projects.
- Generate note sprites.
- Map MIDI note information to Scratch blocks.
- Support note velocity and volume handling.

Important discovery:
- Scratch has an audio block equivalent to setting sound volume to a percentage.
- This can be used to represent MIDI velocity/volume variations.
- CC11 support was requested and discussed.

## 5.2 Black MIDI and MIDI playback

Recurring topics:
- Extremely large MIDI files.
- NPS (notes per second).
- Polyphony.
- MIDI parsing and rendering.
- Memory consumption.
- Ultralight MIDIPlayer (ump).
- Synthesia.
- MuseScore.
- Black MIDI generation.

Previously discussed examples:
- A MIDI file of approximately 82.5 MB.
- A MIDI with around 66,402 notes.
- A file with 5 tracks and 5:02 duration.
- Playback at 60 FPS.
- Maximum NPS around 3,100 in one test.

## 5.3 Scratch / TurboWarp

User experiments include:
- Scratch coordinate edge cases.
- NaN and Infinity propagation.
- Unusual timing blocks.
- MIDI-generated Scratch projects.
- Large-scale Scratch projects.

One example:

    1 second later:
    x = -Inf
    y = NaN

Other experiments involved:
- NaN, 0
- -Inf, NaN
- -Inf, 0

These were experimental Scratch behaviors, not production functionality.

## 5.4 Polar Audio Visualizer

GitHub project previously mentioned:

marimutukisinnmu-rgb/Polar-Audio-Visualizer

Technologies:
- HTML
- JavaScript
- Web Audio
- Three.js

## 5.5 Midipb

Java project using:
- OpenGL
- LWJGL

## 5.6 ASCII Bad Apple

C# console application that displays an ASCII-style video.

## 5.7 Local Python servers

Previously developed:
- Flask HTTP server.
- CORS-enabled local server.
- FTP server using pyftpdlib.

## 5.8 OBS automation

Used:
- OBS Studio
- obs-websocket 5.7.3
- obsws-python

Tasks included:
- Recording management.
- Monitoring OBS state.
- Reading event logs.
- Managing recording workflows.
- Diagnosing crashes and lingering processes.

## 5.9 GPS playback visualization

A web application involving:
- GPS trajectory visualization.
- Map rendering.
- Playback controls.
- Timestamped position display.

Previously reported issues:
- GPS information remained stuck at "読み込み中...".
- Map appeared but trajectory lines were absent.
- Playback remained on the first timestamp.
- Playback was choppy.

Improvements discussed:
- Smooth interpolation.
- Better playback timing.
- Efficient trajectory rendering.

## 5.10 Audio processing / waveform replication

Previous experiments:
- Demucs stem separation.
- Vocal, bass, percussion, and instrumental separation.
- Python recreation of Scratch audio processing.
- Comparing generated waveforms against original audio.
- Compression and phase differences.

A prior user-reported comparison value: 0.0385.

The waveform was not yet considered a perfect match.

# 6. MHN News System

MHN means:

**News about Miho-hakun and Miana-chan**

Repository:

marimutukisinnmu-rgb/News-about-Miho-hakun-and-Miana-chan

The project includes:
- General news.
- Emergency information.
- HTML news pages.
- Character-based conversation between みほはくん and みあなちゃん.

### Main operating rules
- A = General news (MHN).
- B = Emergency news.
- A and B must operate independently.
- Failure in one must not prevent the other.
- Use Japan time (Asia/Tokyo).
- Use trustworthy sources.
- Do not invent unverified information.
- Existing valid data must not be overwritten by guesses.
- The news number is maintained through repository files such as number.txt.

Sources previously discussed:
- 気象庁
- 防災情報XML
- WeatherNews
- Tenki.jp

The user expects conversational news scripts involving the two characters.

### MHN abbreviations
- にゅーたい: News task status confirmation.
- tk: Task confirmation.
- すぱべ: Supabase.
- tn / tか: Previously mentioned as possible equivalents of task confirmation.

### MHN HTML production

User has previously discussed a workflow where requests are placed into:

    generated-sites/

The user wants queued HTML production requests handled and the results generated.

A recent conversation on 2026-11-27 included:
- User asked whether MHN HTML production requests were pending.
- User believed the production had not actually been done.
- User then said: 「今ここで作ろう」.

No assumption should be made that the HTML work has been completed.

### Supabase comments

A query previously supplied by the user:

    SELECT
        id,
        nickname,
        message,
        source_page,
        status,
        submitted_at
    FROM public.mhn_private_comments
    ORDER BY submitted_at DESC
    LIMIT 15;

The user wanted to inspect submitted comments.

# 7. Scheduled tasks / automations

The following active scheduled tasks were present in the retained context. These are historical recorded states, not a newly verified live status check.

## 7.1 全国 開店予定JSON（回転JSON）

Task ID: 6a9128a9e16c8191aab8712f0b016bd7

Schedule:
- Hourly
- Minute 15

Last run recorded:

2026-11-27T08:17:15.309498+00:00

Purpose:
- Periodically update nationwide planned store-opening JSON.
- Verify current reliable information.
- Do not overwrite correct existing data with guesses.
- Preserve data and mark unverified information when sources cannot be confirmed.

## 7.2 MHN（総合ニュース速報）＋緊急速報

Task ID: 6a97df5b297c819190c0bc537a882389

Schedule:
- Hourly
- Minute 0

Last run recorded:

2026-11-27T09:04:51.282043+00:00

Purpose:
- Execute A = MHN general news.
- Execute B = Emergency news.
- Keep both processing branches independent.

## 7.3 A/B統合：タスク監視＋MHN HTML制作

Task ID: 6a9d1d5122288191ac3a4579f49b2abc

Schedule:
- Hourly
- Minute 30

Last run recorded:

2026-11-27T08:37:13.172634+00:00

Purpose:
- A: Monitor task state and execution results.
- B: MHN HTML production and page-specific change proposals.
- Both sections run within one scheduled task.

# 8. Other major projects and experiments

## 8.1 Large-scale HTML economic / social simulation game

A large HTML game project was discussed.

Features:
- Company management.
- Company registration.
- Employees.
- Popularity.
- Bank loans.
- Cash and accounting ledger.
- Bankruptcy rules.

A demo game state previously mentioned:

    Company: みらい商店
    Company type: 株式会社
    Incorporated: true
    Employees: 3
    Popularity: 30
    Day: 9
    Cash: ¥13,300

Legal considerations:
- Japanese legal data review.
- A file named japan-law-data-review.md.
- Explicitly avoid real-world employment/compensation mechanisms in the game design.

## 8.2 YouTube / MHN promotional video

Previously discussed:
- Vertical promotional video.
- Canva workflow.
- MHN name card.

## 8.3 GitHub workflow automation

Previous ideas:
- Cloudflare Worker Auto Deploy.
- Git automation.
- Auto-pull commands.
- GitHub Actions-based TTS generation.

A TTS idea:
- Free text-to-speech.
- Multiple voices.
- Generate MP3/WAV.
- Use from Windows 10 PowerShell without requiring manual downloads.

## 8.4 PowerShell helper functions

Previously created or discussed:

| Function | Alias |
|---|---|
| Start-Monitoring | sm |
| Count-Text | ct |
| Invoke-GitAutoPull | gpul, igtp, gitpull |
| scb | Function |
| pip | Function |
| /clear | Alias |

There were also discussions about optimizing profile startup time.

Reported timing examples:
- Earlier startup approximately 67 seconds.
- Later optimized to 7.554 seconds.
- Other logs reported 9,989 ms and 180,840 ms, depending on the environment/run.

# 9. Other preferences and technical observations

- Likes experimenting with edge cases rather than only building ordinary applications.
- Often wants complete code when requesting code modifications.
- Uses command-line tools frequently.
- Prefers not to introduce unnecessary GUI workflows.
- Likes performance measurements with concrete timing and memory values.
- Often shares execution logs and expects analysis based on the actual output.
- When diagnosing code, distinguish between confirmed root causes and possible hypotheses.
- Do not assume benchmark output or repository contents without inspecting them.
- Existing user data and benchmark logs should be preserved rather than overwritten.

# 10. Recent conversation topics

These are recent topics in the retained conversation history:

| Topic | Summary |
|---|---|
| MHN HTML | Checking pending HTML production requests |
| Touhou short story | Reimu and Marisa scenario, including a nighttime scene |
| Black MIDI | MIDI note count, memory estimate, performance |
| Coin simulation | Random coin accumulation and unexpectedly low occurrence |
| Scratch coordinate behavior | NaN / Infinity experiments |
| Human/werewolf game | Multi-agent discussion |
| IMPOSBEAT | Chart editing and command-line errors |
| GPS playback | Smooth playback and trajectory display |
| MIDI → SB3 | Volume handling and CC11 |
| Audio processing | Reproducing Scratch audio processing |
| Absolute Pitch AI | v0.4 implementation and benchmark debugging |

# 11. Current conversation state

Most recent work:
1. User ran Absolute Pitch AI v0.4.0 benchmark.
2. AI returned LOW_CONF for A#-1.
3. Expected and predicted notes matched.
4. Benchmark crashed with KeyError: 'B'.
5. GitHub main/benchmark.py was inspected.
6. Root cause identified: NOTE_PC did not define B.
7. Correct fix identified: add "B": 11.
8. The fix has not yet been committed to GitHub.

**Immediate next action:** Update benchmark.py in the GitHub main branch, then have the user pull the fix and rerun the benchmark. Before writing, fetch the latest file SHA again to avoid overwriting intervening changes.
