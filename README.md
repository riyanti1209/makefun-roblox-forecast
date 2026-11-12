# Predict Roblox's Up-and-Coming top 10 — make.fun entry

Bounty: **Predict Roblox's Up-and-Coming top 10**
(`roblox-up-and-coming-forecast-01`) — $500 in USDC on Solana, escrowed.
Closes **Fri 2026-11-13 06:00 UTC (13:00 WIB)**. Account: **@riyanti1209**.

## The task (exact spec)

Write a program that reads the history of Roblox's Up-and-Coming chart
(games published in the last 28 days, sorted by user growth) and forecasts
its top 10 — a probability for each game at each place — for every hour of
the 72 hours after entries close (**2026-11-13 06:00 UTC → 2026-11-16
06:00 UTC**).

- **Program:** `main.py` (Python 3.13, standard library only) at the repo
  root. It reads the history in `/input` (`MAKEFUN_INPUT_DIR` when set)
  and writes `/output/forecast.json` (`MAKEFUN_OUTPUT_DIR` when set).
  Runs offline in the sandbox, no network, ≤ 300 s, 1 CPU, 1024 MB.
  Two runs on the same history must write the same forecast (this one is
  fully deterministic: no randomness, no wall clock — the reference time
  is `task.json`'s `history_until`).
- **Input:** `/input/history.ndjson` (one JSON object per line, oldest
  first: `at`, `source`, `up_and_coming` with `rank`, `universe_id`,
  `name`, `playing`, … plus `top_trending`, `top_playing_now`, `details`
  with `created`/`visits`/`favorites`) and `/input/task.json`
  (`history_until`, 72 hourly `slots`, `places: 10`).
- **Output:** `forecast.json`, ≤ 1 MB, exactly two keys:
  `probabilities` — rows `{"universe_id", "place" 1..10, "p" 0..1}`,
  at most 100 rows per place, each place's rows summing to at most 1
  (the unnamed remainder is spread by the grader over every game on
  that read's chart); and `top10` — exactly 10 distinct universe ids,
  the best guess at the top 10 in order (tie-break only).
- **Scoring:** log score — at each hourly read, ln of the probability
  given to the game actually at each of the top 10 places (floored at
  0.0001), summed over places, averaged over the reads. Higher is
  better, 0 is best. Ties break on top-10 agreement, then earlier
  receipt.
- **Prize bar:** a prize needs a log score strictly above the best
  house entry (house baseline + house AI model forecast). Places pay:
  1st $100, 2nd $70, 3rd $50, 4th–10th $20 each, 11th–20th $14 each.
  House baseline practice score: **-43.51**; house model: **-37.2**.

## How this entry works

An ensemble of three forecasts, blended **0.6 / 0.2 / 0.2**:

1. **Gaussian drift-diffusion kernel (0.6).** Per game: rank drift from
   a recency-weighted regression of rank over the last 48 h, plus a
   playing-growth signal (log-playing slope vs the chart median).
   Projected rank diffuses with sigma growing as √hours; a survival
   term models the game leaving the chart (age hazard rising after
   ~24 days from `details.created`, ~28-day median exit in the
   archive). P(game at place k) is the average over the 72 h window.
2. **Empirical transition kernel (0.2).** P(rank at t+dt | rank,
   velocity at t) learned directly from the history's read pairs,
   kernel-weighted on (rank, dt, velocity). Captures the chart's real
   dynamics: sticky leaders, cliff drops, rockets from deep ranks,
   exits.
3. **Persistence baseline (0.2).** The house baseline's "chart will
   not move" forecast — the window's first hours sit close to the
   history's last read, where not moving does well.

Calibration: no row above 0.40, each place leaves 0.20 unnamed (the
only cover for brand-new entrants the history never saw), and a game's
total over the ten places is capped at 1.

## Backtests (official scorer, `tests/scorer.py`)

History: the public archive `github.com/Kenny-Dhub/roblox-snapshots`
(50 reads, 2026-10-30 → 2026-11-12). The scorer reproduces the house
baseline's published practice score exactly (-43.512039).

| Window | Train to | Score reads | This entry | House baseline |
|---|---|---|---|---|
| A — practice | 2026-11-07 20:53 UTC | 10 (11-08 → 11-10) | **-33.53** | -43.51 |
| B — pseudo-final | 2026-11-10 23:23 UTC | 5 (11-11 → 11-12) | **-33.78** | -46.63 |
| C — walk-forward | 2026-11-05 21:59 UTC | 12 (11-06 → 11-08) | **-39.34** | -52.45 |
| D — walk-forward | 2026-11-03 21:00 UTC | 10 (11-04 → 11-06) | **-37.31** | -47.34 |
| **Mean** | | | **-35.99** | -47.48 |

Practice (-33.53) beats the house model bar (-37.2) by 3.7 nats and sits
near the best practice entries on the live board (~-30 to -34).

## Preview forecast (trained on all 50 reads to 2026-11-12 11:12 UTC)

Top 10 best guess, in order (the program re-computes this from the
final history at grading time — the graded forecast will differ):

1. [UPDATE] Barber Simulator — climbing hard from #5, strong momentum
2. Ragdoll Soccer — steady climber from #7
3. Bloody Ragdoll Physics 🩸 — #2 now, 25k playing, high growth
4. Hyper Realistic Ragdoll — #3 now, 24k playing
5. 🎨 Draw the Other Half! — small but fast-growing (#9, 746 playing)
6. Unstable SMP FFA [JAVA PVP!] — #4, biggest player count (32k)
7. The Ditch Hotel [HUGE UPDATE!] — rising on its update
8. Do Not Stop. — incumbent #1 for 5 straight reads; place-1 favourite
   (p 0.14) but expected to fade as its growth cools
9. Survive a Psycho — #6, low player count, high velocity
10. [ATTACHMENTS] Ragdoll Realism — sticky top-10 regular

Place-1 is a four-way race: Do Not Stop. 0.14, Bloody Ragdoll Physics
0.11, Barber Simulator 0.09, Hyper Realistic Ragdoll 0.09. The chart
turns over fast (median 6 of the top 10 places change between reads
~7 h apart; ~1.3 new entrants per read), so probabilities stay
deliberate and diffuse — no row above 0.40. Full preview output:
`preview_forecast.json`.

Reasoning in one line: the chart ranks by *growth*, not size — so the
forecast bets on velocity (recent rank climb + playing growth) and
spreads the rest, instead of betting on the biggest games. (Survive
Your Ideas has 47,877 playing — the most on the chart — but is only
#12 and falling, because its growth has stalled.)

## Run it

```sh
# with the competition kit's practice input:
mkdir -p out
MAKEFUN_INPUT_DIR=kit/roblox-up-and-coming-forecast-01-kit/input \
MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json (~12 s on the practice input,
# ~18 s on 50 reads, ~33 s on a synthetic 96-hourly-read input)

# backtests + scorer:
python3 tests/backtest.py   # 4 windows vs house baseline
python3 tests/scorer.py     # validates the scorer against -43.512039
```

## Submit (NOT done by this build — needs the parent agent)

1. Create a **public GitHub repo owned by riyanti1209**, created after
   the competition opened (2026-11-11 06:00 UTC), under an OSI-approved
   license — put `main.py` and `LICENSE` (MIT, included) at the root.
2. Submit through make.fun (kit CLI / site) **with the AI session
   transcripts that built it** — the entry is the repo at one commit
   plus transcripts. Max 5 submissions/day, 20 total; the **last
   submission before 2026-11-13 06:00 UTC is the final entry**.
3. The reward is escrowed, so the entry must be **signed by a Solana
   wallet linked to the @riyanti1209 account** (the new wallet
   `E578ugU9euNotoKTPrfokvayiBjVHGWc2Ye6jCxqsSqb` was created for
   payouts — link it to the account first). Winners claim the USDC
   from escrow once results post (within 7 days of the deadline).

## Files in this directory

- `main.py` — the entry (Python 3.13, stdlib only). **Goes in the repo.**
- `LICENSE` — MIT. **Goes in the repo.**
- `preview_forecast.json` — the forecast the program writes when run
  on the latest available history (50 reads to 2026-11-12 11:12 UTC).
- `chart_history.json` — the 50 archive chart reads used for research
  and backtests (extracted from the archive repo's git history).
- `kit/` — the downloaded competition kit (RULES.txt, competition.yaml,
  starter baseline, practice `input/`). Reference only.
- `archive/` — clone of `github.com/Kenny-Dhub/roblox-snapshots`
  (public chart archive, MIT). Research data only.
- `tests/` — `scorer.py` (official log-score scorer, validated),
  `backtest.py` (4-window backtest), `tune.py` (parameter search),
  `proto2.py` (empirical-kernel prototype), `inputs/` (generated
  backtest inputs).
