#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster for make.fun
bounty roblox-up-and-coming-forecast-01 ($500, closes 2026-11-13 06:00 UTC).

Reads the chart history in /input (MAKEFUN_INPUT_DIR) and writes one
forecast to /output/forecast.json (MAKEFUN_OUTPUT_DIR):
  {"probabilities": [{"universe_id", "place", "p"}], "top10": [10 ids]}

Method: an ensemble of three forecasts, blended 0.6 / 0.2 / 0.2.
  1. Gaussian drift-diffusion kernel (weight 0.6). For every game on
     the last chart snapshot we estimate a rank drift (recent rank
     velocity, with playing-growth as a second signal) and a diffusion
     that widens with horizon, plus a survival term for the game
     leaving the chart (age hazard: games leave around 28 days after
     they were published). P(game at place k) is the average over the
     72 hours of the scored window of the Gaussian mass of the game's
     projected rank falling in place k, times its survival probability.
  2. Empirical transition kernel (weight 0.2). P(rank at t+dt | rank,
     velocity at t) learned directly from the history's read pairs,
     with Gaussian kernel weights on (rank, dt, velocity). Captures
     the chart's real dynamics: sticky leaders, cliff drops, rockets
     from deep ranks, exits.
  3. Persistence baseline (weight 0.2): the house baseline's forecast
     that the chart will not move. The window's first hours sit close
     to the history's last read, where not moving does well.

The blend is honest and diffuse: no row above 0.40, each place leaves
0.20 unnamed (spread by the grader over every game on the read's
chart, which is the only cover for brand-new entrants), and a game's
total over the ten places is capped at 1.

Standard library only. No network, no randomness, no wall-clock use:
the reference time is task.json's history_until, so two runs on the
same history write the same forecast byte for byte. Runs in well
under the 300 s limit (the empirical kernel thins dense hourly reads
to >= 2 h spacing when building transition pairs).

Backtests (official log-score scorer, tests/): practice window
(train to 2026-11-07, score 10 archive reads 11-08..10) -33.99;
pseudo-final (train to 11-10, score 5 reads 11-11..12) -35.25;
two walk-forward windows -38.95 / -36.98. Mean -36.29, against the
house baseline's -47.48 mean and the house model's -37.2 practice.
"""

import json
import math
import os

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

GAUSS = {
    "sigma0": 3.0,        # diffusion: sigma(h) = sigma0 + sigma1*sqrt(h)
    "sigma1": 0.5,
    "drift_tau_h": 24.0,  # rank drift decays with this time constant (hours)
    "growth_to_rank": 12.0,  # playing-growth signal -> ranks/day
    "velocity_half_life_h": 48.0,
    "velocity_window_h": 48.0,
    "velocity_shrink": 2.0,
    "cap": 0.30,
    "unnamed": 0.30,
    "hazard_base": 0.01,  # exit hazard per day
    "hazard_age_max": 0.10,
    "age_ramp_start": 24.0,
    "age_ramp_end": 35.0,
    "age_unknown_lag": 14.0,  # assumed publish->first-chart lag (days)
    "trending_sigma_mult": 0.92,
}

EMPIRICAL = {
    "bw_rank": 4.0,
    "bw_dt": 12.0,
    "bw_vel": 6.0,
    "anchors": [6, 12, 24, 36, 48, 60, 72],
    "anchor_weights": [9, 9, 12, 12, 12, 12, 6],
    "cap": 0.40,
    "unnamed": 0.20,
    "velocity_half_life_h": 30.0,
    "velocity_window_h": 96.0,
    "velocity_shrink": 2.0,
    "thin_hours": 2.0,  # min spacing of reads used for transition pairs
}

BLEND = {"gauss": 0.6, "empirical": 0.2, "baseline": 0.2}
BLEND_CAP = 0.40
BLEND_UNNAMED = 0.20

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _parse_time(s):
    from datetime import datetime, timezone
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _read_history(input_dir):
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda d: d["at"])
    return snaps


def _read_task(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        return json.load(f)


def _rank_velocity(points, half_life, window_h, shrink):
    """Weighted linear fit of rank vs hours-ago over the recent window.
    points: [(hours_ago, rank)]. Returns ranks/day (negative = climbing),
    shrunk toward 0."""
    pts = [(h, r) for h, r in points if h <= window_h]
    if len(pts) < 2:
        return 0.0
    ws = [0.5 ** (h / half_life) for h, _ in pts]
    sw = sum(ws)
    mh = sum(w * h for w, (h, _) in zip(ws, pts)) / sw
    mr = sum(w * r for w, (_, r) in zip(ws, pts)) / sw
    num = sum(w * (h - mh) * (r - mr) for w, (h, r) in zip(ws, pts))
    den = sum(w * (h - mh) ** 2 for w, (h, _) in zip(ws, pts))
    if den <= 0:
        return 0.0
    vel = -(num / den) * 24.0
    return vel * (sw / (sw + shrink))


def _log_growth(playing, window_h):
    """Slope of log(playing) per day over the recent window, or None."""
    pts = sorted((h, v) for h, v in playing if h <= window_h and v > 0)
    if len(pts) < 2:
        return None
    dt_days = max((pts[0][0] - pts[-1][0]) / 24.0, 0.25)
    return math.log(pts[-1][1] / pts[0][1]) / dt_days


# ---------------------------------------------------------------------------
# component 1: Gaussian drift-diffusion kernel
# ---------------------------------------------------------------------------


def _gaussian_forecast(snaps, task):
    p = GAUSS
    places = int(task.get("places", 10))
    now = _parse_time(task["history_until"])
    horizon = 72
    win = task.get("window") or {}
    if isinstance(win.get("slots"), int):
        horizon = int(win["slots"])

    games = {}
    last = snaps[-1]
    last_chart = sorted(last.get("up_and_coming") or [], key=lambda r: r["rank"])
    trending = {}
    for row in last.get("top_trending") or []:
        trending[row["universe_id"]] = row["rank"]

    for snap in snaps:
        t = _parse_time(snap["at"])
        hours_ago = max((now - t).total_seconds() / 3600.0, 0.0)
        det = snap.get("details") or {}
        for row in snap.get("up_and_coming") or []:
            uid = row["universe_id"]
            g = games.get(uid)
            if g is None:
                g = games[uid] = {"ranks": [], "playing": [],
                                  "created": None, "first_seen": t}
            g["ranks"].append((hours_ago, row["rank"]))
            if row.get("playing"):
                g["playing"].append((hours_ago, float(row["playing"])))
            if isinstance(det, dict):
                d = det.get(str(uid)) or det.get(uid)
                if isinstance(d, dict) and d.get("created") and g["created"] is None:
                    try:
                        g["created"] = _parse_time(d["created"])
                    except Exception:
                        pass
    det_last = last.get("details")
    if isinstance(det_last, dict):
        for key, d in det_last.items():
            try:
                uid = int(key)
            except Exception:
                continue
            if isinstance(d, dict) and d.get("created"):
                g = games.get(uid)
                if g is not None and g["created"] is None:
                    try:
                        g["created"] = _parse_time(d["created"])
                    except Exception:
                        pass

    order = [r["universe_id"] for r in last_chart]
    last_rank = {r["universe_id"]: r["rank"] for r in last_chart}

    growths = []
    for uid in order:
        gr = _log_growth(games[uid]["playing"], p["velocity_window_h"])
        if gr is not None:
            growths.append(gr)
    growths.sort()
    med_growth = growths[len(growths) // 2] if growths else 0.0

    prob = {}
    for uid in order:
        g = games[uid]
        r0 = float(last_rank[uid])
        vel = _rank_velocity(g["ranks"], p["velocity_half_life_h"],
                             p["velocity_window_h"], p["velocity_shrink"])
        gr = _log_growth(g["playing"], p["velocity_window_h"])
        drift = vel + (p["growth_to_rank"] * (gr - med_growth)
                       if gr is not None else 0.0)
        if g["created"] is not None:
            age_days = (now - g["created"]).total_seconds() / 86400.0
        else:
            age_days = ((now - g["first_seen"]).total_seconds() / 86400.0
                        + p["age_unknown_lag"])
        age_days = max(age_days, 0.0)
        sig_mult = p["trending_sigma_mult"] if uid in trending else 1.0
        dist = {k: 0.0 for k in range(1, places + 1)}
        for h in range(1, horizon + 1):
            mu = r0 + drift * (h / 24.0) * math.exp(-h / p["drift_tau_h"])
            mu = min(max(mu, 1.0), 120.0)
            sigma = (p["sigma0"] + p["sigma1"] * math.sqrt(h)) * sig_mult
            age_h = age_days + h / 24.0
            if age_h <= p["age_ramp_start"]:
                haz = p["hazard_base"]
            elif age_h >= p["age_ramp_end"]:
                haz = p["hazard_base"] + p["hazard_age_max"]
            else:
                frac = ((age_h - p["age_ramp_start"])
                        / (p["age_ramp_end"] - p["age_ramp_start"]))
                haz = p["hazard_base"] + p["hazard_age_max"] * frac
            surv = math.exp(-haz * h / 24.0)
            for k in range(1, places + 1):
                dist[k] += surv * (_phi((k + 0.5 - mu) / sigma)
                                   - _phi((k - 0.5 - mu) / sigma))
        for k in dist:
            dist[k] /= horizon
        prob[uid] = dist
    return prob, order, last_rank


# ---------------------------------------------------------------------------
# component 2: empirical transition kernel
# ---------------------------------------------------------------------------


def _empirical_forecast(snaps, task):
    p = EMPIRICAL
    places = int(task.get("places", 10))
    now = _parse_time(task["history_until"])
    times = [_parse_time(s["at"]) for s in snaps]
    t0 = times[0]
    hours = [(t - t0).total_seconds() / 3600.0 for t in times]
    now_h = (now - t0).total_seconds() / 3600.0

    # per-game rank series on hours-since-start
    series = {}
    for snap, h in zip(snaps, hours):
        for row in snap.get("up_and_coming") or []:
            series.setdefault(row["universe_id"], []).append((h, row["rank"]))
    for uid in series:
        series[uid].sort()

    def velocity_at(uid, upto_h):
        pts = [(now_h - h, r) for h, r in series[uid]
               if h <= upto_h and (now_h - h) <= p["velocity_window_h"]]
        return _rank_velocity(pts, p["velocity_half_life_h"],
                              p["velocity_window_h"], p["velocity_shrink"])

    last = snaps[-1]
    last_chart = sorted(last.get("up_and_coming") or [], key=lambda r: r["rank"])
    order = [r["universe_id"] for r in last_chart]
    last_rank = {r["universe_id"]: r["rank"] for r in last_chart}
    cur_vel = {uid: velocity_at(uid, now_h) for uid in order}

    # thin dense reads for transition pairs (keeps runtime bounded)
    keep = []
    last_kept = None
    for i, h in enumerate(hours):
        if last_kept is None or h - last_kept >= p["thin_hours"] or i == len(hours) - 1:
            keep.append(i)
            last_kept = h
    # precompute velocities at each kept read for every game (cheap once)
    vel_at = {}
    for i in keep:
        for uid in series:
            if any(h <= hours[i] for h, _ in series[uid]):
                vel_at[(uid, i)] = velocity_at(uid, hours[i])

    samples = []  # (r_start, vel_start, dt_h, r_end|None)
    for a in range(len(keep)):
        i = keep[a]
        rank_i = {row["universe_id"]: row["rank"]
                  for row in snaps[i].get("up_and_coming") or []}
        for b in range(a + 1, len(keep)):
            j = keep[b]
            dt = hours[j] - hours[i]
            if dt > 96:
                break
            rank_j = {row["universe_id"]: row["rank"]
                      for row in snaps[j].get("up_and_coming") or []}
            for uid, r in rank_i.items():
                samples.append((r, vel_at.get((uid, i), 0.0), dt,
                                rank_j.get(uid)))

    def kernel_dist(uid):
        r0 = float(last_rank[uid])
        v0 = cur_vel[uid]
        dist = {k: 0.0 for k in range(1, places + 1)}
        for H, wH in zip(p["anchors"], p["anchor_weights"]):
            num = {k: 0.0 for k in range(1, places + 1)}
            den = 0.0
            for (r, v, dt, r2) in samples:
                w = math.exp(-((r - r0) / p["bw_rank"]) ** 2)
                if w < 1e-4:
                    continue
                w *= math.exp(-((dt - H) / p["bw_dt"]) ** 2)
                if w < 1e-4:
                    continue
                w *= math.exp(-((v - v0) / p["bw_vel"]) ** 2)
                if w < 1e-4:
                    continue
                den += w
                if r2 is not None and 1 <= r2 <= places:
                    num[r2] += w
            if den > 0:
                for k in range(1, places + 1):
                    dist[k] += wH * num[k] / den
        tot_w = sum(p["anchor_weights"])
        for k in dist:
            dist[k] /= tot_w
        return dist

    return {uid: kernel_dist(uid) for uid in order}, order, last_rank


# ---------------------------------------------------------------------------
# component 3: persistence baseline (house baseline)
# ---------------------------------------------------------------------------


def _baseline_forecast(snaps, task):
    places = int(task.get("places", 10))
    last = snaps[-1]
    chart = sorted(last.get("up_and_coming") or [], key=lambda r: r["rank"])
    ids = [r["universe_id"] for r in chart]
    prob = {}
    for place in range(1, places + 1):
        d = {}
        if place - 1 < len(ids):
            d[ids[place - 1]] = 0.5
        if place == 1:
            if len(ids) > 1:
                d[ids[1]] = 0.3
        else:
            if place - 2 >= 0:
                d[ids[place - 2]] = d.get(ids[place - 2], 0.0) + 0.15
            if place < len(ids):
                d[ids[place]] = d.get(ids[place], 0.0) + 0.15
        for uid, v in d.items():
            prob.setdefault(uid, {})[place] = v
    order = ids
    last_rank = {r["universe_id"]: r["rank"] for r in chart}
    return prob, order, last_rank


# ---------------------------------------------------------------------------
# blend + output
# ---------------------------------------------------------------------------


def build_forecast(input_dir):
    task = _read_task(input_dir)
    snaps = _read_history(input_dir)
    places = int(task.get("places", 10))

    pg, order, last_rank = _gaussian_forecast(snaps, task)
    pe, _, _ = _empirical_forecast(snaps, task)
    pb, _, _ = _baseline_forecast(snaps, task)

    # weighted average of the three components' per-(game, place) mass
    mix = {}
    for comp, w in ((pg, BLEND["gauss"]), (pe, BLEND["empirical"]),
                    (pb, BLEND["baseline"])):
        for uid, dist in comp.items():
            for k, v in dist.items():
                if v > 0:
                    mix[(uid, k)] = mix.get((uid, k), 0.0) + w * v

    rows = []
    for place in range(1, places + 1):
        cands = sorted(((v, uid) for (uid, pl), v in mix.items() if pl == place),
                       key=lambda x: (-x[0], x[1]))
        capped = [(min(v, BLEND_CAP), uid) for v, uid in cands]
        total = sum(v for v, _ in capped)
        target = 1.0 - BLEND_UNNAMED
        if total > target:
            scale = target / total
            capped = [(v * scale, uid) for v, uid in capped]
        for v, uid in capped:
            if v > 0.0005:
                rows.append({"universe_id": uid, "place": place,
                             "p": round(v, 6)})
    by_game = {}
    for r in rows:
        by_game.setdefault(r["universe_id"], []).append(r)
    for uid, rs in by_game.items():
        tot = sum(r["p"] for r in rs)
        if tot > 1.0:
            for r in rs:
                r["p"] = round(r["p"] / tot, 6)
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # top10: most likely to be in the top 10 first, then best expected
    # place in the blended forecast, then current rank
    intop = {}
    exp_place = {}
    for (uid, k), v in mix.items():
        intop[uid] = intop.get(uid, 0.0) + v
        exp_place[uid] = exp_place.get(uid, 0.0) + k * v
    for uid in intop:
        exp_place[uid] = exp_place[uid] / intop[uid]
    top10 = sorted(order, key=lambda u: (-intop.get(u, 0.0),
                                         exp_place.get(u, 999.0),
                                         last_rank[u]))[:places]
    return {"probabilities": rows, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task = _read_task(input_dir)
    forecast = build_forecast(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w") as f:
        f.write(json.dumps(forecast) + "\n")
    return out_path


if __name__ == "__main__":
    print(main())
