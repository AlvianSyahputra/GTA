"""gta6-twitch-ccu-forecast-01's entry in Python 3.13 (standard library only).

Forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour from
the count in the day's first hour, as five quantiles (0.05, 0.25, 0.5, 0.75,
0.95) for each of hours 2 to 24, scored by pinball loss on the log scale.

Model: a launch's release-day shape is its log ratio of each hour's viewers
to its hour-1 print. Hour 1 (00:00-01:00 UTC) is before the US midnight
unlock, so it measures the category's pre-unlock baseline: a category
already at hundreds of thousands of viewers in hour 1 is near its peak and
stays flat (ELDEN RING's evening peak is 0.95x its hour 1), while a quiet
one explodes after the unlock (Silksong's is 2000x). Across the case's
comparables the log ratio falls almost one-for-one with log hour-1 by the
evening (slope about -0.9 at hours 18-21), so for each hour we fit an OLS
line of log-ratio on log hour-1 and evaluate it at this case's own hour-1
print. The fitted ratio is floored at the median ratio of the big launches
(hour-1 at least 300,000 viewers): the log-linear fit keeps falling past
the top of the data, and no big launch ever fell below its big-launch
median shape (about 0.54x at hour 12, 1.10x at hour 18). Spread: the
starter's 0.15 * sqrt(h - 1) in log units -- the cross-launch residual
spread is far wider than the truth's spread around this centre, so a wider
spread only pays for width; the starter's spread is the honest one here.

Everything is computed from the case file itself at run time (its
comparables, whatever they are), so the same program runs the practice
cases and the final. No network, no randomness, deterministic: the same
case always writes the same bytes.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only. -P keeps this file's directory OFF
sys.path, so a module shipped beside main.py imports only after the
sys.path line below; -s drops user site-packages.

What it does:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there): JSON
    {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start",
    "checkpoint", "levels", "known", "forecast", "history",
    "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print {hour: 1, from, until,
    viewers, channels, reads} and `forecast` lists the hours to forecast,
    2..24, and each comparable holds its release `day_start` and its
    `hourly` points {from, viewers, channels};
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each
    once, no other key, each q five finite numbers above 0 and at most 1e9
    (viewers; fractions allowed), non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Practice (local scorer, pinball loss on the log scale, lower better):
  case  loss (this)  loss (starter)
  p00   0.1679       0.7437
  p01   0.2024       0.2868
  p02   0.1717       0.2146
  p03   0.2060       0.3033
  mean  0.1870       0.3871
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of an hour's move, in natural-log units per square-root hour
# (the starter's spread: the practice cases support it, see the docstring).
SIGMA = 0.15
# A "big launch" for the ratio floor: hour-1 print at least this many viewers.
BIG_LAUNCH = 300_000
# Sanity bounds on a fitted median ratio (the house's own bounds are 0.01-100).
RATIO_MIN = 0.05
RATIO_MAX = 100.0


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares line y = inter + slope * x. Returns (inter, slope)."""
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx else 0.0
    return my - slope * mx, slope


def comparable_day_hours(comp: dict) -> dict[int, float]:
    """A comparable's release-day hourly viewers, {hour 1..24: viewers}."""
    try:
        day0 = parse_iso(comp["day_start"])
    except (KeyError, ValueError):
        return {}  # a comparable with no usable release day contributes nothing
    hours: dict[int, float] = {}
    for pt in comp.get("hourly") or []:
        try:
            t = parse_iso(pt["from"])
        except (KeyError, ValueError):
            continue
        if day0 <= t < day0 + timedelta(hours=24):
            h = int((t - day0).total_seconds() // 3600) + 1
            v = pt.get("viewers")
            if isinstance(v, (int, float)) and v > 0:
                hours[h] = float(v)
    return hours


def median_ratios(p1: float, comparables: list[dict]) -> dict[int, float]:
    """Per-hour median ratio to hour 1, scale-conditioned and floored.

    For each hour, OLS of log(viewers_h / viewers_1) on log(viewers_1)
    across the comparables, evaluated at this case's hour-1 print, floored
    at the median ratio of the big launches. Falls back to 1.0 (persistence)
    for an hour no comparable covers.
    """
    # (log hour-1, {hour: log ratio}) per comparable with a usable hour 1.
    curves = []
    for comp in comparables:
        hours = comparable_day_hours(comp)
        if 1 not in hours or hours[1] <= 0:
            continue
        p1c = hours[1]
        curves.append((math.log(p1c), p1c,
                       {h: math.log(v / p1c) for h, v in hours.items() if h >= 2}))
    ratios: dict[int, float] = {}
    lp = math.log(max(1.0, p1))
    for h in range(2, 25):
        pts = [(x, lr[h]) for x, _, lr in curves if h in lr]
        if not pts:
            ratios[h] = 1.0
            continue
        inter, slope = ols([x for x, _ in pts], [y for _, y in pts])
        centre = inter + slope * lp
        big = [y for x, y in pts if math.exp(x) >= BIG_LAUNCH]
        if big:
            centre = max(centre, median(big))
        ratios[h] = min(RATIO_MAX, max(RATIO_MIN, math.exp(centre)))
    return ratios


def hour_quantiles(p1: float, ratio: float, h: int) -> list[float | int]:
    """The five quantiles for hour h around the fitted median p1 * ratio."""
    med = max(1.0, p1) * ratio
    spread = SIGMA * math.sqrt(h - 1)
    qs: list[float | int] = []
    prev = 0.0
    for z in Z:
        q = round2(med * math.exp(z * spread))
        q = max(0.01, min(1_000_000_000, float(q)))
        q = max(q, prev)  # keep the row non-decreasing after rounding
        prev = q
        qs.append(int(q) if float(q).is_integer() else q)
    return qs


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    ratios = median_ratios(p1, case.get("comparables") or [])
    return {"hours": [{"hour": row["hour"],
                       "q": hour_quantiles(p1, ratios.get(row["hour"], 1.0), row["hour"])}
                      for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
