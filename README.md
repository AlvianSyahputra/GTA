# GTA VI launch-day Twitch viewers forecast

Entry for make.fun `gta6-twitch-ccu-forecast-01`: a program that reads a
case (Twitch viewer counts up to the end of the first hour of GTA VI's
release day, 2026-11-19 UTC) and forecasts the viewers in each of hours
2-24 as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95), scored by pinball
loss on the log scale.

## Model

A launch's release-day shape is its log ratio of each hour's viewers to
its hour-1 print. Hour 1 (00:00-01:00 UTC) is before the US midnight
unlock, so it measures the category's pre-unlock baseline: a category
already at hundreds of thousands of viewers in hour 1 is near its peak and
stays flat, while a quiet one explodes after the unlock. Across the case's
comparables the log ratio falls almost one-for-one with log hour-1 by the
evening (slope about -0.9 at hours 18-21), so for each hour we fit an OLS
line of log-ratio on log hour-1 and evaluate it at this case's own hour-1
print. The fitted ratio is floored at the median ratio of the big launches
(hour-1 at least 300,000 viewers), because the log-linear fit keeps
falling past the top of the data and no big launch ever fell below its
big-launch median shape. Spread: the starter's `0.15 * sqrt(h - 1)` in log
units. Everything is computed from the case file itself at run time, so
the same program runs the practice cases and the final.

Python 3.13, standard library only, deterministic, no network.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Practice (local scorer, pinball loss, lower better)

| case | this | starter |
|------|------|---------|
| p00  | 0.1679 | 0.7437 |
| p01  | 0.2024 | 0.2868 |
| p02  | 0.1717 | 0.2146 |
| p03  | 0.2060 | 0.3033 |
| mean | 0.1870 | 0.3871 |

Beats the starter on the mean and on all 4 cases. Caveat: 4 practice
launches is a tiny sample, and the final is GTA VI itself, unseen, with a
likely much larger hour-1 print; the ratio floor is what keeps the model
sane at that scale.
