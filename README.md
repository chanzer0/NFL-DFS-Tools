# NFL-DFS-Tools

Packaged in this repository is an NFL Optimizer and GPP Simulator for DraftKings and FanDuel, along with other utilities for building lineups and running Monte Carlo tournaments. If you hit issues or want to discuss strategy, you can join our discord: https://discord.gg/ThVAYfVuzU

This tool was created in conjunction with @bjungwirth (DFS game theory + applied Python/data science). Find him on Twitter: https://twitter.com/blainejungwirth?lang=en or his blog: http://http://jungwirb.io/

## Installation

### Prerequisites
- Python 3.12 or newer (https://www.python.org/downloads/)
- Windows, macOS, or Linux. Examples below use Windows PowerShell.

### Recommended: uv package manager
We use uv for fast, reproducible installs from `pyproject.toml`.

```powershell
# Install uv
# Windows (PowerShell)
iwr -useb https://astral.sh/uv/install.ps1 | iex
# macOS/Linux (bash)
# curl -LsSf https://astral.sh/uv/install.sh | sh

# From repo root, create/sync the virtual env and install deps
uv sync

# Run commands through uv
uv run src/main.py dk sim 58823 10000
uv run src/main.py dk sd_sim cid 10000
uv run src/main.py fd sd_sim cid 10000

# Or use the entrypoint defined in pyproject
uv run nfl-dfs dk sim 58823 10000
```

### Manual install (optional)
If you prefer classic pip, install the packages listed under `[project.dependencies]` in `pyproject.toml`. We recommend uv.

### Getting data
After cloning/downloading the repo, import player contest data from DraftKings or FanDuel. Rename the export to `player_ids.csv` and place it under `dk_data/` or `fd_data/` respectively. Put your projections/ownership CSVs there as well (see required columns in the Config section).

![Download image](readme_images/download.png)
![Directory image](readme_images/directory.png)

## Usage
Open a terminal in the repo root (on Windows: File > Open Windows PowerShell).

Generic templates using uv:
```powershell
uv run src/main.py <site> <process> <args>
# or
uv run nfl-dfs <site> <process> <args>
```

Where `<site>` is:
- `dk` for DraftKings
- `fd` for FanDuel

`<process>` is:
- `opto` for running optimal lineup crunches (with or without randomness)
- `sim` for running classic-slate GPP simulations
- `sd_opto` for running showdown crunches
- `sd_sim` for running showdown simulations

Classic sim usage patterns:
- Arbitrary field: `uv run src/main.py <site> sim <field_size> <num_iterations>`
- Contest-driven field/ROI: `uv run src/main.py <site> sim cid <num_iterations>` (reads `dk_data/contest_structure.csv`)

Optionally, upload your own lineups instead of randomly generating the field by adding the `file` flag: `uv run src/main.py <site> sim cid file 10000`. Provide `tournament_lineups.csv` in the repo root with six columns (players) per row.

For example, to generate 1000 DK lineups with 3 uniques:
```powershell
uv run src/main.py dk opto 1000 3
```
Control randomness via `"randomness": X` in `config.json` (0–100).

![Shell image](readme_images/shell.png)
![Example usage](readme_images/usage.png)

## Config
Use `sample.config.json` as a template. When ready, copy it to `config.json` (ensure file extensions are visible on Windows).

```
{
    "projection_path": "projections.csv", // required columns: Name, Position, Team, Salary, Fpts, Own%, StdDev; optional: Ceiling, Field Fpts, CptOwn%
    "player_path": "player_ids.csv",
    "contest_structure_path": "contest_structure.csv",
    "use_double_te": true,
    "global_team_limit": 4,
    "projection_minimum": 5,
    "randomness": 25,
    "min_lineup_salary": 49200,
    "max_pct_off_optimal": 0.25,
    "num_players_vs_def": 0,
    "pct_field_using_stacks": 0.65,
    "pct_field_double_stacks": 0.4,
    "default_qb_var": 0.4,
    "default_skillpos_var": 0.5,
    "default_def_var": 0.5,
    "custom_correlations": {
        "Joe Burrow": {"RB": 0.69, "WR": -0.42}
    }
}
```

You can also define `stack_rules`, `team_limits`, and matchup rules as shown in the original README (omitted here for brevity).

## Output
Data is stored in the `output/` directory. Subsequent runs overwrite previous output files, so rename/move files you want to keep.

### `opto` Process
![Example output](readme_images/opto_output.png)

### `sim` Process
![Example output](readme_images/sim_output.png)

## Simulation Methodology
- Players are segmented by projected fantasy points into windows using `distribution_data/fp_distributions_<site>.csv`.
- For each position/window we use empirically fitted families: gamma, lognormal, Weibull, skew-normal, ex-Gaussian, generalized gamma, shifted gamma.
- For complex families, we sample from the fitted parameters, then apply an affine transform so samples match each player’s projected mean and standard deviation exactly.
- We impose correlations via a Gaussian copula with Iman–Conover rank reordering to preserve marginals while achieving the target correlation structure.
- Correlations are read from YAML by position and projection window (`distribution_data/fp_correlations_<site>.yaml`), falling back to `.npz` when necessary.
- You can override correlations per player/position through `custom_correlations` in `config.json`.
- Heavy-tail guardrail: if a fitted heavy-tailed family yields extreme tails relative to a player’s mean/std, we fall back to a calibrated gamma matching mean and std.
- Showdown and classic sims share the same sampling/correlation logic. We also handle FanDuel captain/FLEX `UniqueKey`s and fuzzy name matching to reduce mismatches in classic.

Lineup generation for the simulated field uses projections, ownership, stacking and constraints from `config.json`. We rank lineups for each simulation, allocate prizes from `contest_structure.csv`, and aggregate ROI and rates (Wins, Top1%, etc.).

## IMPORTANT NOTES
- DraftKings and FanDuel are supported for classic and showdown.
- Place `player_ids.csv` and `projections.csv` in `dk_data/` or `fd_data/` accordingly.
- Distribution and correlation files live under `distribution_data/` and are used automatically.

We believe in open source and collaboration. This simulation toolkit will remain free.

Tips/donations are appreciated:

PayPal: https://www.paypal.com/donate/?hosted_button_id=NALW2B8ZMTCG8

Ethereum Address:

![Eth Wallet QR Code](readme_images/eth_qr_code.png)

0x2D62C15849ddC68DDB2F9dFBC426f0bF46eaE006
