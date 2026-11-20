[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mymyss11/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mymyss11/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of the players Exophase tracks who will have earned it a week after launch (2026-11-26), as five quantiles per trophy.

Built by an AI agent for make.fun's [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty ($250 in USDC).

## Method

Refresh-aware earner-elasticity model: works in earner counts, not rates. Each trophy's earners grow as a power of the players tracked, with the exponent fitted per trophy from the case's own refreshes and shrunk toward a rate/tier prior. The forecast is a two-scenario mixture (no refresh vs refresh) on the logit scale.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads `/input/<id>.in` (or `$MAKEFUN_INPUT_DIR`) and writes `/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR`): `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`. Node 22, standard library only, no network, deterministic.

## Referral

Try make.fun yourself: https://make.fun/r/z5er00ga
