# gpt41_nano_nochange_baseline

- track: `starter_agent`
- primary score: `11.0`
- cross-play mean reward: `11.0`
- self-play mean reward: `0.0`
- mean completion rate: `0.55`
- cross-partner std: `3.3665`
- num eval episodes: `20`

## Notes
- No-change starter policy package.
- Known v0.1 limitation: unresolved wide-layout episodes are retained as visible baseline failures, not hidden from the score.

## Layout Breakdown
- `demo_cook_wide`: mean_reward=`11.4286`, completion=`0.5714`, episodes=`7`
- `test_time_simple`: mean_reward=`20.0`, completion=`1.0`, episodes=`7`
- `test_time_wide`: mean_reward=`0.0`, completion=`0.0`, episodes=`6`

## Partner Breakdown
- `courier`: mean_reward=`10.0`, completion=`0.5`, episodes=`8`
- `handoff`: mean_reward=`10.0`, completion=`0.5`, episodes=`4`
- `noisy`: mean_reward=`6.6667`, completion=`0.3333`, episodes=`3`
- `potter`: mean_reward=`16.0`, completion=`0.8`, episodes=`5`

## Failed Episodes
- episode `5`: layout=`test_time_wide`, partner=`courier`, seed=`5`, reward=`0.0`, steps=`80`
- episode `6`: layout=`test_time_wide`, partner=`courier`, seed=`6`, reward=`0.0`, steps=`80`
- episode `8`: layout=`test_time_wide`, partner=`courier`, seed=`8`, reward=`0.0`, steps=`80`
- episode `12`: layout=`demo_cook_wide`, partner=`courier`, seed=`12`, reward=`0.0`, steps=`80`
- episode `19`: layout=`test_time_wide`, partner=`potter`, seed=`19`, reward=`0.0`, steps=`80`
- episode `30`: layout=`test_time_wide`, partner=`handoff`, seed=`30`, reward=`0.0`, steps=`80`
- episode `33`: layout=`demo_cook_wide`, partner=`handoff`, seed=`33`, reward=`0.0`, steps=`80`
- episode `44`: layout=`test_time_wide`, partner=`noisy`, seed=`44`, reward=`0.0`, steps=`80`
- episode `47`: layout=`demo_cook_wide`, partner=`noisy`, seed=`47`, reward=`0.0`, steps=`80`
