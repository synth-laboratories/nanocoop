# DungeonGrid Crawler Task

DungeonGrid is NanoCoop's text-only dungeon-crawl environment track. It is a
multi-agent DungeonGrid task with five roles in the loop:

- `warden`: the dungeon controller / DM agent.
- `hero_1`: Barbarian, the durable melee anchor.
- `hero_2`: Wizard, the fragile ranged spellcaster.
- `hero_3`: Elf, the flexible scout/support hero.
- `hero_4`: Dwarf, the sturdy trap and door specialist.

Player agents control only the submitted heroes. A run can be solo, duo, trio,
or squad; missing heroes are not filled by ally AI. A rollout can use one shared
policy for every submitted hero or a mapping of `hero_id`/role to distinct
policies for multi-AI play. The partner zoo controls Zargon/the Warden with
scripted, aggressive, and noisy dungeon-controller variants.

On each hero decision point, the active hero's agent sees a compact text board
state and submits a JSON list of structured action objects. The observation does
not dump the full legal action list. High-level rules live in the system prompt,
and rule/action-schema details are available through the `dungeongrid_rules`
tool. The environment validates proposed actions, skips illegal queued actions
with concrete feedback, and returns control when AP/turn ends or a board-state
reveal boundary makes replanning appropriate.

Heroes can spend AP to communicate with the party or a specific submitted hero:

```json
{"type": "message", "target": "party", "payload": {"text": "Wizard scout east; Barbarian hold the entry."}}
```

Message actions follow the SMR-style shape of durable queued facts with stable
IDs, sender, recipient, round, and summary text. DungeonGrid keeps them lightweight:
recent messages are appended to shared dungeon state and rendered into later
hero observations rather than requiring a separate delivery command.

Bundled quests:

- `lantern_crypt`
- `bells_under_blackwater`
- `ashen_pantry`
- `cinder_exit`
- `low_shrine_locks`

Primary score is objective-first: completion/escape dominates, with reward as a
tie-breaker. Reward includes scout shaping: incremental reward for newly known
floor tiles and newly discovered door-separated rooms, plus an episode-end room
coverage reward so partial crawls are credited for rooms explored. Secondary
metrics report survival, rounds, exploration, room coverage, treasure, monsters
defeated, invalid actions, skipped queued actions, and per-hero action counts.

Starter configs:

```bash
nanocoop starter-agent --config configs/starter_agent_gpt41_nano_dungeongrid.yaml
nanocoop starter-agent --config configs/starter_agent_gpt41_nano_dungeongrid_solo.yaml
nanocoop starter-agent --config configs/starter_agent_gpt41_nano_dungeongrid_duo.yaml
nanocoop starter-agent --config configs/starter_agent_gpt41_nano_dungeongrid_trio.yaml
nanocoop starter-agent --config configs/starter_agent_gpt41_nano_dungeongrid_squad.yaml
```

ReAct/tool-call GPT-5-nano smoke config:

```bash
PYTHONPATH=src python -m nanocoop starter-agent \
  --config configs/react_dungeongrid_gpt5_nano_solo.yaml \
  --no-self-play \
  --episodes 1
```

Set `policy.kind: dungeongrid_react` to use the Craftax-like batched tool
contract. The current implementation name is still `dungeongrid_react`, but the
public track name is DungeonGrid. The policy calls `dungeongrid_act` at each
decision point and returns `{"intent": "...", "actions": [...]}` where each
action is a structured JSON object, not a string action name.

Smoke configs are available for the same method tracks as Overcooked:

- `configs/offline_dungeongrid_smoke.yaml`
- `configs/rlvr_dungeongrid_smoke.yaml`
- `configs/prompt_opt_dungeongrid_smoke.yaml`
