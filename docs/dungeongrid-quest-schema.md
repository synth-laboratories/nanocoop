# DungeonGrid Quest Schema

Bundled DungeonGrid quests live in the standalone `dungeongrid` package under
`src/dungeongrid/dungeons/<dungeon_id>/quest.json`.
Quest files are plain JSON so they can be generated, audited, and loaded by the
NanoCoop DungeonGrid backend.

Required top-level fields:

```json
{
  "quest_id": "lantern_crypt",
  "title": "The Lantern Crypt",
  "difficulty": "starter",
  "max_heroes": 4,
  "recommended_heroes": ["barbarian", "wizard", "elf", "dwarf"],
  "map": {"width": 17, "height": 15, "ascii": "..."},
  "objective": {"type": "retrieve_and_escape", "item_id": "ember_idol", "escape_tile": [1, 1]},
  "scripts": {},
  "torch": 20,
  "hero_starts": [[1, 1], [2, 1], [3, 1], [4, 1]]
}
```

Map glyphs:

| Glyph | Meaning |
|---|---|
| `#` | wall |
| `.` | floor |
| `E` | entry / escape tile |
| `D` | closed regular door |
| `S` | hidden secret door |
| `T` | hidden trap |
| `C` | chest |
| `I` | objective item |
| `G` | skitterling |
| `B` | bone guard |
| `K` | gloom cultist |
| `R` | crypt brute |

Coordinates are `[x, y]`, zero-indexed from the upper-left corner.

Supported objective types:

- `retrieve_and_escape`
- `escort_relic_to_exit`

Supported scripts include `on_objective_taken`, `bell_alarm`,
`token_requirement`, `locked_shrine_door`, `round_limit_soft`, and
`alert_reinforcement_threshold`.
