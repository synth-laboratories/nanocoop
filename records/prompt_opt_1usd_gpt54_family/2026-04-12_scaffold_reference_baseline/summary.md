# scaffold_reference_baseline

- track: `prompt_opt_1usd_gpt54_family`
- primary score: `2.1608`
- cross-play mean reward: `2.1608`
- self-play mean reward: `1.9167`
- mean completion rate: `1.0`
- cross-partner std: `0.233`
- num eval episodes: `60`

## Notes
- candidate 0: probe_score=2.1125 flags=['share_hidden_info_early']
- candidate 1: probe_score=2.1125 flags=['share_hidden_info_early']
- candidate 2: probe_score=2.1125 flags=['infer_partner_convention', 'share_hidden_info_early']
- candidate 3: probe_score=2.1125 flags=['avoid_duplicate_work', 'infer_partner_convention', 'share_hidden_info_early']
- candidate 4: probe_score=2.1125 flags=['avoid_duplicate_work', 'infer_partner_convention', 'recover_from_failures', 'share_hidden_info_early']
- candidate 5: probe_score=2.1125 flags=['avoid_duplicate_work', 'infer_partner_convention', 'prefer_complementary_roles', 'recover_from_failures', 'share_hidden_info_early']
- candidate 6: probe_score=2.1125 flags=['avoid_duplicate_work', 'finish_pipeline', 'infer_partner_convention', 'prefer_complementary_roles', 'recover_from_failures', 'share_hidden_info_early']
