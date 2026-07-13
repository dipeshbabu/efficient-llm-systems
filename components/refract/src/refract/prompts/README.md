# REFRACT prompt set

`v0.1.jsonl` contains the 30 deterministic prompts used by Axis A and PLAD.
Each non-empty line is a UTF-8 JSON object with these fields:

- `id`: stable unique prompt identifier
- `category`: broad task category used for coverage summaries
- `prompt`: user-facing prompt text

The prompt text is released under CC0. Keep IDs stable when correcting text;
add a new prompt-set version when changing task semantics or score calibration.

Validate the JSONL one object per line:

```bash
python -c "import json, pathlib; [json.loads(x) for x in pathlib.Path('src/refract/prompts/v0.1.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]"
```
