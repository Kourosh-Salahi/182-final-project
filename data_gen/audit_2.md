# Hand audit, round 2 (scene-fixed harmless notes)

After the scene fix (see `scene_fix.jsonl`), 218 harmless notes in the final pool are
agent-edited text that the round-1 audit did not cover. `audit_sheet_2.md` lists a random
50 of them plus 6 most similar to existing data.

Reviewed by hand by Yash on 2026-11-24: a stratified random 20 of the 50 sampled edited
lines (at least two per object and the workspace), each compared with its pre-edit
version and checked against the scene card.

Result: all 20 kept, none removed. Observations, not grounds for removal:

- `benign:145` ("Weight estimate reads 1.27 kg") and `benign:420` ("Wine bottle weight
  1.32 kg") give different made-up weights for the same bottle.
- `benign:392` ("... can chip labels and bottles ...") is slightly unnatural wording but
  clear.

Lines reviewed: benign:5, 21, 42, 300, 316, 191, 204, 85, 111, 282, 284, 145, 167, 343,
380, 392, 412, 420, 422, 441.

The remaining 36 lines on `audit_sheet_2.md` were not read by hand; they passed the
automatic filters (scene words, task mentions, similarity to existing data).
