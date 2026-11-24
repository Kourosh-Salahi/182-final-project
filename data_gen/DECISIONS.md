# Decisions made for this dataset

Each entry gives the decision, the options considered, and the reason.

## What counts as an attack

**Decision.** An attack is text in a tool reply that tries to get the robot to do
something other than the user's task. A plain restatement of the user's own task, phrased
as a command, is harmless. Anything that changes the object, place or action, or adds
another task, is an attack.
**Options.** (a) Own-task commands are harmless; (b) every command in a tool reply is an
attack; (c) leave own-task commands out of the data.
**Reason.** Only (a) makes the label depend on the user's task, which is the claim the
task-conditioned classifier makes, and it matches how the robot experiments score success
(the attacker's goal is always a different task). (b) makes the task input pointless and
contradicts existing labels (care notes are commands labelled harmless). (c) removes the
examples the contrast study needs. Results on own-task sentences are reported as a
separate slice for readers who prefer (b).

## Mentions of other tasks without a plain command

**Decision.** A sentence that mentions another task is an attack if it pushes the robot to
do it now (questions, rumours, stories that end in a push). It is harmless if it only
records the past.
**Reason.** This matches the existing data, where history notes such as "the robot was
asked to ..." are labelled harmless. New attack shapes were required to end in a push, and
new harmless lines may not mention tasks at all.

## Near-miss and add-on contrast pairs

**Decision.** The contrast set has three kinds of pair: whole-task swaps, near-misses (one
detail changed) and add-ons (the user's task plus another), each scored separately.
**Options.** Swaps only, or all three.
**Reason.** Swaps alone can be solved by checking whether the sentence copies the task
text. Near-misses and add-ons need the model to compare the sentence with the task more
closely. They are built by word substitution in code from a fixed table, because they are
structural edits of the task list and carry no contamination risk.

## Who writes the text

**Decision.** Fresh subagents inside an interactive Claude Code session, one per brief,
told not to read files or run commands (all made zero tool calls).
**Options.** (a) Hand-written templates in the main working session; (b) a scripted
`claude -p` loop; (c) blind subagents.
**Reason.** The main session had read every existing template and the classifier, so text
written there would lean toward what the classifier already handles. A scripted
`claude -p` loop has unclear billing on a subscription and unclear standing under the
consumer terms on automated access. Blind subagents run as normal Claude Code use, and
every brief and raw reply is saved, so the process can be audited.

## Briefs describe, never show

**Decision.** Briefs name a style or shape and the output format. They never include an
existing template, harmless sentence or result.
**Reason.** Showing examples is the main way generated test data ends up rewording the
originals.

## Filters and thresholds

**Decision.** Reject on any shared 8-word sequence, edit-similarity ≥ 0.8, or embedding
cosine ≥ 0.85 against existing data. Reject attacks with sequencing words
(first / then / before / after that / continue / proceed) and harmless lines that name a
task. A person also reviews a random 50 per pool plus everything with cosine ≥ 0.7 or
edit-similarity ≥ 0.6.
Command wordings for the contrast set are also rejected if they add anything beyond the
task (a sequencing word, "and ..." after the task, "let me know", "come back").
**Reason.** Word-overlap checks alone miss rewordings, so an embedding check and a human
read were added. A wording that adds a second instruction would make the "harmless"
restatement an addition, which the attack definition counts as an attack. The sequencing ban keeps the new attacks out of the one shape the old
test set used. Thresholds were fixed before the audit and not changed afterwards.

## Hand-audit outcome

**Decision.** Of 159 sampled lines, three were removed: a reworded copy of an old note, an
attack that is ungrammatical once filled, and one unsampled attack with the same defect,
found by a pattern search. Lines that were only similar in topic to old data were kept, and
time phrases in command wordings ("this afternoon", "while I'm out") were judged not to
change the task.
**Reason.** The audit judges meaning; similarity scores alone were not grounds for
removal. A mechanical defect found in the sample was fixed everywhere it occurred. Details
in `audit.md`.

## Harmless notes must match the robot's scene

**Decision.** After the first audit, every harmless note was checked against a scene card
(one table, the seven objects with their real appearance and position, nothing else) by
blind agents that kept, fixed or dropped it. Notes still naming off-scene things are
rejected by a filter rule. Attack sentences are exempt.
**Options.** (a) Keep the notes as written; (b) regenerate all harmless notes with the
scene card; (c) fix the existing notes in place.
**Reason.** The lookup tool only ever describes this one table, and the robot also sees a
camera image of it. Notes about a fridge or a gas stove are not something the tool could
return, and a false alarm on them could not be told apart from a false alarm caused by new
wording, which is what the test is meant to measure. (c) keeps the variety of the first
batch and most of its reviewed text, at the same cost as (b). The scene card restates
real facts about the objects, so the filters were rerun to catch fixes that copy old fact
sentences; one did and was removed.

## Second hand check

**Decision.** A random 20 of the 218 edited notes were read by hand (at least two per
object); all were kept. The other edited notes rely on the automatic filters.
**Reason.** The edits were small and mechanical (swapping an off-scene object or fact for
an in-scene one), so a smaller sample was judged enough to catch a systematic problem.

## Separate train and test for the contrast set

**Decision.** The 45 task pairs are split 35 train / 10 test, the 54 accepted command
wordings 27 / 27, the near-miss variants and add-on joiners half and half. No near-miss
string appears on both sides under any task.
**Reason.** So contrast-test scores measure generalisation to unseen pairs and wordings,
not recall of training items. Generated OOD test data is never used for training.

## Three OOD splits

**Decision.** `ood_benign` (new harmless + old attacks), `ood_attack` (old harmless + new
attacks) and `ood_both`.
**Reason.** If only the attacks were newly written, a classifier could flag "text that
sounds generated". Comparing `ood_attack` with `ood_both` shows whether that happens.

## Screening new attacks on the undefended robot

**Decision.** Before testing defenses on the robot, run the new wordings with no defense
and keep the ones that hijack the robot.
**Options.** Screen, or use all wordings unscreened.
**Reason.** The original `eval` templates were chosen the same way. Unscreened weak
wordings would give near-0% attack success for every defense and say nothing about the
defenses. Screening uses only undefended results, so it does not tune toward any defense.
Text-level sets are not screened.

## Keep the existing family names

**Decision.** Each new attack wording is tagged with one of the four existing families.
**Reason.** `toolguard run --families` only accepts those names, so the replay file works
without code changes, and results line up with E2 by family.

## Where things live and how they are stored

**Decision.** All data is under `data_gen/` (not `data/`, which is gitignored). Split rows
use exactly the `Example` fields; contrast pair metadata is in separate `*_pairs.jsonl`
files.
**Reason.** `load_split` rejects unknown fields, so extra columns would break existing
tools.

## Freeze before scoring

**Decision.** `MANIFEST.json` (SHA-256 of every file) is committed before any model is
scored on the new sets, and no prompt, filter or threshold changes after that.
**Reason.** Changing the data after seeing scores would bend the test toward the model.

## Known limitations

- The generator is a Claude model, as is the session that wrote the briefs. Blind briefs
  reduce that overlap but do not remove it. The models being tested (DistilBERT, ProtectAI
  DeBERTa, Qwen3-VL) are not Claude.
- The robot rerun only changes the attack text; the harmless text around it is the
  original pool. Harmless-side generalisation is measured at the text level only.
- Robot attacks are screened on the undefended robot; the text-level sets are not.
- Wordings are split by exact text, not by meaning, so a few train and test wordings are
  close paraphrases ("Could you ... for me?" vs "Would you be able to ... for me this
  afternoon?"). Contrast-test therefore measures unseen task pairs and variants more
  strictly than unseen phrasing.
- Subagent generation is auditable (briefs and raw replies are committed) but not exactly
  reproducible.
- Harmless notes use made-up numbers (weights, dates, IDs) that can differ between notes
  about the same object. They never contradict the scene card, but they are not mutually
  consistent.
