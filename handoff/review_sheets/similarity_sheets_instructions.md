# Similarity sheets: test clusters, twin thresholds and contamination

Three sheets ask the team to judge pairs of texts that a similarity score linked. A score only finds
candidates; a person decides what each pair is. This guide covers all three.

| Sheet | Rows | Decision per | People | When | Time |
|---|---|---|---|---|---|
| `tier_k_test_clusters.csv` | 10 (5 clusters of 2 test questions) | cluster | one | before the freeze (13 Nov) | about 10 minutes |
| `tier_k_twin_threshold_review.csv` | 235 pairs of training questions | pair | one | before the freeze | about 4 hours |
| `contamination_tier_k_test.csv`, `contamination_tier_k_dev.csv`, `contamination_tier_f_test.csv`, `contamination_tier_f_dev.csv` | 188, 30, 17 and 6 pairs | pair | two (section C) | after the freeze | about 4 hours, plus about 1.5 hours for the second person |

The two Tier-A contamination sheets have no flagged pairs (a header row only), so there is nothing to
do in them. The order matters: the twin review can move the lexical threshold, which is also the
TF-IDF threshold of the contamination audit, and the item sets must be final before their flags are
judged.

## Rules for all three sheets

- **No model sees these sheets.** Never paste anything from them into ChatGPT, Claude or any other
  model or online tool (eval plan §6). The cluster sheet and the two test contamination sheets hold
  test questions and tasks.
- **Work alone on your own copy.** Copy the sheet to `<sheet name>_<your initials>.csv` and fill in
  only the decision column, `reviewer` (your initials) and `notes`. Do not discuss pairs until every
  copy is in.
- **Judge from the texts.** The score columns (`band`, `score`, `linked_now`, `linked_to (evidence)`,
  `signals`, `ngram_shared`, `tfidf`, `embed`) say why a pair is in the sheet, not what it is.
  Questions written from the same template often score high without being the same question.
- **Write values exactly as shown** (lower case, the same spelling; table at the end), so the sheets
  can be read by code.
- **Disagreements go to the Qoherent expert**, as on every other sheet. The coordinator writes the
  final decisions into the sheets in `handoff/review_sheets/`.

## A. Test clusters (`tier_k_test_clusters.csv`)

Linked test questions share a `cluster_id`, and the error bars count a cluster once. Two questions
are linked by a shared word 8-gram, by the dataset team's `keep_both` decision in its dedupe step, or
by TF-IDF similarity. Five clusters have two questions each. The sheet has one row per question, and
`linked_to (evidence)` reads `<other id>: <evidence>`, for example `8gram:<the shared phrase>` or
`lexical:<TF-IDF cosine>`.

Fill in `team_decision (same family / split out)` for each cluster:

- `same family`: the two questions test the same knowledge or skill, so a model that answers one is
  likely to answer the other;
- `split out`: they share only wording, for example a stock opening phrase, and test different
  things.

Write the same value on both rows of a cluster, with a one-line reason in `notes`. When unsure, write
`same family`: keeping two questions together only widens the error bars a little, while splitting
a real family makes them too narrow. The coordinator removes the link of every `split out` cluster
before the freeze, and its two questions then form clusters of their own.

## B. Twin thresholds (`tier_k_twin_threshold_review.csv`)

Two training questions are **twins** when an answer to one would substantially answer the other. A
validation question serves as a dev question only if no twin of it is left in training
(`reports/tier_k_dev_split.md`), so the twin groups decide the 107 dev questions. Pairs that share a
word 8-gram are always twins. Other pairs are linked when their score reaches a threshold: 0.92 on
the dataset's semantic similarity score, 0.55 on TF-IDF cosine. Claude set both thresholds by reading
every pair in the score bands around them; this sheet is the team's check of that reading. Every
question in the sheet is a training question.

Each row is one pair: `source` (`semantic` or `lexical`, the score used), `band`, `score`,
`linked_now` (whether the current threshold links the pair), the two ids, where each question came
from (`origin_a`, `origin_b`) and the two questions.

Fill in `team_label (twin / not twin)` for every pair:

- `twin`: an answer to one question would substantially answer the other, for example the same
  question reworded, or the same ask with details that do not change the answer;
- `not twin`: the questions share a topic or wording but ask different things, so an answer to one
  leaves the other largely unanswered;
- `unsure`: only when you cannot decide; write why in `notes`. The Qoherent expert labels these
  pairs.

Label every pair before you open `reports/tier_k_dev_split.md`, which shows Claude's counts per band,
and ignore `linked_now` while you label.

The coordinator then counts the twins in each band. A threshold is the lower edge of the lowest band
in which at least half of the pairs are twins. If the team's labels give 0.92 and 0.55, nothing
moves. If they give another threshold, the coordinator rebuilds the dev split with it and records
the deviation in the pre-registration (§10).

## C. Contamination (`contamination_<tier>.csv`)

The contamination audit compares every benchmark item with every training corpus: training
questions, teacher answers, repaired answers, SFT sessions, task specs and agent traces. It flags a
pair when the two share a word 8-gram, or their TF-IDF cosine is at least 0.55, or their embedding
cosine is at least 0.88 (`reports/contamination.md`). Nothing is removed from any dataset. The team's
decisions give each tier's confirmed contamination rate, which is reported next to its flagged rate
(pre-registration §2).

Each row is one item and one training document:

- `item_id`, `corpus` and `doc_id` (for an answer, `<model>/<question id>`);
- `doc_ref`: the training question or task the document belongs to;
- `signals`: the signals that flagged the pair, joined by `+`, and their scores;
- `item_text`: the first 300 characters of the item;
- `training_text`: the part of the training document that matched, at most 300 characters.

If the texts are cut off and you need more, ask the coordinator for the full item and training
document.

Fill in `decision` for every pair:

- `contaminated`: the training text contains this item. That is the same question or task, reworded
  or not, with the same specifics (numbers, blocks, settings), or a worked answer to this item's
  exact case, so a model could answer the item by recalling it;
- `not contaminated`: the training text teaches the knowledge or skill the item tests, but through a
  different question or case, or it shares only a stock phrase or the topic;
- `unsure`: you cannot tell from the texts.

For every `contaminated` or `unsure`, write in `notes` what matches, for example "same question, same
numbers" or "answer works out the same 48 kHz case".

**Two people:**

1. Reviewer 1 labels every row of the four sheets.
2. The coordinator gives reviewer 2 a copy holding every row reviewer 1 marked `contaminated` or
   `unsure`, plus every fifth row of each sheet (data rows 5, 10, 15, …). Reviewer 1's columns are
   left empty in that copy, and reviewer 2 labels these rows alone.
3. Where both agree, that is the decision. Where they disagree, or either wrote `unsure`, the
   Qoherent expert decides. A row only reviewer 1 saw keeps reviewer 1's decision.
4. The coordinator writes the final `decision`, both reviewers' initials in `reviewer` (separated by
   `;`) and the notes into the four sheets in `handoff/review_sheets/`. `./gb contamination` keeps
   these three columns on every re-run.

An item counts as contaminated when at least one of its pairs is finally `contaminated`. The
confirmed rate of a tier is the share of its items that count as contaminated. The agreement of the
two reviewers is reported on the every-fifth-row rows.

A row whose `signals` reads `not flagged in this run` was judged in an earlier run and is no longer
flagged. Leave it as it is.

## Allowed values

| Sheet | Column | Values |
|---|---|---|
| `tier_k_test_clusters` | `team_decision (same family / split out)` | `same family`, `split out` |
| `tier_k_twin_threshold_review` | `team_label (twin / not twin)` | `twin`, `not twin`, `unsure` |
| `contamination_<tier>` | `decision` | `contaminated`, `not contaminated`, `unsure` |
| all three | `reviewer` | initials; in the final contamination sheets both reviewers' initials, separated by `;` |
