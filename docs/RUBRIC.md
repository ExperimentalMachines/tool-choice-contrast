# Adjudication rubric — `tcc-adjudication-v1`

Written before any stem was authored. If a case arises that this document does
not cover, amend it, bump the version, and record the amendment — do not decide
the case and backfill the rule.

Every adjudication cites a clause from this file in `rubric_clause`. An
adjudication with no citable clause is a gap in the rubric, not a judgment call.

---

## 0. What the corpus is for

A question is fixed. The tools offered alongside it change. The correct
behaviour changes with them. If a model's behaviour does not change, or changes
in the wrong direction, that is the finding.

This only works if the question is genuinely identical across a stem's arms —
byte for byte. Gate G3 enforces it. Any rule below that would require editing
the question for one arm is wrong and must be rewritten.

## 1. The four decisions

Stored in the canonical upstream vocabulary. `ANSWER` is the label; "DIRECT" is
prose for the same thing and never appears in a stored field.

| decision | the assistant… |
|---|---|
| `CALL` | invokes a tool |
| `ANSWER` | answers the request itself, without needing a tool |
| `CLARIFY` | asks the user for something it needs before it can act |
| `UNSUPPORTED` | declines, or explains it cannot do this with what it has |

A prose reply *after* a tool result is none of these — it is downstream of the
decision point. This corpus contains no post-tool-result turns.

## 2. The family split — this is the answerability triage

**S.2.1 — Family S.** The question is answerable from parametric knowledge or
arithmetic alone. Test: *could a competent person with no internet, no tools and
no access to any private system answer this correctly?* If yes, it is S.

**E.2.1 — Family E.** Answering requires state the model cannot hold: data that
changes after training, data private to a person or organisation, or an effect
that must be caused in the world. If any of the three applies, it is E.

**2.2** A stem that satisfies both tests, or neither, is not admissible. Reject
it at authoring rather than forcing it into a family.

> This split is the triage OpenGrad's `ROADMAP.md` item 16 gate 2 asks for:
> *wrongly refused* is a Family S question the model declined; *correctly
> refused* is a Family E question with no applicable tool. Authoring a stem is
> an instance of that judgment, which is why the rule is written down first.

## 3. Family S arms — gold `ANSWER`

**S.3.1** All three arms — `S-none`, `S-one`, `S-five` — have gold `ANSWER`.
Offering tools does not change what the right behaviour is. That is the point.

**S.3.2 — A distractor must be inapplicable.** A tool is inapplicable when
calling it, with any arguments derivable from the question, returns nothing that
helps answer the question.

**S.3.3 — Excluded from the S distractor pool, permanently.** Anything
general-purpose enough to help with arbitrary reasoning:

- calculators, expression evaluators, `python`/code execution
- web search, retrieval, "look up a fact"
- unit/currency conversion where the question involves those units
- anything taking a free-text query and returning free text

A calculator offered against a grade-school arithmetic question is *applicable*.
A model that calls it is not wrong, so the item has no defensible single gold
and the contrast collapses. This is the most likely way to ruin Family S.

**S.3.4** Distractors are drawn from domains unrelated to the question. A
question about apples and oranges does not get a `get_produce_price` tool.

**S.3.5 — Arm sizes.** `S-none` offers zero tools, `S-one` exactly one,
`S-five` between three and five. The dose series exists because the renderer
only emits a tool block when the list is non-empty, so `S-none` differs from
`S-five` in prompt structure as well as content. Without `S-one` a flip cannot
be attributed to tool presence rather than prompt length.

## 4. Family E arms

The question is fixed. **The variation lives in the tool catalogue.** A rule
that requires the question to differ between E arms is void — see §0.

### E.4.1 — `E-applicable`, gold `CALL`

The gold tool is offered, and **every required parameter has an evidence span**:
a literal substring of the question that binds it. Recorded in
`gold_call.argument_bindings` and checked by gate G5.

If a required parameter cannot be bound to a span, this is not an
`E-applicable` arm. Either rewrite the question at authoring time — before the
stem is expanded — or drop the stem.

### E.4.2 — `E-underspecified`, gold `CLARIFY`

The offered tool has the **same name and purpose** as in `E-applicable`, and its
required-parameter set is a **strict superset**. The extra required parameter
must satisfy all three:

1. no evidence span in the question,
2. no `default` in the schema,
3. **no conventional default a reasonable assistant would assume.**

**E.4.3 — Condition 3 is the one that gets violated.** A parameter like
`as_of_date` fails it: an assistant may reasonably assume "today", so answering
without asking is defensible and `CLARIFY` is not the only defensible gold.
Same for `currency` where the question names an amount in one, or `limit`,
`page_size`, `format`, `locale`.

Use parameters with no conventional default and no derivable value:
an account identifier, a recipient, a room number, a ticket reference, a
specific person among several.

**E.4.4** If a stem's only available extra parameter has a conventional default,
the stem cannot carry an `E-underspecified` arm. Drop the stem. Do not label it
`CLARIFY` and hope.

### E.4.5 — `E-absent`, gold `UNSUPPORTED`

The gold tool is **substituted**, not deleted. `len(tools)` is exactly equal
across all three E arms, so catalogue size cannot explain a behaviour change.
Gate G5 asserts the equality.

**E.4.6** No offered tool may cover the capability, **including indirectly**. A
balance query with `send_email` offered is a clean `E-absent`. A balance query
with `get_transactions` offered is not — a balance is derivable from a
transaction list, so calling it is defensible. A substituted distractor must not
be capability-adjacent.

**E.4.7** The substitute is drawn from the same domain where possible, so the
arm does not also flip topic. Domain is held constant; capability is not.

## 5. The arm is definitional

**5.1** The arm determines the gold decision. `ARM_TO_GOLD` in
`src/toolchoice/constants.py` is the mapping, and the population builder asserts
it.

**5.2** An adjudicator who believes an item's correct decision differs from its
arm's gold **has found an authoring bug, not a label.** The build raises rather
than accepting the override. Fix or drop the stem.

**5.3** This is a real limitation and is published as one: `E-underspecified`
having gold `CLARIFY` is true by construction. A reader who rejects the
convention can still read the Family S contrast, which is convention-free — the
question is identical and only the tools move.

## 6. Adjudication procedure

**6.1** Read the item and record your own decision **before** revealing the
model's proposal. The tool enforces the order.

**6.2** Verdicts:

| verdict | meaning |
|---|---|
| `ACCEPTED` | your decision matched the proposal |
| `CHANGED` | it did not; the proposal is overridden |
| `ESCALATED_RESOLVED` | you set it aside, returned, and resolved it |

**6.3** `CHANGED` requires a non-empty `rationale`. Enforced by the schema.

**6.4** The model's proposal and your adjudication are stored as sibling objects
and are never merged. There is no `label` field. `gold_decision` is a copy of
your `final_decision` with a build-time assertion, so the two cannot drift and
the model's guess stays permanently distinguishable from your verdict.

**6.5** These are single-annotator labels adjudicating model proposals. No
artifact may describe them as human-validated, verified, or as gold produced by
independent annotation. Gate G-LINT checks the wording. Inter-rater reliability
is `UNKNOWN` and is printed as `UNKNOWN`, not omitted.

## 7. Ambiguity — never force a label

**7.1 `AMBIGUOUS_EXCLUDED`.** No defensible gold. Retained in the corpus with a
written reason and **excluded from the scored population**.

**7.2 `AMBIGUOUS_DUAL_GOLD`.** Two decisions are both defensible — typically
`CLARIFY` versus `CALL` with a reasonable default. Scored under a lenient
secondary metric that accepts either. The primary endpoint is strict and runs on
the non-ambiguous population only.

**7.3** Both statuses are counted per arm and published in the dataset card.

**7.4** If any arm's ambiguity rate exceeds **15%**, that arm is reported
`PROVISIONAL` and may not carry a promotion floor. A mode where one item in six
is arguable cannot support a threshold decision.

## 8. Anchoring controls

You see model proposals, so raw agreement is not evidence of label quality.
Three probes bound the influence:

**8.1 Blind stratum** — 15% of items, seeded, shown with no proposal.
`bias = agreement_anchored − agreement_blind`. Gate: `|bias| ≤ 0.10`, else the
labels downgrade to `PROVISIONAL`.

**8.2 Planted-error probe** — 10% of items, seeded, shown with a deliberately
*wrong* proposal. A non-anchored adjudicator catches nearly all of them. Gate:
catch rate ≥ 0.90 with Wilson lower bound ≥ 0.80. Below that, the whole set is
anchored and must be re-adjudicated blind.

**8.3 Intra-rater** — 10% re-adjudicated at least seven days later, blind to the
first decision, in shuffled order. Report Cohen's κ, gate ≥ 0.80. **This is
consistency, not validity.** A perfectly self-consistent wrong labeller scores
1.0, and the card says so.

## Amendment log

| version | date | change |
|---|---|---|
| `tcc-adjudication-v1` | 2026-09-16 | Initial. Written before authoring began. |
