# tool-choice-contrast

A frozen evaluation corpus that holds the user's question fixed and varies only
the offered tools, so the correct decision changes while the question does not.

It exists because of a measurable hole. A tool-calling fine-tune of Qwen3.5-2B
improved its tool metrics and then refused 100% of bare GSM8K questions its base
model answered at 67.4%. No gate caught it: the frozen held-out set contains zero
`ANSWER` examples, so `no_call_accuracy` has been `0.0` for every model ever
measured, including the baseline. The metric existed and never had a denominator.

## The design

| family | arm | tools offered | gold |
|---|---|---|---|
| S | `S-none` | none | `ANSWER` |
| S | `S-one` | 1 irrelevant | `ANSWER` |
| S | `S-five` | 3–5 irrelevant | `ANSWER` |
| E | `E-applicable` | gold tool, every required parameter bound | `CALL` |
| E | `E-underspecified` | same tool, required set a strict superset | `CLARIFY` |
| E | `E-absent` | gold tool substituted by a distractor | `UNSUPPORTED` |

900 items over 300 stems. Every arm of a stem shares a byte-identical question;
that invariant is gate G3 and without it every number is void.

## Status

Under construction. Nothing is frozen and nothing has been measured.

## Licence

Apache-2.0 for the code, CC-BY-4.0 for the corpus. See `LICENSE` and
`DATA_LICENSE`.
