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

Replies are parsed in the model's native tool format: `qwen` by default, `lfm`
for LFM2.5's `<|tool_call_start|>[name(arg="v")]<|tool_call_end|>` calls. Every
score names the format it was parsed with. A reply without a tool call is decided
by the same prose rules under both formats.

Those prose rules default to the keyword rules in the vendored OpenGrad parser,
which miss declines such as "I don't have access to banking systems". `tcc judge`
asks a pinned local model to decide each reply without a tool call and writes the
judgments to a file. `score(..., judgments=read_judgments(path))` uses them in place
of the keyword rules and never calls a model itself. Every score names its prose
classifier: the keyword rules, or the judge's model, digest and prompt hash.

## Status

Under construction. Nothing is frozen and nothing has been measured.

## Licence

Apache-2.0 for the code, CC-BY-4.0 for the corpus. See `LICENSE` and
`DATA_LICENSE`.
