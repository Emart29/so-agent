# so-agent

**Structured output is solved. This measures what it costs you.**

Getting an LLM to return valid JSON was a 2023 problem, and native constrained
decoding solved it. So this is not another retry loop. It measures the three
things that broke instead:

- **Does enforcement cost reasoning quality?** Semantic accuracy at each
  enforcement tier, not just parse rate. Nearly everyone assumes enforcement is
  free; nobody publishes the per-model measurement.
- **Is structurally valid output actually correct?** A green schema check now
  hides a wrong answer that would previously have crashed.
- **What does per-call reliability mean once you chain it?** 99% per call is
  roughly 60% success across a 50-step agent trajectory.

The headline artifact is a measured matrix: parse-failure rate and semantic
accuracy by provider, model, enforcement tier, and schema complexity — from real
API calls, against a live provider, on a stated date.

> **Status: in progress.** The capability matrix below is measured. The
> benchmark numbers are not in yet, and nothing is claimed here before it is
> measured.

## The measured capability matrix

Groq, 14 August 2026. Every cell is a real API call, not a documentation claim.

| model | json_schema | json_object | tools | prompt_only |
|---|---|---|---|---|
| `openai/gpt-oss-120b` | yes | yes | yes | yes |
| `openai/gpt-oss-20b` | yes | yes | yes | yes |
| `qwen/qwen3.6-27b` | yes | yes | **ignored** | — |
| `llama-3.3-70b-versatile` | no | yes | yes | yes |
| `llama-3.1-8b-instant` | no | yes | yes | yes |
| `allam-2-7b` | no | yes | no | yes |
| `groq/compound-mini` | no | yes | no | yes |
| `groq/compound` | no | no | no | — |

**Three of eight models enforce a native JSON schema.** The rest reject the
directive outright, which is why the enforcement ladder exists rather than being
a formality — on most of this provider's line-up there is nothing to fall back
from.

**`qwen/qwen3.6-27b` accepts a tool definition and emits no tool call.** The
request succeeds and nothing is enforced. That is the failure this project was
built to detect: indistinguishable from working code until the output is checked.

### What strict mode will not accept

Probed against every model that enforces natively, and identical on all three:

| schema feature | result |
|---|---|
| nested objects, arrays of objects, enums, `anyOf` unions, `$ref`/`$defs` | accepted |
| nullable field (`"type": ["string", "null"]`, still in `required`) | accepted |
| **optional field omitted from `required`** | **rejected everywhere** |
| maximum nesting depth | 6 |

That last rejection is the one that matters for anything built on Pydantic.
Pydantic's natural output for an optional field leaves it out of `required`, and
strict mode refuses it: *"`required` is required"*. Optionality has to be
rewritten as a nullable type with the field still required — which is a
translation step, not a configuration flag.

### The same model on two serving stacks

`openai/gpt-oss-20b` is served by both Groq and OpenRouter, so running the probe
against both holds the weights constant and varies only the infrastructure.

**Enforcement behaviour is identical.** All four tiers conform on both, and both
accept schemas to a nesting depth of 6. Whatever constrained decoding these
providers do, they do it the same way for this model.

**Schema acceptance is not identical:**

| schema feature | Groq | OpenRouter |
|---|---|---|
| optional field omitted from `required` | **rejected (400)** | accepted |
| everything else probed | accepted | accepted |

So a schema that works on one provider returns a 400 on the other, for the same
model. The difference is in the serving stack's schema validator rather than in
the model, and it is exactly the kind of thing that turns a provider switch into
an afternoon of debugging. The translation layer has to target the stricter of
the two, not whichever one happened to be developed against.

## Providers

Any OpenAI-compatible endpoint. A provider is a row in `provider/registry.py`,
never a code path — Groq, OpenRouter, OpenAI, Together, Cerebras, and local
Ollama or vLLM servers all work through the same client.

**Groq is the default**, because it is the best place to see the problem: open
models where enforcement support genuinely varies, and a free tier that covers a
benchmark of this size. **OpenRouter** is used for one slice, comparing the same
model across different serving stacks so any difference is attributable to
infrastructure rather than weights.

Neither requires a payment method. That is deliberate: the benchmark should be
reproducible by a reader who cannot or will not add a card.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY
python examples/check_provider.py
```

`check_provider.py` confirms credentials work, lists the models your key can
reach, and prints one normalised completion with its metadata.

## Model ids are measured, not hardcoded

Provider line-ups change often enough that any model id written into this repo
would eventually be wrong. Nothing here hardcodes one: models are discovered from
the provider, probed for what they actually support, and the measured result is
cached to disk.

That includes the case docs never cover — a provider that *accepts* a
`response_format` and then ignores it. It looks like enforcement and isn't, and
the only way to catch it is to check the output actually conformed.

## Licence

MIT
