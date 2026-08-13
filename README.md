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

> **Status: in progress.** Numbers will appear here once the benchmark has run.
> Nothing in this README is claimed before it is measured.

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
