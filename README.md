# Photo Geolocator Agent

Agent for the make.fun bounty [photo-geolocator-01](https://make.fun/bounty/photo-geolocator-01): given 25 street-level photos, it estimates where each was taken.

## How it works

For each photo, `main.py` asks `claude-sonnet-5-5` once through the make.fun model gateway (one call per photo, four in parallel), parses the latitude/longitude out of the answer, and posts all 25 answers to the trial env at the end. A photo whose model call is refused still gets a deterministic fallback guess picked from a short list of major world cities by the photo's own bytes, which beats leaving it unanswered.

## Run

```sh
python3 -P -s main.py
```

Requires the trial environment: `MAKEFUN_ENV_URL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` (or the OpenAI equivalents). Standard library only.
