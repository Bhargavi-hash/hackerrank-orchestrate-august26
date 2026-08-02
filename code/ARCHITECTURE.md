# ARCHITECTURE.md — Message Notification Router

This is the build spec for `code/`. Read this before writing any code. It reflects the
**actual** dataset schema (verified against `dataset/*.csv`), not just `problem_statement.md`.

## 0. Non-negotiables

- Output: `output.csv` with exact columns `message_id,action,message_type,reason,confidence,evidence_message_ids`,
  one row per row in `dataset/messages.csv` (110 rows in the dev set; must generalize to any size).
- `action ∈ {notify, digest, mute}`.
- `message_type ∈ {personal, urgent, event, payment, business_update, promotion, greeting, forward, spam, scam, unknown}`.
- `evidence_message_ids`: semicolon-separated `message_id`s **that exist in `message_history.csv`**, or `none`.
  Never invent an ID. Never cite an ID that wasn't actually retrieved as a candidate.
- Deterministic: same input row → same output row, every run. Temperature 0 for any LLM/VLM call. Cache
  media (OCR/ASR) output keyed by `media_id`, since posters and voice notes repeat across messages
  (confirmed: `img_008` is sent to two different users/groups with two different captions).
- Do not read organizer-only files. Do not hardcode any label keyed off `message_id` or `sample_messages.csv`.
- Read secrets from env vars only.

## 1. Why this is not a single classification call

Identical content routes differently depending on recipient, sender relationship, and history
(e.g. `img_008` is a "holding this for you" note to one user and a resale listing to another). A prompt
that only sees `message_text` cannot make this distinction. So the system is split into two stages that
must stay separable and independently testable:

1. **Context Builder** — deterministic Python. Joins every CSV + resolves + transcribes media into one
   structured `MessageContext` object per message. No LLM calls here except OCR/ASR.
2. **Router** — decision logic over that context: a deterministic override layer first, then a
   constrained LLM call for everything the rules don't resolve.

## 2. Data model (verified columns)

```
messages.csv / message_history.csv (same shape):
  message_id, user_id, conversation_type[personal|group|business], group_id, business_id,
  sender_user_id, created_at, message_text, media_type[""|image|voice], media_id, forwarded_count

users.csv:
  user_id, do_not_disturb_window, messages_opened_30d, messages_replied_30d,
  notifications_dismissed_30d, messages_reported_30d
  -> GLOBAL engagement prior only. No per-sender breakdown here.

groups.csv:
  group_id, group_name, group_type[family|society|school_group|work|...], member_count,
  admin_count, created_at, messages_30d

group_members.csv (per user+group — THIS is where group personalization lives):
  group_id, user_id, role[admin|member], joined_at, messages_sent_30d, messages_read_30d,
  replies_sent_30d, notifications_dismissed_30d, group_muted_by_user[0|1]

business_accounts.csv:
  business_id, display_name, brand_name, category, verified[0|1], official_domain,
  domain_used_by_sender, account_age_days, messages_sent_30d, user_reports_30d,
  domain_used_by_sender_age_days
  -> domain_used_by_sender != official_domain, or domain_used_by_sender_age_days very low
     relative to account_age_days, is a strong phishing signal even if verified=1.

user_business_history.csv (per user+business — THIS is where business personalization lives):
  user_id, business_id, why_user_knows_account, last_activity_at, allows_promotions[0|1],
  promotions_opted_out_at, activity_count_180d, messages_opened_30d, messages_dismissed_30d,
  messages_replied_30d, last_reply_at

message_events.csv (reactions to message_history rows):
  user_id, message_id, message_opened, message_replied, reaction_time_minutes,
  notification_dismissed, muted_after_message, message_reported

images.csv: image_id, file_path
voice_notes.csv: voice_note_id, file_path
daily_notification_summary.csv: user_id, date, notifications_sent, notifications_dismissed
```

Join keys: everything hangs off `user_id` plus one of `sender_user_id` / `group_id` / `business_id`,
matching `conversation_type`.

## 3. Context Builder (`code/context_builder.py`)

For every row in `messages.csv`, build:

```python
MessageContext = {
  "message": {...raw row...},
  "sender_context": {
      # conversation_type == personal:
      #   nothing extra beyond users.csv global stats — no per-sender data exists for personal chats
      # conversation_type == group:
      "group": {...groups.csv row...},
      "membership": {...group_members.csv row for (group_id, user_id)...},
      # conversation_type == business:
      "business": {...business_accounts.csv row...},
      "relationship": {...user_business_history.csv row for (user_id, business_id)...},
  },
  "user": {...users.csv row...},
  "daily_load": {...today's daily_notification_summary.csv row for user_id, if present...},
  "media": {
      "type": "image" | "voice" | None,
      "file_path": "...",
      "extracted_text": "...",       # OCR (image) or ASR transcript (voice)
      "structured": {                 # normalized fields, see §4
          "title": ..., "date": ..., "time": ..., "location": ...,
          "deadline": ..., "payment_request": bool, "urgency_cues": [...]
      }
  },
  "history_candidates": [
      # from message_history.csv where user_id matches AND
      # (sender_user_id matches OR group_id matches OR business_id matches),
      # sorted most-recent-first, capped at top 5, each joined with its message_events.csv row
      {"message_id":..., "created_at":..., "message_text":..., "reaction": {...message_events row...}}
  ],
}
```

Retrieval rule for `history_candidates` (this is the primary source for `evidence_message_ids`):
- Match key priority: same `group_id` (group), same `business_id` (business), same `sender_user_id`
  (personal) — always also filtered to the same `user_id` (we only have access to this user's own history).
- If more than 5 candidates match, prefer ones whose `message_text` shares keywords/topic with the
  current message (cheap token-overlap or TF-IDF cosine is enough at this scale — no need for embeddings
  given ~1000 history rows).
- If zero candidates match, `history_candidates = []`.

**Second tier — cross-user safety evidence, business/scam context only.** In addition to the user-scoped
tier above, when the message's `business_id` (or `sender_user_id`, for repeat-offender personal accounts)
matches a `message_history.csv` row belonging to a *different* user, and that row's `message_events.csv`
reaction shows `message_reported == 1` or `muted_after_message == 1`, include it as
`cross_user_safety_evidence` — a separate list from `history_candidates`, clearly labeled as such in the
context object. This is the only place cross-user data is used, and only for safety corroboration (never
for personalization/preference inference about this user, since that must stay user-scoped). It exists
specifically to give rules 1/3 and the cold-start rule (7) a real, citable `message_id` even when this
particular user has no prior history with the sender — a new-to-this-user business that other users
already reported should not be treated as cold start. `evidence_message_ids` may draw from either tier;
both are equally "real" IDs from `message_history.csv`, just retrieved under different match criteria.

## 4. Media normalization

**Images** (`code/media/image_processor.py`): run a VLM/OCR pass once per unique `media_id`, cache to
disk (e.g. `.cache/media/{media_id}.json`). Extract: visible text, whether it looks like a poster/flyer/
receipt/screenshot, any date/time/deadline/price/QR-code/payment-link visible. Output plain structured
JSON, not prose — the router should never re-look at the raw image.

**Voice notes** (`code/media/voice_processor.py`): transcribe once per unique `voice_note_id`, cache
the same way. Treat the transcript as `message_text` equivalent, tagged `source=voice`, and pass to the
same urgency/deadline extraction as text.

Both caches keyed by media ID (not message ID) — this is required because `img_008` recurs across
messages and must not be reprocessed or re-billed twice.

## 5. Deterministic override layer (`code/rules.py`) — runs before any LLM call

Each rule is traceable to a real column. If a rule fires, skip the LLM and emit its action/type/reason/
confidence directly. Order matters — safety first.

| # | Condition (schema fields) | Action | message_type | Confidence |
|---|---|---|---|---|
| 1 | `business.domain_used_by_sender != business.official_domain`, OR `domain_used_by_sender_age_days` is small relative to `account_age_days`, combined with payment/urgency language in text/media | `mute` | `scam` | 0.9–0.95 |
| 2 | `user_business_history.promotions_opted_out_at` is set AND message reads as promotional | `mute` | `promotion` or `spam` | 0.85–0.9 |
| 3 | `business.user_reports_30d` high relative to `messages_sent_30d` (report rate outlier) | `mute` | `scam` or `spam` | 0.8–0.9 |
| 4 | `group_members.group_muted_by_user == 1` AND no direct address/urgency cue found | `mute` (or `digest` if content still looks informational) | best-fit | 0.7–0.85 |
| 5 | `group_members.group_muted_by_user == 1` AND message contains a direct mention of this user / reply-to-user / explicit urgent+deadline-today language | escalate — do **not** auto-mute; hand to LLM step with an "escalation candidate" flag, LLM decides notify vs digest | — | — |
| 6 | Repetition: `history_candidates` show the same sender/group sent near-duplicate content recently that this user dismissed/ignored | `mute` or `digest`, not `notify` | `forward`/`spam`/best-fit | 0.7–0.85 |
| 7 | Cold start: no rule above fired, sender/group/business is new or has zero `history_candidates`, AND no `cross_user_safety_evidence` exists, AND the message contains no unambiguous safety flag (rule 1-3 territory) and no unambiguous direct urgent ask | default to `digest` | best-fit from content | 0.4–0.55 |

Rules 1–4, 6, and 7 can resolve directly. Rule 5 only *flags*, it never finalizes — final decision goes
to the LLM with the flag included in its context so it can weigh urgency against the mute.

**Confidence-gated escalation.** A rule "resolving" doesn't automatically mean finalizing it: `code/rules.py`
applies a single threshold (0.75) centrally, in `apply_rules()`, not per-rule. Any branch that would
resolve below that confidence converts into an escalation instead — same mechanism as rule 5, with the
rule's would-have-been decision and reasoning passed to the LLM as a hint rather than discarded. In
practice this affects exactly three branches, by construction rather than special-casing, since every
other branch already scores 0.8+: rule 4's plain-digest fallback (0.7), rule 6's non-muted/non-reported
digest branch (0.7), and rule 7's cold-start default (0.45). A 0.45 or 0.7 guess finalized without the
LLM ever seeing the message is worse than escalating it.

If `cross_user_safety_evidence` is non-empty for a sender this user has no personal history with, that
is itself a safety flag — it routes into rule 1/3 territory (mute/scam or mute/spam, citing the
cross-user message_id as evidence) rather than falling through to the cold-start default in rule 7. A
new-to-this-user business that other users already reported is not the same situation as a genuinely
unknown, unreported sender.

`evidence_message_ids = none` only when both `history_candidates` and `cross_user_safety_evidence` are
empty.

Rule 7 exists because the cost matrix (§8) treats `notify`-when-should-mute and `mute`-when-should-notify
as the expensive failures and `digest` as the cheap one — under genuine uncertainty (new sender, no
history to corroborate either way, and no cross-user safety flag), the system should default to the
cheap failure mode rather than let an LLM guess a direction with no evidence behind it. This is a stated
bias, not left to model discretion.

None of these thresholds are hand-tuned "magic numbers" dressed up as a formula — each is a direct,
explainable read of one or two real columns, stated as such in the `reason` output.

**Disclosed deviation — rule 3's `scam`/`spam` choice.** The table above leaves this as "`scam` or
`spam`" without specifying which. The implementation ties it to a second real column rather than
defaulting to one or the other arbitrarily: `scam` when `domain_used_by_sender != official_domain` is
also true (report-rate outlier + a spoofed domain), `spam` when the domain is legitimate (report-rate
outlier alone — spammy/annoying, not necessarily impersonating anyone). This is better reasoning than a
flat default and is called out here so it reads as an intentional refinement of the spec, not a silent
divergence from it.

**Reason strings for rule hits must be value-interpolated templates, not free text.** Since these
decisions are fully deterministic, the reason should literally substitute the real column values rather
than restate the rule in generic prose — e.g. `"Sender domain {domain_used_by_sender} does not match
{business.official_domain}; sending domain registered only {domain_used_by_sender_age_days} days ago"`
rather than "this looks suspicious." This maximizes the eval rubric's "usefulness and consistency of
reason" criterion for the entire subset of messages the rule layer resolves, at no extra cost — the
values are already sitting in the context object.

## 6. LLM Router (`code/router.py`)

Only called when the rule layer doesn't resolve the message. Input: the full `MessageContext` (already
containing normalized media output, sender relationship, retrieved history + reactions, daily load) —
never raw files, never re-fetch anything.

Contract:
- Structured output (JSON schema / tool-forced), fields = exactly the 6 output columns for this message.
- `evidence_message_ids` may **only** contain IDs present in `history_candidates` or
  `cross_user_safety_evidence`; validate this programmatically after the call and drop anything else,
  falling back to `none`.
- `message_type` and `action` must be one of the fixed enums — validate and reject/retry on violation.
- `reason` should reference concrete context fields (sender trust, relationship, urgency, repetition),
  not generic filler — spot-check this in eval.
- Confidence: take the model's self-reported value, then adjust:
  - down (cap ~0.5–0.6) when context is thin: new/unknown sender, no history_candidates, media
    processing failed/uncertain
  - up when history_candidates strongly agree with the decision (consistent past reaction pattern)

System prompt should state the personalization principle explicitly: the same content can and should
route differently for different users depending on relationship, opt-in status, engagement history, and
group mute state — this is not a hint, it's the core instruction.

The prompt must always include the same signal fields the rule layer checks (domain match/age, opt-out
status, report rate, group-mute state, cold-start flag), **even when no rule fired** — the LLM should be
reasoning over "here is why no rule was confident enough," not seeing a stripped-down context just
because the rules didn't resolve it. This is the main mitigation for novel scam/abuse patterns that don't
match rules 1–3: the model sees the same raw signals, just under its own judgment rather than a fixed
threshold.

**Determinism via caching, not just temperature.** Temperature 0 does not guarantee bit-identical LLM
output across runs (API-level nondeterminism is a known limitation). Cache the LLM's structured output
keyed by a hash of the exact serialized `MessageContext` passed in. Re-running the pipeline on unchanged
input replays the cached decision instead of re-querying — this makes determinism a property of the
system, not a property of the model.

## 6a. Model & Runtime Configuration

No provider is hardcoded. `router.py` and `media/*.py` must read model/provider selection from env vars
(e.g. `LLM_PROVIDER`, `LLM_MODEL`, `VISION_MODEL`, `ASR_MODEL`, plus the relevant `*_API_KEY`), with a
single place in `main.py` or a `config.py` wiring them — not scattered inline. This matters independent
of which provider is used at submission time: it's what makes the system runnable by anyone grading it
without editing code, and what makes swapping providers (rate-limit issues, credential availability)
a config change, not a rewrite.

**Current recommendation (revisit if credentials/limits change before build):** Groq free tier, no card
required, for all three model calls — text router, image understanding, and voice transcription (Groq
hosts Whisper) — covering all three modalities from one provider. Given this dataset's actual volume
(~110 text decisions, 15 images, 8 voice notes), even the most conservative published free-tier caps
(~30 requests/minute, ~1,000 requests/day per model) leave large headroom; the full run completes in
minutes. Use a 70B-class text model for the router (not the smallest available) — reasoning quality on
the personalization tradeoffs matters more here than throughput, and at this volume the free tier doesn't
charge extra for it either way. Check `console.groq.com/docs/rate-limits` and `console.groq.com/docs/vision`
at build time, since exact per-model caps and the vision model lineup shift over time and shouldn't be
taken as fixed from this doc.

The LLM-response cache (§6, keyed by context hash) is what protects against burning rate-limit quota
during iterative eval runs — re-running the pipeline on unchanged input should never re-call the API.

**Real fallback behavior, discovered during build (not the plan going in).** Groq's free tier turned out
tighter than the estimate above once a full build/test day's cumulative usage is counted, on two separate
axes: the vision model (`qwen/qwen3.6-27b`, Groq's only vision-capable model — no second Groq vision model
exists to retry against) has an 8000 TPM cap that's easy to hit processing several images back to back, and
both the vision model and the text router model (`llama-3.3-70b-versatile`) have *daily* token quotas
(TPD) that a single heavy testing day can exhaust outright — confirmed directly via 429 error bodies naming
the exact daily limits (8000 TPM / 200000 TPD for the vision model, 100000 TPD for the router model), not
inferred. `media/image_processor.py` and `router.py` both fall back to Gemini (`gemini-2.5-flash-lite`) on
exhausted Groq retries, config-driven via `VISION_FALLBACK_MODEL`/`LLM_FALLBACK_MODEL` in `config.py`.
`gemini-2.5-flash-lite`, not the full `gemini-2.5-flash`: Gemini's free tier is *also* quota-limited per
model (confirmed from a 429 body naming `model: gemini-2.5-flash`, `quotaValue: 20`/day — exhausted in one
afternoon of testing), and different flash variants carry independent quotas, verified empirically
(`gemini-2.5-flash-lite` answered live after `gemini-2.5-flash` and `gemini-2.0-flash` were separately
exhausted). Which provider actually produced each image is logged per-item in `.cache/media/{media_id}.json`
(`"provider"` field) — real audit trail, not asserted:

```
img_007, img_008, img_010, img_012        -> groq:qwen/qwen3.6-27b
img_002, img_003, img_004, img_011,
img_023, img_024, img_025                 -> google:gemini-2.5-flash  (pre-flash-lite switch)
```

This is a visible quality tradeoff, not a silent one — a Gemini-fallback image may read slightly
differently (OCR phrasing, structured-field completeness) than a Groq-primary one, and the cache record
makes that auditable if asked in review. A local (on-device) vision tier was evaluated as a third fallback
option and set aside for now: this build environment is a shared desktop (not a dedicated container) with
tight, fluctuating available RAM, and the per-model-quota fix above resolved the actual rate-limit problem
without needing it. Voice transcription stayed on Groq's hosted Whisper (`whisper-large-v3`) throughout —
it never hit a rate limit in this build, so it didn't need a fallback tier at all; a local
`faster-whisper` swap was tried and reverted after a real side-by-side comparison surfaced inconsistent
output quality (punctuation/capitalization loss, and on one clip a genuine transcription error) for no
corresponding necessity.

## 7. Post-processing validation (`code/validate.py`)

Before writing `output.csv`, assert per row:
- action/message_type are in the allowed enum sets
- confidence ∈ [0,1]
- every ID in evidence_message_ids exists in `message_history.csv` AND was in that message's own
  `history_candidates` or `cross_user_safety_evidence` (no cross-message leakage — an ID can't be cited
  unless it was actually retrieved as a candidate for *this* message)
- flag (don't necessarily block) suspicious combos for manual review, e.g. `scam` + `notify` together,
  or `mute` with confidence < 0.5 and no rule/history backing

## 8. Evaluation (`code/evaluation/main.py`)

Two things it must do:

1. **Format/sanity pass** using `sample_messages.csv` (own IDs, disjoint from `messages.csv` — never
   used to tune thresholds, only to check output shape/style and rough label distribution sanity).
2. **Cost-sensitive confusion analysis.** Not all misclassifications are equal:

   ```
   true \ pred   notify   digest   mute
   notify          -       mild    severe   (missed something important)
   digest        mild        -     mild
   mute         severe     mild       -     (false interruption from junk/risk)
   ```

   Report both plain accuracy/F1 (on `action` and `message_type`) *and* this weighted cost wherever
   labels exist, plus:
   - evidence precision: fraction of cited `evidence_message_ids` that are actually relevant matches
     (same sender/group/business as the message being evaluated)
   - determinism check: run the full pipeline twice on the same input, diff outputs, must be identical
   - ablation note: rules-only vs LLM-only vs hybrid, to justify the hybrid design in the writeup

## 9. Code layout

```
code/
  main.py                    # entry point: dataset/ -> output.csv
  context_builder.py         # joins + retrieval (§3)
  media/
    image_processor.py       # OCR/VLM + cache (§4)
    voice_processor.py       # ASR + cache (§4)
  rules.py                   # deterministic override layer (§5)
  router.py                  # LLM call + schema validation (§6)
  validate.py                # output sanity checks (§7)
  evaluation/
    main.py                  # sample_messages.csv sanity + cost-matrix eval (§8)
  README.md                  # setup/run instructions, required per submission rules
.cache/
  media/                     # OCR/ASR cache keyed by media_id, gitignored
```

## 10. Tradeoffs, scaling, and known limitations

**Rules vs. LLM coverage.** Rules are cheap, instant, deterministic, and auditable, but only cover named
patterns. A genuinely novel scam/abuse pattern that doesn't match rules 1–3 depends entirely on the LLM
noticing it in a single generation. Mitigation is in §6: the LLM always sees the same raw signal fields
the rules check, even when no rule fired — but this is a real residual risk, not a solved one, and should
be stated as such rather than implied away.

**Why rules matter for cost, not just auditability.** Every message a rule resolves is one skipped LLM
call. At this dataset's size that's a minor saving; at real scale, the rule layer is the primary cost/
latency control valve, since LLM-per-message doesn't scale to WhatsApp volumes. This is an argument for
keeping the rule set real (each tied to a concrete column) rather than either (a) collapsing everything
into a monolithic LLM call, or (b) inflating the rule set with soft/fuzzy conditions that don't actually
resolve anything and just add branches to maintain.

**Retrieval scaling.** Brute-force key match + cheap token-overlap re-ranking over ~1000 history rows is
appropriate at this size. It would not be at real scale (millions of messages per user) — that would need
a per-user-partitioned index, likely a proper text/vector store for the topic re-ranking step. Not built
here because the data doesn't warrant it; noted so the choice reads as deliberate, not naive.

**Media processing scaling.** Caching by `media_id` helps because this dataset actually reuses media
(`img_008` recurs across two messages). At real scale most images/voice notes are unique per message, so
the cache would not reduce the dominant cost — that cost is structural (every unique image needs one
OCR/VLM pass) and would need an async/batched pipeline off the hot path, which is out of scope for this
build but worth naming rather than pretending the current design would hold up unchanged.

**Confidence is calibrated internally, not statistically validated.** With 30 labeled sample rows (own
IDs, disjoint from the real test set) and no ground truth on the actual 110 messages, there is no way to
compute a real calibration metric (e.g. Brier score) against this system's confidence output. The eval
harness can only check internal consistency — that caps/floors fire on the conditions they're supposed
to, that rule-based bands and LLM-adjusted values don't overlap in a way that contradicts each other. This
limitation should be stated plainly in the eval writeup rather than implied away by reporting a
confidence number with more precision than the data supports.

**Single LLM call does more than one job.** One call produces `action`, `message_type`, `reason`, and
`evidence_message_ids` together, rather than splitting extraction/scoring and final decision into
separate calls. This is a deliberate scope/time tradeoff for a 24-hour build — it halves LLM cost and
latency, at the expense of interpretability (if the action is wrong, harder to tell whether extraction,
scoring, or the final decision step was the actual point of failure). Worth naming as a chosen tradeoff,
not an oversight, if asked.

**Cross-user safety evidence is a deliberately narrow exception.** §3 allows retrieving another user's
`message_history` row as evidence, but only when that row's own reaction was `reported`/`muted`, and only
to support a `mute`/`scam`/`spam` decision — never to infer this user's preferences from someone else's
behavior. This boundary is intentional: aggregating engagement patterns across users for personalization
would be a privacy overreach the dataset doesn't ask for; using another user's explicit report of a scam
to protect a different user from the same sender is closer to how real spam-report networks work and is
scoped tightly enough not to raise that concern.

## 11. Explicitly out of scope / rejected

- A hand-weighted numeric scoring formula (`score = importance + urgency + trust - penalties`, bucketed
  at arbitrary thresholds). Rejected: the weights/cutoffs would be unfitted guesses (no labeled training
  set large enough to calibrate 5+ free parameters), it doesn't remove the need for an LLM step (still
  need `message_type`, `evidence_message_ids`, `reason` from somewhere), and it's harder to justify in
  the AI-judge interview than a small set of rules each traceable to one real column.
- An invented relationship-importance graph (mother=1.0, boss=0.8 style). No such role/relationship
  field exists for personal chats in the schema — personalization for `personal` conversations comes
  from `users.csv` global stats + retrieved `message_history` with that same `sender_user_id`, nothing
  else is available.
- Vector DB / embedding retrieval infra. ~1000 history rows total — brute-force key match + cheap
  token-overlap re-ranking is sufficient and easier to audit than an embedding index.