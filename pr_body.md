<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Keepalive workflows should use the dedicated GitHub App pool and fall back to PATs when app rate limits are low.

#### Tasks
- [ ] Switch from `WORKFLOWS_APP` to `KEEPALIVE_APP` (dedicated rate limit pool)
- [x] Add PAT fallback when app rate limit is low

#### Acceptance criteria
- [ ] Switch from `WORKFLOWS_APP` to `GH_APP`
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- needs-human: update `.github/workflows/agents-keepalive-loop.yml` to use `KEEPALIVE_APP_ID`/`KEEPALIVE_APP_PRIVATE_KEY` (and align acceptance target with `GH_APP_ID`/`GH_APP_PRIVATE_KEY`) in the preflight/env blocks.
- reviewed: keepalive scripts now normalize legacy WORKFLOWS_APP env into KEEPALIVE_APP and report auth sources without direct WORKFLOWS_APP lookups.
- reviewed: keepalive post-work summary now adds remediation guidance when legacy WORKFLOWS_APP env values are detected.
- reconciled: no additional tasks completed yet; workflow env update still blocks the remaining switch.

<!-- auto-status-summary:end -->
