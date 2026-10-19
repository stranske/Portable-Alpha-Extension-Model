# Dashboard provider settings audit (#2370)

Inspected source: `e1070935532bf1afc2f32ac9853a36fdd8b70c99` (2026-10-19).
This is an offline construction/payload audit. No live model request, provider
account check, endpoint-compatibility claim, default-model change or production
repair is included. The executable matrix uses synthetic keys and denies DNS,
socket connections and synchronous/asynchronous HTTP sends.

## Complete bounded inventory

The inventory guard enumerates all direct dashboard calls to
`resolve_llm_provider_config`; there are three current entry points. Factory
construction for OpenAI, Anthropic and Azure OpenAI is in
`pa_core/llm/provider.py:create_llm`. Explain reaches it through
`pa_core/llm/result_explain.py`; Compare through `pa_core/llm/compare_runs.py`.

| Entry / consumer | Selector and key ownership | Endpoint / optional settings | Evidence and disposition |
| --- | --- | --- | --- |
| `dashboard/components/explain_results.py:render_explain_results_panel` | provider selector; provider-scoped typed/saved key; legacy unscoped key removed; shared resolver defaults | model override; resolver environment endpoint/version/org | Existing `test_provider_defaults_switches_and_secret_free_exports` executes the panel and real downstream config boundary with recording factory; includes OpenAI, Anthropic and explicit-key Azure switches. Source #2366 owns credential behavior. Offline construction supported below; no remote compatibility claim. |
| `dashboard/components/comparison_llm.py:render_comparison_llm_panel` | selector scoped by run; typed/saved key scoped by run AND provider; legacy run-only key removed | model scoped by run; Azure endpoint/version widgets scoped by run and passed only for Azure | Existing `test_comparison_provider_defaults_switches_and_secret_free_exports` plus `test_render_comparison_llm_panel_passes_azure_provider_config` and `test_comparison_panel_ignores_stale_azure_fields_for_openai`. Source #2366 owns credential behavior. Non-Azure stale Azure widget values are ignored. |
| `dashboard/pages/3_Scenario_Wizard.py:_build_config_chat_llm_invoke` | provider selector; `_CONFIG_CHAT_API_KEY_KEY` is one unscoped key | common model override; shared resolver environment options | Exact helper probe confirms an existing typed synthetic OpenAI key is reused after switching to Anthropic. See witness JSON and reproduction below. This is an unresolved instance already in #2365's Config Chat scope; no duplicate issue or bundled fix. Full rendered-page transitions and remote invocation remain UNKNOWN. |
| `dashboard/components/llm_settings.py:resolve_llm_provider_config` | explicit key/name → explicit `PA_STREAMLIT_API_KEY` override → OpenAI or Anthropic provider-specific key | provider-specific packaging below | Executed resolver and real SDK constructor matrix, tracing-only refusal and repeated OpenAI/Anthropic/OpenAI switch test. |

The global `PA_STREAMLIT_API_KEY` is an explicit shared override by the current
resolver contract; it is distinct from provider-specific automatic defaults.
Azure intentionally has no automatic `AZURE_OPENAI_API_KEY` fallback in this
dashboard. It needs an explicit key/shared override plus endpoint and API
version. Lack of automatic Azure fallback is intentional, not a missing-key
compatibility PASS. Optional model strings and global endpoint overrides can
still be inappropriate for a remote provider; local construction cannot decide
that. The common Config Chat model state is observed statically, not proven
compatible across remote providers.

## Supported provider settings matrix

`test_audit_provider_options_reach_real_offline_constructor` resolves the real
config and constructs the installed real SDK client for every row, while all
network paths are denied. This proves local accepted constructor shape and
key/option routing, not a successful remote request or billing authorization.

| Provider | Credentials | Endpoint and API version | Organization | Evidence classification |
| --- | --- | --- | --- | --- |
| OpenAI | explicit synthetic key → `ChatOpenAI.openai_api_key` | `base_url` → `openai_api_base` | retained as `openai_organization` | Executed local real SDK constructor, zero network attempts |
| Anthropic | explicit synthetic key → `ChatAnthropic.anthropic_api_key` | `base_url` → `anthropic_api_url`; Azure API version not forwarded | omitted; not in emitted client kwargs/model kwargs | Executed local real SDK constructor, zero network attempts |
| Azure OpenAI | explicit synthetic key → `AzureChatOpenAI.openai_api_key` | endpoint and API version in required credentials; no duplicate generic `base_url` | retained as `openai_organization` | Executed local real SDK constructor, zero network attempts |

Tracing-only credentials refuse all three providers before construction and do
not appear in errors. Resolver switching has no sticky provider key. Existing
missing-endpoint/version tests retain safe Azure refusal evidence. Existing
panel switch/export tests supply UI-boundary evidence for Explain/Compare;
the constructor matrix alone is not represented as full UI coverage.

## Known ownership and remaining work

- F30 / [#2366](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2366):
  selected-provider credentials, provider-scoped Explain/Compare state and
  tracing separation. Current working controls are retained, not re-fixed.
- F32 / [#2368](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2368):
  unsupported Anthropic organization payload and supported provider options.
  Current resolver strips Anthropic organization; real SDK matrix verifies it.
- Prior credential audit [#2365](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2365):
  owns the already-declared Config Chat unscoped-key path, log/fleet/export
  inventory and source-to-sink residuals. This audit supplies the exact helper
  witness; it does not declare that broader audit complete or create a new owner.
- This #2370 delivery adds the finite option/provider matrix and inventory
  freshness guard. No additional unowned instance was established. Additional
  discoveries require a scoped proposal/human decision before issue filing or
  production repairs, per the source acceptance boundary.

## Reproduce the limited Config Chat witness

`provider-settings-audit-witness.json` binds the file bytes, exact source head,
function, selected providers and observed reuse. It executes the exact extracted
helper, real settings resolver and a recording factory, with synthetic session
state. It does not execute the top-level page, invoke the returned closure or
assert that an abnormal page exit is a passing compatibility test.

```python
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch
from dashboard.components import llm_settings as settings

path = Path("dashboard/pages/3_Scenario_Wizard.py")
tree = ast.parse(path.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "_build_config_chat_llm_invoke")
keys = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {"_CONFIG_CHAT_PROVIDER_KEY",
            "_CONFIG_CHAT_MODEL_KEY", "_CONFIG_CHAT_API_KEY_KEY"}}
state = {keys["_CONFIG_CHAT_PROVIDER_KEY"]: "openai",
         keys["_CONFIG_CHAT_API_KEY_KEY"]: "audit-synthetic-override"}
seen = []
def factory(config):
    seen.append(config)
    return SimpleNamespace()
ns = dict(keys, Callable=Callable, st=SimpleNamespace(session_state=state),
          default_provider=settings.default_provider,
          resolve_api_key_input=settings.resolve_api_key_input,
          default_api_key=settings.default_api_key,
          resolve_llm_provider_config=settings.resolve_llm_provider_config,
          create_llm=factory)
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
with patch.object(settings, "read_secret", lambda name: None):
    assert callable(ns[fn.name]()[0])
    state[keys["_CONFIG_CHAT_PROVIDER_KEY"]] = "anthropic"
    assert callable(ns[fn.name]()[0])
assert [c.provider_name for c in seen] == ["openai", "anthropic"]
assert seen[0].credentials["api_key"] == seen[1].credentials["api_key"]
```

## Validation and its limits

Required command: `python -m pytest tests/test_dashboard_llm_settings.py -q`
(with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and owner-default Anaconda Python in
this worktree). The focused entry-point validation additionally runs
`tests/test_dashboard_explain_results.py`, `tests/test_dashboard_comparison_llm.py`
and `tests/test_llm_provider_missing_keys.py`. Final counts/results are recorded
in the PR body and run receipt.

Actual mutation controls change only `dashboard/components/llm_settings.py`
temporarily: allowing Anthropic organization makes the new option-contract
assertion fail; restoring the exact bytes passes. Reintroducing tracing-key
fallback makes all three tracing-only refusal cases fail; exact restoration
passes. Initial test drafting used the wrong Anthropic model attribute name;
that harness error was corrected, and is not claimed as a useful RED proof.

No workflow, model default, credential, provider implementation or production
module is changed. Remote API behavior, account access, UI browser operation,
full Config Chat page execution and endpoint semantics remain unverified.


Final local results: the required unfiltered settings suite collected and passed
81 tests; the settings/panel/provider-key combined suite collected and passed
129 tests in 16.14 seconds, without marker exclusions. Organization mutation:
1 expected contract assertion failure → exact restore 1 PASS. Tracing fallback
mutation: 3 expected refusal failures → exact restore 3 PASS. Black100, Ruff and
diff checks pass. Installed constructor versions are retained separately in
`provider-settings-sdk-versions.json`. These are local observed results, not
current-head hosted CI, review closure, merge or live endpoint acceptance.


## Optional-extra and pinned-runtime review recovery

The real-constructor cases now use `pytest.importorskip` for their provider SDK
and HTTPX. Base/development installations without `.[llm]` skip exactly those
three matrix rows; resolver, inventory, credential and payload tests still run.
A controlled missing-provider import run of the complete settings suite passed
78 and skipped 3. A skip is explicitly not a real-constructor compatibility PASS.

Initial SDK evidence above used older locally installed provider versions and
is superseded for supported-runtime constructor claims. The final run used a
clean Anaconda-derived virtual environment without system-site-packages and
installed the complete repository `requirements.lock`. The retained version
receipt records `langchain-openai==1.4.1`, `langchain-anthropic==1.5.3`,
`langchain-core==1.5.3`, `openai==2.48.0`, `anthropic==0.120.0`, and the lock
file SHA-256. `pip check` reports no broken requirements. The final pinned
settings/panel/provider-key suite passed all 129, with zero skips, in 21.64s.
The required unfiltered settings command is also rerun separately on that
interpreter. This establishes the constructor matrix for the repository pins.

Actual CI job `118258616727` stopped before pytest: the dependency verifier
reported a newly undeclared direct `httpx` test import. The test now obtains it
through the explicit optional-extra guard. The exact failing command,
`python scripts/sync_test_dependencies.py --verify`, passes without changing
project dependencies, lock files, synced scripts or workflow gates. The clean
venv is test-only; no shared interpreter or launcher setting was changed.
