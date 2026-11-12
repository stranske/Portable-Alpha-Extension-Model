# Production mutation replay, 2026-11-12 UTC

Named test: `tests/test_result_explain.py::test_analysis_output_contains_required_sections_and_is_json_serializable`

Command for every red and green run: `timeout -k 5s 120s /opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pytest tests/test_result_explain.py::test_analysis_output_contains_required_sections_and_is_json_serializable -m 'not slow and not live_llm' -q`


## head-instead-of-tail

```diff
--- a/pa_core/llm/result_explain.py
+++ b/pa_core/llm/result_explain.py
@@ -169,5 +169,5 @@
     if details_df.empty:
         return []
-    tail_df = details_df.tail(_TAIL_ROW_LIMIT).copy()
+    tail_df = details_df.head(_TAIL_ROW_LIMIT).copy()
     return cast(list[dict[str, Any]], _to_json_safe(tail_df.to_dict(orient="records")))

```

Red exit 1, 4.291s wall:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
___ test_analysis_output_contains_required_sections_and_is_json_serializable ___

    def test_analysis_output_contains_required_sections_and_is_json_serializable() -> None:
        details = pd.DataFrame(
            {"Agent": ["A", "B", "C", "D", "Total"], "monthly_TE": [0, 10, 20, 30, 40]},
            index=[7, 2, 9, 1, 5],
        )
        original = details.copy(deep=True)
        _text, _trace_url, payload = result_explain.explain_results_details(details)

        analysis_output = payload["analysis_output"]
        assert analysis_output["rows"] == 5
        assert analysis_output["columns"] == ["Agent", "monthly_TE"]
        assert analysis_output["basic_statistics"] == {
            "monthly_TE": pytest.approx({"count": 5, "mean": 20, "std": 250**0.5, "min": 0, "max": 40})
        }
>       assert analysis_output["tail_sample_rows"] == [
            {"Agent": "C", "monthly_TE": 20},
            {"Agent": "D", "monthly_TE": 30},
            {"Agent": "Total", "monthly_TE": 40},
        ]
E       AssertionError: assert [{'Agent': 'A...thly_TE': 20}] == [{'Agent': 'C...thly_TE': 40}]
E
E         At index 0 diff: {'Agent': 'A', 'monthly_TE': 0} != {'Agent': 'C', 'monthly_TE': 20}
E
E         Full diff: (-: missing in left side, +: extra in left side)
E           [
E         +     {
E         +         'Agent': 'A',
E         +         'monthly_TE': 0,
E         +     },
E         +     {
E         +         'Agent': 'B',
E         +         'monthly_TE': 10,
E         +     },
E               {
E                   'Agent': 'C',
E                   'monthly_TE': 20,
E               },
E         -     {
E         -         'Agent': 'D',
E         -         'monthly_TE': 30,
E         -     },
E         -     {
E         -         'Agent': 'Total',
E         -         'monthly_TE': 40,
E         -     },
E           ]

tests/test_result_explain.py:346: AssertionError
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_result_explain.py::test_analysis_output_contains_required_sections_and_is_json_serializable - AssertionError: assert [{'Agent': 'A...thly_TE': 20}] == [{'Agent': 'C...thly_TE': 40}]

  At index 0 diff: {'Agent': 'A', 'monthly_TE': 0} != {'Agent': 'C', 'monthly_TE': 20}

  Full diff: (-: missing in left side, +: extra in left side)
    [
  +     {
  +         'Agent': 'A',
  +         'monthly_TE': 0,
  +     },
  +     {
  +         'Agent': 'B',
  +         'monthly_TE': 10,
  +     },
        {
            'Agent': 'C',
            'monthly_TE': 20,
        },
  -     {
  -         'Agent': 'D',
  -         'monthly_TE': 30,
  -     },
  -     {
  -         'Agent': 'Total',
  -         'monthly_TE': 40,
  -     },
    ]
1 failed, 13 warnings in 2.12s
```

Restored exit 0, 4.036s wall:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.91s
```


## two-tail-rows

```diff
--- a/pa_core/llm/result_explain.py
+++ b/pa_core/llm/result_explain.py
@@ -42,5 +42,5 @@


-_TAIL_ROW_LIMIT = 3
+_TAIL_ROW_LIMIT = 2
 _QUANTILE_LEVELS = [0.05, 0.5, 0.95]
 _METRIC_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
```

Red exit 1, 4.195s wall:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
___ test_analysis_output_contains_required_sections_and_is_json_serializable ___

    def test_analysis_output_contains_required_sections_and_is_json_serializable() -> None:
        details = pd.DataFrame(
            {"Agent": ["A", "B", "C", "D", "Total"], "monthly_TE": [0, 10, 20, 30, 40]},
            index=[7, 2, 9, 1, 5],
        )
        original = details.copy(deep=True)
        _text, _trace_url, payload = result_explain.explain_results_details(details)

        analysis_output = payload["analysis_output"]
        assert analysis_output["rows"] == 5
        assert analysis_output["columns"] == ["Agent", "monthly_TE"]
        assert analysis_output["basic_statistics"] == {
            "monthly_TE": pytest.approx({"count": 5, "mean": 20, "std": 250**0.5, "min": 0, "max": 40})
        }
>       assert analysis_output["tail_sample_rows"] == [
            {"Agent": "C", "monthly_TE": 20},
            {"Agent": "D", "monthly_TE": 30},
            {"Agent": "Total", "monthly_TE": 40},
        ]
E       AssertionError: assert [{'Agent': 'D...thly_TE': 40}] == [{'Agent': 'C...thly_TE': 40}]
E
E         At index 0 diff: {'Agent': 'D', 'monthly_TE': 30} != {'Agent': 'C', 'monthly_TE': 20}
E         Right contains one more item: {'Agent': 'Total', 'monthly_TE': 40}
E
E         Full diff: (-: missing in left side, +: extra in left side)
E           [
E         -     {
E         -         'Agent': 'C',
E         -         'monthly_TE': 20,
E         -     },
E               {
E                   'Agent': 'D',
E                   'monthly_TE': 30,
E               },
E               {
E                   'Agent': 'Total',
E                   'monthly_TE': 40,
E               },
E           ]

tests/test_result_explain.py:346: AssertionError
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_result_explain.py::test_analysis_output_contains_required_sections_and_is_json_serializable - AssertionError: assert [{'Agent': 'D...thly_TE': 40}] == [{'Agent': 'C...thly_TE': 40}]

  At index 0 diff: {'Agent': 'D', 'monthly_TE': 30} != {'Agent': 'C', 'monthly_TE': 20}
  Right contains one more item: {'Agent': 'Total', 'monthly_TE': 40}

  Full diff: (-: missing in left side, +: extra in left side)
    [
  -     {
  -         'Agent': 'C',
  -         'monthly_TE': 20,
  -     },
        {
            'Agent': 'D',
            'monthly_TE': 30,
        },
        {
            'Agent': 'Total',
            'monthly_TE': 40,
        },
    ]
1 failed, 13 warnings in 2.07s
```

Restored exit 0, 4.024s wall:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.91s
```


## missing-standard-deviation

```diff
--- a/pa_core/llm/result_explain.py
+++ b/pa_core/llm/result_explain.py
@@ -154,5 +154,5 @@
     if numeric_df.empty:
         return {}
-    stat_rows = numeric_df.agg(["count", "mean", "std", "min", "max"]).to_dict()
+    stat_rows = numeric_df.agg(["count", "mean", "min", "max"]).to_dict()
     return cast(dict[str, dict[str, float]], _to_json_safe(stat_rows))

```

Red exit 1, 4.200s wall:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
___ test_analysis_output_contains_required_sections_and_is_json_serializable ___

    def test_analysis_output_contains_required_sections_and_is_json_serializable() -> None:
        details = pd.DataFrame(
            {"Agent": ["A", "B", "C", "D", "Total"], "monthly_TE": [0, 10, 20, 30, 40]},
            index=[7, 2, 9, 1, 5],
        )
        original = details.copy(deep=True)
        _text, _trace_url, payload = result_explain.explain_results_details(details)

        analysis_output = payload["analysis_output"]
        assert analysis_output["rows"] == 5
        assert analysis_output["columns"] == ["Agent", "monthly_TE"]
>       assert analysis_output["basic_statistics"] == {
            "monthly_TE": pytest.approx({"count": 5, "mean": 20, "std": 250**0.5, "min": 0, "max": 40})
        }
E       AssertionError: assert {'monthly_TE'... 'max': 40.0}} == {'monthly_TE'...0 ± 4.0e-05})}
E
E         Differing items:
E         {'monthly_TE': {'count': 5.0, 'mean': 20.0, 'min': 0.0, 'max': 40.0}} != {'monthly_TE': approx({'count': 5 ± 5.0e-06, 'mean': 20 ± 2.0e-05, 'std': 15.811388300841896 ± 1.6e-05, 'min': 0 ± 1.0e-12, 'max': 40 ± 4.0e-05})}
E
E         Full diff: (-: missing in left side, +: extra in left side)
E           {
E         -     'monthly_TE': approx({'count': 5 ± 5.0e-06, 'mean': 20 ± 2.0e-05, 'std': 15.811388300841896 ± 1.6e-05, 'min': 0 ± 1.0e-12, 'max': 40 ± 4.0e-05}),
E         +     'monthly_TE': {
E         +         'count': 5.0,
E         +         'mean': 20.0,
E         +         'min': 0.0,
E         +         'max': 40.0,
E         +     },
E           }

tests/test_result_explain.py:343: AssertionError
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_result_explain.py::test_analysis_output_contains_required_sections_and_is_json_serializable - AssertionError: assert {'monthly_TE'... 'max': 40.0}} == {'monthly_TE'...0 ± 4.0e-05})}

  Differing items:
  {'monthly_TE': {'count': 5.0, 'mean': 20.0, 'min': 0.0, 'max': 40.0}} != {'monthly_TE': approx({'count': 5 ± 5.0e-06, 'mean': 20 ± 2.0e-05, 'std': 15.811388300841896 ± 1.6e-05, 'min': 0 ± 1.0e-12, 'max': 40 ± 4.0e-05})}

  Full diff: (-: missing in left side, +: extra in left side)
    {
  -     'monthly_TE': approx({'count': 5 ± 5.0e-06, 'mean': 20 ± 2.0e-05, 'std': 15.811388300841896 ± 1.6e-05, 'min': 0 ± 1.0e-12, 'max': 40 ± 4.0e-05}),
  +     'monthly_TE': {
  +         'count': 5.0,
  +         'mean': 20.0,
  +         'min': 0.0,
  +         'max': 40.0,
  +     },
    }
1 failed, 13 warnings in 2.07s
```

Restored exit 0, 4.041s wall:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.91s
```


## quartiles-instead-of-tails

```diff
--- a/pa_core/llm/result_explain.py
+++ b/pa_core/llm/result_explain.py
@@ -43,5 +43,5 @@

 _TAIL_ROW_LIMIT = 3
-_QUANTILE_LEVELS = [0.05, 0.5, 0.95]
+_QUANTILE_LEVELS = [0.25, 0.5, 0.75]
 _METRIC_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
     "tracking_error": (
```

Red exit 1, 4.219s wall:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
___ test_analysis_output_contains_required_sections_and_is_json_serializable ___

    def test_analysis_output_contains_required_sections_and_is_json_serializable() -> None:
        details = pd.DataFrame(
            {"Agent": ["A", "B", "C", "D", "Total"], "monthly_TE": [0, 10, 20, 30, 40]},
            index=[7, 2, 9, 1, 5],
        )
        original = details.copy(deep=True)
        _text, _trace_url, payload = result_explain.explain_results_details(details)

        analysis_output = payload["analysis_output"]
        assert analysis_output["rows"] == 5
        assert analysis_output["columns"] == ["Agent", "monthly_TE"]
        assert analysis_output["basic_statistics"] == {
            "monthly_TE": pytest.approx({"count": 5, "mean": 20, "std": 250**0.5, "min": 0, "max": 40})
        }
        assert analysis_output["tail_sample_rows"] == [
            {"Agent": "C", "monthly_TE": 20},
            {"Agent": "D", "monthly_TE": 30},
            {"Agent": "Total", "monthly_TE": 40},
        ]
>       assert analysis_output["key_quantiles"] == {
            "monthly_TE": pytest.approx({"0.05": 2, "0.5": 20, "0.95": 38})
        }
E       AssertionError: assert {'monthly_TE'...'0.75': 30.0}} == {'monthly_TE'...8 ± 3.8e-05})}
E
E         Differing items:
E         {'monthly_TE': {'0.25': 10.0, '0.5': 20.0, '0.75': 30.0}} != {'monthly_TE': approx({'0.05': 2 ± 2.0e-06, '0.5': 20 ± 2.0e-05, '0.95': 38 ± 3.8e-05})}
E
E         Full diff: (-: missing in left side, +: extra in left side)
E           {
E         -     'monthly_TE': approx({'0.05': 2 ± 2.0e-06, '0.5': 20 ± 2.0e-05, '0.95': 38 ± 3.8e-05}),
E         +     'monthly_TE': {
E         +         '0.25': 10.0,
E         +         '0.5': 20.0,
E         +         '0.75': 30.0,
E         +     },
E           }

tests/test_result_explain.py:351: AssertionError
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_result_explain.py::test_analysis_output_contains_required_sections_and_is_json_serializable - AssertionError: assert {'monthly_TE'...'0.75': 30.0}} == {'monthly_TE'...8 ± 3.8e-05})}

  Differing items:
  {'monthly_TE': {'0.25': 10.0, '0.5': 20.0, '0.75': 30.0}} != {'monthly_TE': approx({'0.05': 2 ± 2.0e-06, '0.5': 20 ± 2.0e-05, '0.95': 38 ± 3.8e-05})}

  Full diff: (-: missing in left side, +: extra in left side)
    {
  -     'monthly_TE': approx({'0.05': 2 ± 2.0e-06, '0.5': 20 ± 2.0e-05, '0.95': 38 ± 3.8e-05}),
  +     'monthly_TE': {
  +         '0.25': 10.0,
  +         '0.5': 20.0,
  +         '0.75': 30.0,
  +     },
    }
1 failed, 13 warnings in 2.09s
```

Restored exit 0, 4.043s wall:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.91s
```


## zero-quantile-values

```diff
--- a/pa_core/llm/result_explain.py
+++ b/pa_core/llm/result_explain.py
@@ -162,5 +162,5 @@
     if numeric_df.empty:
         return {}
-    quantiles = numeric_df.quantile(_QUANTILE_LEVELS).to_dict()
+    quantiles = numeric_df.quantile(_QUANTILE_LEVELS).mul(0).to_dict()
     return cast(dict[str, dict[str, float]], _to_json_safe(quantiles))

```

Red exit 1, 4.229s wall:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
___ test_analysis_output_contains_required_sections_and_is_json_serializable ___

    def test_analysis_output_contains_required_sections_and_is_json_serializable() -> None:
        details = pd.DataFrame(
            {"Agent": ["A", "B", "C", "D", "Total"], "monthly_TE": [0, 10, 20, 30, 40]},
            index=[7, 2, 9, 1, 5],
        )
        original = details.copy(deep=True)
        _text, _trace_url, payload = result_explain.explain_results_details(details)

        analysis_output = payload["analysis_output"]
        assert analysis_output["rows"] == 5
        assert analysis_output["columns"] == ["Agent", "monthly_TE"]
        assert analysis_output["basic_statistics"] == {
            "monthly_TE": pytest.approx({"count": 5, "mean": 20, "std": 250**0.5, "min": 0, "max": 40})
        }
        assert analysis_output["tail_sample_rows"] == [
            {"Agent": "C", "monthly_TE": 20},
            {"Agent": "D", "monthly_TE": 30},
            {"Agent": "Total", "monthly_TE": 40},
        ]
>       assert analysis_output["key_quantiles"] == {
            "monthly_TE": pytest.approx({"0.05": 2, "0.5": 20, "0.95": 38})
        }
E       AssertionError: assert {'monthly_TE'... '0.95': 0.0}} == {'monthly_TE'...8 ± 3.8e-05})}
E
E         Differing items:
E         {'monthly_TE': {'0.05': 0.0, '0.5': 0.0, '0.95': 0.0}} != {'monthly_TE': approx({'0.05': 2 ± 2.0e-06, '0.5': 20 ± 2.0e-05, '0.95': 38 ± 3.8e-05})}
E
E         Full diff: (-: missing in left side, +: extra in left side)
E           {
E         -     'monthly_TE': approx({'0.05': 2 ± 2.0e-06, '0.5': 20 ± 2.0e-05, '0.95': 38 ± 3.8e-05}),
E         +     'monthly_TE': {
E         +         '0.05': 0.0,
E         +         '0.5': 0.0,
E         +         '0.95': 0.0,
E         +     },
E           }

tests/test_result_explain.py:351: AssertionError
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_result_explain.py::test_analysis_output_contains_required_sections_and_is_json_serializable - AssertionError: assert {'monthly_TE'... '0.95': 0.0}} == {'monthly_TE'...8 ± 3.8e-05})}

  Differing items:
  {'monthly_TE': {'0.05': 0.0, '0.5': 0.0, '0.95': 0.0}} != {'monthly_TE': approx({'0.05': 2 ± 2.0e-06, '0.5': 20 ± 2.0e-05, '0.95': 38 ± 3.8e-05})}

  Full diff: (-: missing in left side, +: extra in left side)
    {
  -     'monthly_TE': approx({'0.05': 2 ± 2.0e-06, '0.5': 20 ± 2.0e-05, '0.95': 38 ± 3.8e-05}),
  +     'monthly_TE': {
  +         '0.05': 0.0,
  +         '0.5': 0.0,
  +         '0.95': 0.0,
  +     },
    }
1 failed, 13 warnings in 2.08s
```

Restored exit 0, 4.057s wall:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.91s
```


## modify-caller-after-payload

```diff
--- a/pa_core/llm/result_explain.py
+++ b/pa_core/llm/result_explain.py
@@ -365,4 +365,5 @@
         except Exception:
             logger.warning("Failed to record result-explanation fleet event", exc_info=True)
+        details_df.iloc[0, 0] = "altered"
         return _with_disclaimer(text), None, payload

```

Red exit 1, 4.229s wall:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
___ test_analysis_output_contains_required_sections_and_is_json_serializable ___

    def test_analysis_output_contains_required_sections_and_is_json_serializable() -> None:
        details = pd.DataFrame(
            {"Agent": ["A", "B", "C", "D", "Total"], "monthly_TE": [0, 10, 20, 30, 40]},
            index=[7, 2, 9, 1, 5],
        )
        original = details.copy(deep=True)
        _text, _trace_url, payload = result_explain.explain_results_details(details)

        analysis_output = payload["analysis_output"]
        assert analysis_output["rows"] == 5
        assert analysis_output["columns"] == ["Agent", "monthly_TE"]
        assert analysis_output["basic_statistics"] == {
            "monthly_TE": pytest.approx({"count": 5, "mean": 20, "std": 250**0.5, "min": 0, "max": 40})
        }
        assert analysis_output["tail_sample_rows"] == [
            {"Agent": "C", "monthly_TE": 20},
            {"Agent": "D", "monthly_TE": 30},
            {"Agent": "Total", "monthly_TE": 40},
        ]
        assert analysis_output["key_quantiles"] == {
            "monthly_TE": pytest.approx({"0.05": 2, "0.5": 20, "0.95": 38})
        }
        assert json.loads(json.dumps(analysis_output, sort_keys=True)) == analysis_output
>       pd.testing.assert_frame_equal(details, original)

tests/test_result_explain.py:355:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/pandas/_testing/asserters.py:1019: in assert_extension_array_equal
    _testing.assert_almost_equal(
pandas/_libs/testing.pyx:64: in pandas._libs.testing.assert_almost_equal
    ???
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

>   ???
E   AssertionError: DataFrame.iloc[:, 0] (column name="Agent") are different
E
E   DataFrame.iloc[:, 0] (column name="Agent") values are different (20.0 %)
E   [index]: [7, 2, 9, 1, 5]
E   [left]:  [altered, B, C, D, Total]
E   [right]: [A, B, C, D, Total]
E   At positional index 0, first diff: 'altered' != 'A'

pandas/_libs/testing.pyx:227: AssertionError
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_result_explain.py::test_analysis_output_contains_required_sections_and_is_json_serializable - AssertionError: DataFrame.iloc[:, 0] (column name="Agent") are different

DataFrame.iloc[:, 0] (column name="Agent") values are different (20.0 %)
[index]: [7, 2, 9, 1, 5]
[left]:  [altered, B, C, D, Total]
[right]: [A, B, C, D, Total]
At positional index 0, first diff: 'altered' != 'A'
1 failed, 13 warnings in 2.10s
```

Restored exit 0, 4.028s wall:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.92s
```
