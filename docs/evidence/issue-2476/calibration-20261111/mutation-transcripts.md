# Calibration CLI production mutations, 2026-11-11 UTC

Source revision: `7e3e46965c3713d1acdb95cca59f704b6b5c3dbc`.
Each mutation changed only `pa_core/pa.py`; original bytes were restored in
`finally` before the passing rerun. Tests and expectations were unchanged
between red and green. The recorded hashes verify restoration.

## default-window

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[defaults]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[defaults]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.228
- Green exit: 0; wall seconds: 3.872
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `bf44ad656de64764eb9a98211ac13d5a300942c1f0b64bdd0477c9e6c7607f77`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -446 +446 @@
-            vol_regime_window = 12
+            vol_regime_window = 6
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_______ test_pa_calibrate_input_forwards_settings_and_exports[defaults] ________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7ee1f78243e0>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-2/test_pa_calibrate_input_forwar0')
capsys = <_pytest.capture.CaptureFixture object at 0x7ee1f789e660>
mapping_text = None, cli_options = []
expected_settings = ('none', 'single', 12)

    @pytest.mark.parametrize(
        ("mapping_text", "cli_options", "expected_settings"),
        [
            pytest.param(None, [], ("none", "single", 12), id="defaults"),
            pytest.param(
                "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n",
                [],
                ("ledoit_wolf", "two_state", 5),
                id="flat-template",
            ),
            pytest.param(
                "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: '9'\n",
                [],
                ("ledoit_wolf", "two_state", 9),
                id="nested-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: 9\n",
                ["--cov-shrinkage", "none", "--vol-regime", "single", "--vol-regime-window", "3"],
                ("none", "single", 3),
                id="cli-overrides-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n",
                [],
                ("ledoit_wolf", "single", 12),
                id="partial-template",
            ),
        ],
    )
    def test_pa_calibrate_input_forwards_settings_and_exports(
        monkeypatch, tmp_path, capsys, mapping_text, cli_options, expected_settings
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        importer = (
            importer_class.return_value
            if mapping_text is None
            else importer_class.from_template.return_value
        )
        importer.min_obs = 7 if mapping_text is None else 19
        importer.allow_below_min_obs = False
        frame = pd.DataFrame({"id": ["IDX"], "return": [0.01]})
        importer.load.return_value = frame
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        input_path = tmp_path / "returns.csv"
        output_path = tmp_path / "assets.yaml"
        args = [
            "calibrate",
            "--input",
            str(input_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--min-obs",
            "7",
        ]
        if mapping_text is not None:
            mapping_path = tmp_path / "mapping.yaml"
            mapping_path.write_text(mapping_text)
            args.extend(["--mapping", str(mapping_path)])
        pa_entry.main([*args, *cli_options])

        if mapping_text is None:
            importer_class.assert_called_once_with(min_obs=7, allow_below_min_obs=True)
            importer_class.from_template.assert_not_called()
        else:
            importer_class.assert_not_called()
            importer_class.from_template.assert_called_once_with(str(mapping_path))
            assert importer.allow_below_min_obs is True
        importer.load.assert_called_once_with(str(input_path))
        covariance_shrinkage, vol_regime, window = expected_settings
>       calibrator_class.assert_called_once_with(
            min_obs=importer.min_obs,
            covariance_shrinkage=covariance_shrinkage,
            vol_regime=vol_regime,
            vol_regime_window=window,
        )

tests/test_coverage_round_2381.py:621:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:998: in assert_called_once_with
    return self.assert_called_with(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <MagicMock spec='CalibrationAgent' id='139508984989744'>, args = ()
kwargs = {'min_obs': 7, 'covariance_shrinkage': 'none', 'vol_regime': 'single', 'vol_regime_window': 12}

    def assert_called_with(self, /, *args, **kwargs):
        """assert that the last call was made with the specified arguments.

        Raises an AssertionError if the args and keyword args passed in are
        different to the last call to the mock."""
        if self.call_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            actual = 'not called.'
            error_message = ('expected call not found.\nExpected: %s\n  Actual: %s'
                    % (expected, actual))
            raise AssertionError(error_message)

        def _error_message():
            msg = self._format_mock_failure_message(args, kwargs)
            return msg
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        actual = self._call_matcher(self.call_args)
        if actual != expected:
            cause = expected if isinstance(expected, Exception) else None
>           raise AssertionError(_error_message()) from cause
E           AssertionError: expected call not found.
E           Expected: mock(min_obs=7, covariance_shrinkage='none', vol_regime='single', vol_regime_window=12)
E             Actual: mock(min_obs=7, covariance_shrinkage='none', vol_regime='single', vol_regime_window=6)

/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:986: AssertionError
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to /tmp/pytest-of-runner/pytest-2/test_pa_calibrate_input_forwar0/assets.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[defaults] - AssertionError: expected call not found.
Expected: mock(min_obs=7, covariance_shrinkage='none', vol_regime='single', vol_regime_window=12)
  Actual: mock(min_obs=7, covariance_shrinkage='none', vol_regime='single', vol_regime_window=6)
1 failed, 13 warnings in 2.01s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.70s
```

## template-short-samples

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[flat-template]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[flat-template]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.046
- Green exit: 0; wall seconds: 3.876
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `d8dc267eeaf25dece07e2cdc2c87fa2ba93bec53d3ff273fd281f4412d6e7dc5`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -418 +418 @@
-            importer.allow_below_min_obs = True
+            importer.allow_below_min_obs = False
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_____ test_pa_calibrate_input_forwards_settings_and_exports[flat-template] _____

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7a4cb225bbb0>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-4/test_pa_calibrate_input_forwar0')
capsys = <_pytest.capture.CaptureFixture object at 0x7a4cb2282900>
mapping_text = "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n"
cli_options = [], expected_settings = ('ledoit_wolf', 'two_state', 5)

    @pytest.mark.parametrize(
        ("mapping_text", "cli_options", "expected_settings"),
        [
            pytest.param(None, [], ("none", "single", 12), id="defaults"),
            pytest.param(
                "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n",
                [],
                ("ledoit_wolf", "two_state", 5),
                id="flat-template",
            ),
            pytest.param(
                "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: '9'\n",
                [],
                ("ledoit_wolf", "two_state", 9),
                id="nested-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: 9\n",
                ["--cov-shrinkage", "none", "--vol-regime", "single", "--vol-regime-window", "3"],
                ("none", "single", 3),
                id="cli-overrides-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n",
                [],
                ("ledoit_wolf", "single", 12),
                id="partial-template",
            ),
        ],
    )
    def test_pa_calibrate_input_forwards_settings_and_exports(
        monkeypatch, tmp_path, capsys, mapping_text, cli_options, expected_settings
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        importer = (
            importer_class.return_value
            if mapping_text is None
            else importer_class.from_template.return_value
        )
        importer.min_obs = 7 if mapping_text is None else 19
        importer.allow_below_min_obs = False
        frame = pd.DataFrame({"id": ["IDX"], "return": [0.01]})
        importer.load.return_value = frame
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        input_path = tmp_path / "returns.csv"
        output_path = tmp_path / "assets.yaml"
        args = [
            "calibrate",
            "--input",
            str(input_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--min-obs",
            "7",
        ]
        if mapping_text is not None:
            mapping_path = tmp_path / "mapping.yaml"
            mapping_path.write_text(mapping_text)
            args.extend(["--mapping", str(mapping_path)])
        pa_entry.main([*args, *cli_options])

        if mapping_text is None:
            importer_class.assert_called_once_with(min_obs=7, allow_below_min_obs=True)
            importer_class.from_template.assert_not_called()
        else:
            importer_class.assert_not_called()
            importer_class.from_template.assert_called_once_with(str(mapping_path))
>           assert importer.allow_below_min_obs is True
E           AssertionError: assert False is True
E            +  where False = <MagicMock name='mock.from_template()' id='134469822284864'>.allow_below_min_obs

tests/test_coverage_round_2381.py:618: AssertionError
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to /tmp/pytest-of-runner/pytest-4/test_pa_calibrate_input_forwar0/assets.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[flat-template] - AssertionError: assert False is True
 +  where False = <MagicMock name='mock.from_template()' id='134469822284864'>.allow_below_min_obs
1 failed, 13 warnings in 1.87s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.72s
```

## nested-overrides

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[nested-template]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[nested-template]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 3.997
- Green exit: 0; wall seconds: 3.94
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `68d56ead0392da6f1b77beeacbcefe726d124030433d5baef868614d7d2e5c57`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -97 +97 @@
-        return calib
+        return data
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
____ test_pa_calibrate_input_forwards_settings_and_exports[nested-template] ____

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7d242dcefce0>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-6/test_pa_calibrate_input_forwar0')
capsys = <_pytest.capture.CaptureFixture object at 0x7d242db22900>
mapping_text = "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n  vol_regime: two_state\n  vol_regime_window: '9'\n"
cli_options = [], expected_settings = ('ledoit_wolf', 'two_state', 9)

    @pytest.mark.parametrize(
        ("mapping_text", "cli_options", "expected_settings"),
        [
            pytest.param(None, [], ("none", "single", 12), id="defaults"),
            pytest.param(
                "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n",
                [],
                ("ledoit_wolf", "two_state", 5),
                id="flat-template",
            ),
            pytest.param(
                "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: '9'\n",
                [],
                ("ledoit_wolf", "two_state", 9),
                id="nested-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: 9\n",
                ["--cov-shrinkage", "none", "--vol-regime", "single", "--vol-regime-window", "3"],
                ("none", "single", 3),
                id="cli-overrides-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n",
                [],
                ("ledoit_wolf", "single", 12),
                id="partial-template",
            ),
        ],
    )
    def test_pa_calibrate_input_forwards_settings_and_exports(
        monkeypatch, tmp_path, capsys, mapping_text, cli_options, expected_settings
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        importer = (
            importer_class.return_value
            if mapping_text is None
            else importer_class.from_template.return_value
        )
        importer.min_obs = 7 if mapping_text is None else 19
        importer.allow_below_min_obs = False
        frame = pd.DataFrame({"id": ["IDX"], "return": [0.01]})
        importer.load.return_value = frame
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        input_path = tmp_path / "returns.csv"
        output_path = tmp_path / "assets.yaml"
        args = [
            "calibrate",
            "--input",
            str(input_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--min-obs",
            "7",
        ]
        if mapping_text is not None:
            mapping_path = tmp_path / "mapping.yaml"
            mapping_path.write_text(mapping_text)
            args.extend(["--mapping", str(mapping_path)])
        pa_entry.main([*args, *cli_options])

        if mapping_text is None:
            importer_class.assert_called_once_with(min_obs=7, allow_below_min_obs=True)
            importer_class.from_template.assert_not_called()
        else:
            importer_class.assert_not_called()
            importer_class.from_template.assert_called_once_with(str(mapping_path))
            assert importer.allow_below_min_obs is True
        importer.load.assert_called_once_with(str(input_path))
        covariance_shrinkage, vol_regime, window = expected_settings
>       calibrator_class.assert_called_once_with(
            min_obs=importer.min_obs,
            covariance_shrinkage=covariance_shrinkage,
            vol_regime=vol_regime,
            vol_regime_window=window,
        )

tests/test_coverage_round_2381.py:621:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:998: in assert_called_once_with
    return self.assert_called_with(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <MagicMock spec='CalibrationAgent' id='137594338216656'>, args = ()
kwargs = {'min_obs': 19, 'covariance_shrinkage': 'ledoit_wolf', 'vol_regime': 'two_state', 'vol_regime_window': 9}

    def assert_called_with(self, /, *args, **kwargs):
        """assert that the last call was made with the specified arguments.

        Raises an AssertionError if the args and keyword args passed in are
        different to the last call to the mock."""
        if self.call_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            actual = 'not called.'
            error_message = ('expected call not found.\nExpected: %s\n  Actual: %s'
                    % (expected, actual))
            raise AssertionError(error_message)

        def _error_message():
            msg = self._format_mock_failure_message(args, kwargs)
            return msg
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        actual = self._call_matcher(self.call_args)
        if actual != expected:
            cause = expected if isinstance(expected, Exception) else None
>           raise AssertionError(_error_message()) from cause
E           AssertionError: expected call not found.
E           Expected: mock(min_obs=19, covariance_shrinkage='ledoit_wolf', vol_regime='two_state', vol_regime_window=9)
E             Actual: mock(min_obs=19, covariance_shrinkage='none', vol_regime='single', vol_regime_window=12)

/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:986: AssertionError
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to /tmp/pytest-of-runner/pytest-6/test_pa_calibrate_input_forwar0/assets.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[nested-template] - AssertionError: expected call not found.
Expected: mock(min_obs=19, covariance_shrinkage='ledoit_wolf', vol_regime='two_state', vol_regime_window=9)
  Actual: mock(min_obs=19, covariance_shrinkage='none', vol_regime='single', vol_regime_window=12)
1 failed, 13 warnings in 1.87s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.72s
```

## cli-window-precedence

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[cli-overrides-template]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[cli-overrides-template]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.036
- Green exit: 0; wall seconds: 3.826
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `cf46ae896cae1dc3df32bb5443958c95e8d8928cc36505ab3a786a32fa51d3ef`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -441 +441 @@
-            args.vol_regime_window
+            calibration_overrides.get("vol_regime_window")
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_pa_calibrate_input_forwards_settings_and_exports[cli-overrides-template] _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7ca7f232fce0>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-8/test_pa_calibrate_input_forwar0')
capsys = <_pytest.capture.CaptureFixture object at 0x7ca7f235e900>
mapping_text = 'calibration:\n  covariance_shrinkage: ledoit_wolf\n  vol_regime: two_state\n  vol_regime_window: 9\n'
cli_options = ['--cov-shrinkage', 'none', '--vol-regime', 'single', '--vol-regime-window', '3']
expected_settings = ('none', 'single', 3)

    @pytest.mark.parametrize(
        ("mapping_text", "cli_options", "expected_settings"),
        [
            pytest.param(None, [], ("none", "single", 12), id="defaults"),
            pytest.param(
                "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n",
                [],
                ("ledoit_wolf", "two_state", 5),
                id="flat-template",
            ),
            pytest.param(
                "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: '9'\n",
                [],
                ("ledoit_wolf", "two_state", 9),
                id="nested-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: 9\n",
                ["--cov-shrinkage", "none", "--vol-regime", "single", "--vol-regime-window", "3"],
                ("none", "single", 3),
                id="cli-overrides-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n",
                [],
                ("ledoit_wolf", "single", 12),
                id="partial-template",
            ),
        ],
    )
    def test_pa_calibrate_input_forwards_settings_and_exports(
        monkeypatch, tmp_path, capsys, mapping_text, cli_options, expected_settings
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        importer = (
            importer_class.return_value
            if mapping_text is None
            else importer_class.from_template.return_value
        )
        importer.min_obs = 7 if mapping_text is None else 19
        importer.allow_below_min_obs = False
        frame = pd.DataFrame({"id": ["IDX"], "return": [0.01]})
        importer.load.return_value = frame
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        input_path = tmp_path / "returns.csv"
        output_path = tmp_path / "assets.yaml"
        args = [
            "calibrate",
            "--input",
            str(input_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--min-obs",
            "7",
        ]
        if mapping_text is not None:
            mapping_path = tmp_path / "mapping.yaml"
            mapping_path.write_text(mapping_text)
            args.extend(["--mapping", str(mapping_path)])
        pa_entry.main([*args, *cli_options])

        if mapping_text is None:
            importer_class.assert_called_once_with(min_obs=7, allow_below_min_obs=True)
            importer_class.from_template.assert_not_called()
        else:
            importer_class.assert_not_called()
            importer_class.from_template.assert_called_once_with(str(mapping_path))
            assert importer.allow_below_min_obs is True
        importer.load.assert_called_once_with(str(input_path))
        covariance_shrinkage, vol_regime, window = expected_settings
>       calibrator_class.assert_called_once_with(
            min_obs=importer.min_obs,
            covariance_shrinkage=covariance_shrinkage,
            vol_regime=vol_regime,
            vol_regime_window=window,
        )

tests/test_coverage_round_2381.py:621:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:998: in assert_called_once_with
    return self.assert_called_with(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <MagicMock spec='CalibrationAgent' id='137060764273360'>, args = ()
kwargs = {'min_obs': 19, 'covariance_shrinkage': 'none', 'vol_regime': 'single', 'vol_regime_window': 3}

    def assert_called_with(self, /, *args, **kwargs):
        """assert that the last call was made with the specified arguments.

        Raises an AssertionError if the args and keyword args passed in are
        different to the last call to the mock."""
        if self.call_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            actual = 'not called.'
            error_message = ('expected call not found.\nExpected: %s\n  Actual: %s'
                    % (expected, actual))
            raise AssertionError(error_message)

        def _error_message():
            msg = self._format_mock_failure_message(args, kwargs)
            return msg
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        actual = self._call_matcher(self.call_args)
        if actual != expected:
            cause = expected if isinstance(expected, Exception) else None
>           raise AssertionError(_error_message()) from cause
E           AssertionError: expected call not found.
E           Expected: mock(min_obs=19, covariance_shrinkage='none', vol_regime='single', vol_regime_window=3)
E             Actual: mock(min_obs=19, covariance_shrinkage='none', vol_regime='single', vol_regime_window=9)

/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:986: AssertionError
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to /tmp/pytest-of-runner/pytest-8/test_pa_calibrate_input_forwar0/assets.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[cli-overrides-template] - AssertionError: expected call not found.
Expected: mock(min_obs=19, covariance_shrinkage='none', vol_regime='single', vol_regime_window=3)
  Actual: mock(min_obs=19, covariance_shrinkage='none', vol_regime='single', vol_regime_window=9)
1 failed, 13 warnings in 1.89s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.71s
```

## partial-default-regime

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[partial-template]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[partial-template]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.032
- Green exit: 0; wall seconds: 3.868
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `9923143abed30246a8d668ff5977c902fcc5fe8ec6326824fc68fa2925a75763`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -452 +452 @@
-            vol_regime if vol_regime is not None else "single"
+            vol_regime if vol_regime is not None else "two_state"
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
___ test_pa_calibrate_input_forwards_settings_and_exports[partial-template] ____

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7178a29fbbb0>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-10/test_pa_calibrate_input_forwar0')
capsys = <_pytest.capture.CaptureFixture object at 0x7178a2a26900>
mapping_text = 'calibration:\n  covariance_shrinkage: ledoit_wolf\n'
cli_options = [], expected_settings = ('ledoit_wolf', 'single', 12)

    @pytest.mark.parametrize(
        ("mapping_text", "cli_options", "expected_settings"),
        [
            pytest.param(None, [], ("none", "single", 12), id="defaults"),
            pytest.param(
                "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n",
                [],
                ("ledoit_wolf", "two_state", 5),
                id="flat-template",
            ),
            pytest.param(
                "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: '9'\n",
                [],
                ("ledoit_wolf", "two_state", 9),
                id="nested-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: 9\n",
                ["--cov-shrinkage", "none", "--vol-regime", "single", "--vol-regime-window", "3"],
                ("none", "single", 3),
                id="cli-overrides-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n",
                [],
                ("ledoit_wolf", "single", 12),
                id="partial-template",
            ),
        ],
    )
    def test_pa_calibrate_input_forwards_settings_and_exports(
        monkeypatch, tmp_path, capsys, mapping_text, cli_options, expected_settings
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        importer = (
            importer_class.return_value
            if mapping_text is None
            else importer_class.from_template.return_value
        )
        importer.min_obs = 7 if mapping_text is None else 19
        importer.allow_below_min_obs = False
        frame = pd.DataFrame({"id": ["IDX"], "return": [0.01]})
        importer.load.return_value = frame
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        input_path = tmp_path / "returns.csv"
        output_path = tmp_path / "assets.yaml"
        args = [
            "calibrate",
            "--input",
            str(input_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--min-obs",
            "7",
        ]
        if mapping_text is not None:
            mapping_path = tmp_path / "mapping.yaml"
            mapping_path.write_text(mapping_text)
            args.extend(["--mapping", str(mapping_path)])
        pa_entry.main([*args, *cli_options])

        if mapping_text is None:
            importer_class.assert_called_once_with(min_obs=7, allow_below_min_obs=True)
            importer_class.from_template.assert_not_called()
        else:
            importer_class.assert_not_called()
            importer_class.from_template.assert_called_once_with(str(mapping_path))
            assert importer.allow_below_min_obs is True
        importer.load.assert_called_once_with(str(input_path))
        covariance_shrinkage, vol_regime, window = expected_settings
>       calibrator_class.assert_called_once_with(
            min_obs=importer.min_obs,
            covariance_shrinkage=covariance_shrinkage,
            vol_regime=vol_regime,
            vol_regime_window=window,
        )

tests/test_coverage_round_2381.py:621:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:998: in assert_called_once_with
    return self.assert_called_with(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <MagicMock spec='CalibrationAgent' id='124762937772752'>, args = ()
kwargs = {'min_obs': 19, 'covariance_shrinkage': 'ledoit_wolf', 'vol_regime': 'single', 'vol_regime_window': 12}

    def assert_called_with(self, /, *args, **kwargs):
        """assert that the last call was made with the specified arguments.

        Raises an AssertionError if the args and keyword args passed in are
        different to the last call to the mock."""
        if self.call_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            actual = 'not called.'
            error_message = ('expected call not found.\nExpected: %s\n  Actual: %s'
                    % (expected, actual))
            raise AssertionError(error_message)

        def _error_message():
            msg = self._format_mock_failure_message(args, kwargs)
            return msg
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        actual = self._call_matcher(self.call_args)
        if actual != expected:
            cause = expected if isinstance(expected, Exception) else None
>           raise AssertionError(_error_message()) from cause
E           AssertionError: expected call not found.
E           Expected: mock(min_obs=19, covariance_shrinkage='ledoit_wolf', vol_regime='single', vol_regime_window=12)
E             Actual: mock(min_obs=19, covariance_shrinkage='ledoit_wolf', vol_regime='two_state', vol_regime_window=12)

/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:986: AssertionError
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to /tmp/pytest-of-runner/pytest-10/test_pa_calibrate_input_forwar0/assets.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[partial-template] - AssertionError: expected call not found.
Expected: mock(min_obs=19, covariance_shrinkage='ledoit_wolf', vol_regime='single', vol_regime_window=12)
  Actual: mock(min_obs=19, covariance_shrinkage='ledoit_wolf', vol_regime='two_state', vol_regime_window=12)
1 failed, 13 warnings in 1.88s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.71s
```

## invalid-covariance

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-covariance]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-covariance]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 3.929
- Green exit: 0; wall seconds: 3.894
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `7875c39b645d20f9e7877726919b6c2580b93ce345ff37ff9a6ae934bebcf420`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -115 +115 @@
-    if value_str not in allowed:
+    if value_str not in allowed and value_str != "unsupported":
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-covariance] _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x72d79914be10>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-12/test_pa_calibrate_rejects_inva0')
setting = 'covariance_shrinkage', bad_value = 'unsupported'
allowed = "['ledoit_wolf', 'none']"

    @pytest.mark.parametrize(
        ("setting", "bad_value", "allowed"),
        [
            ("covariance_shrinkage", "unsupported", "['ledoit_wolf', 'none']"),
            ("vol_regime", "unsupported", "['single', 'two_state']"),
        ],
        ids=["invalid-covariance", "invalid-regime"],
    )
    def test_pa_calibrate_rejects_invalid_template_before_loading_data(
        monkeypatch, tmp_path, setting, bad_value, allowed
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        mapping_path = tmp_path / "mapping.yaml"
        mapping_path.write_text(f"calibration:\n  {setting}: {bad_value}\n")
>       with pytest.raises(ValueError) as exc_info:
             ^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_coverage_round_2381.py:654: Failed
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to asset_library.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-covariance] - Failed: DID NOT RAISE ValueError
1 failed, 13 warnings in 1.83s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.73s
```

## invalid-regime

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-regime]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-regime]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.154
- Green exit: 0; wall seconds: 3.917
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `7875c39b645d20f9e7877726919b6c2580b93ce345ff37ff9a6ae934bebcf420`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -115 +115 @@
-    if value_str not in allowed:
+    if value_str not in allowed and value_str != "unsupported":
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-regime] _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x71a78dd2fe10>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-14/test_pa_calibrate_rejects_inva0')
setting = 'vol_regime', bad_value = 'unsupported'
allowed = "['single', 'two_state']"

    @pytest.mark.parametrize(
        ("setting", "bad_value", "allowed"),
        [
            ("covariance_shrinkage", "unsupported", "['ledoit_wolf', 'none']"),
            ("vol_regime", "unsupported", "['single', 'two_state']"),
        ],
        ids=["invalid-covariance", "invalid-regime"],
    )
    def test_pa_calibrate_rejects_invalid_template_before_loading_data(
        monkeypatch, tmp_path, setting, bad_value, allowed
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        mapping_path = tmp_path / "mapping.yaml"
        mapping_path.write_text(f"calibration:\n  {setting}: {bad_value}\n")
>       with pytest.raises(ValueError) as exc_info:
             ^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_coverage_round_2381.py:654: Failed
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to asset_library.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-regime] - Failed: DID NOT RAISE ValueError
1 failed, 13 warnings in 1.89s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.72s
```

## calibration-frame

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[defaults]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[defaults]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.068
- Green exit: 0; wall seconds: 3.95
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `5bbd5d5a5e13120e554902491683efe72324103304212f1c609a804f0c59f90e`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -460 +460 @@
-        data_result = calib.calibrate(df, index_id=args.index_id)
+        data_result = calib.calibrate(df.copy(), index_id=args.index_id)
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_______ test_pa_calibrate_input_forwards_settings_and_exports[defaults] ________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x79f928d4fbb0>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-16/test_pa_calibrate_input_forwar0')
capsys = <_pytest.capture.CaptureFixture object at 0x79f928d7a900>
mapping_text = None, cli_options = []
expected_settings = ('none', 'single', 12)

    @pytest.mark.parametrize(
        ("mapping_text", "cli_options", "expected_settings"),
        [
            pytest.param(None, [], ("none", "single", 12), id="defaults"),
            pytest.param(
                "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n",
                [],
                ("ledoit_wolf", "two_state", 5),
                id="flat-template",
            ),
            pytest.param(
                "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: '9'\n",
                [],
                ("ledoit_wolf", "two_state", 9),
                id="nested-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: 9\n",
                ["--cov-shrinkage", "none", "--vol-regime", "single", "--vol-regime-window", "3"],
                ("none", "single", 3),
                id="cli-overrides-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n",
                [],
                ("ledoit_wolf", "single", 12),
                id="partial-template",
            ),
        ],
    )
    def test_pa_calibrate_input_forwards_settings_and_exports(
        monkeypatch, tmp_path, capsys, mapping_text, cli_options, expected_settings
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        importer = (
            importer_class.return_value
            if mapping_text is None
            else importer_class.from_template.return_value
        )
        importer.min_obs = 7 if mapping_text is None else 19
        importer.allow_below_min_obs = False
        frame = pd.DataFrame({"id": ["IDX"], "return": [0.01]})
        importer.load.return_value = frame
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        input_path = tmp_path / "returns.csv"
        output_path = tmp_path / "assets.yaml"
        args = [
            "calibrate",
            "--input",
            str(input_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--min-obs",
            "7",
        ]
        if mapping_text is not None:
            mapping_path = tmp_path / "mapping.yaml"
            mapping_path.write_text(mapping_text)
            args.extend(["--mapping", str(mapping_path)])
        pa_entry.main([*args, *cli_options])

        if mapping_text is None:
            importer_class.assert_called_once_with(min_obs=7, allow_below_min_obs=True)
            importer_class.from_template.assert_not_called()
        else:
            importer_class.assert_not_called()
            importer_class.from_template.assert_called_once_with(str(mapping_path))
            assert importer.allow_below_min_obs is True
        importer.load.assert_called_once_with(str(input_path))
        covariance_shrinkage, vol_regime, window = expected_settings
        calibrator_class.assert_called_once_with(
            min_obs=importer.min_obs,
            covariance_shrinkage=covariance_shrinkage,
            vol_regime=vol_regime,
            vol_regime_window=window,
        )
        calibrator = calibrator_class.return_value
        calibrator.calibrate.assert_called_once()
>       assert calibrator.calibrate.call_args.args[0] is frame
E       assert     id  return\n0  IDX    0.01 is     id  return\n0  IDX    0.01

tests/test_coverage_round_2381.py:629: AssertionError
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to /tmp/pytest-of-runner/pytest-16/test_pa_calibrate_input_forwar0/assets.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[defaults] - assert     id  return\n0  IDX    0.01 is     id  return\n0  IDX    0.01
1 failed, 13 warnings in 1.89s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.74s
```

## export-destination

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[defaults]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[defaults]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.037
- Green exit: 0; wall seconds: 3.822
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `c8af6f3cc35850cf19713e72ebab9fdda01ae717705bfa9cc4cb755ee814f39c`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -461 +461 @@
-        calib.to_yaml(data_result, args.output)
+        calib.to_yaml(data_result, "wrong.yaml")
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_______ test_pa_calibrate_input_forwards_settings_and_exports[defaults] ________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7824b7f8fbb0>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-18/test_pa_calibrate_input_forwar0')
capsys = <_pytest.capture.CaptureFixture object at 0x7824b7fba900>
mapping_text = None, cli_options = []
expected_settings = ('none', 'single', 12)

    @pytest.mark.parametrize(
        ("mapping_text", "cli_options", "expected_settings"),
        [
            pytest.param(None, [], ("none", "single", 12), id="defaults"),
            pytest.param(
                "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n",
                [],
                ("ledoit_wolf", "two_state", 5),
                id="flat-template",
            ),
            pytest.param(
                "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: '9'\n",
                [],
                ("ledoit_wolf", "two_state", 9),
                id="nested-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: 9\n",
                ["--cov-shrinkage", "none", "--vol-regime", "single", "--vol-regime-window", "3"],
                ("none", "single", 3),
                id="cli-overrides-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n",
                [],
                ("ledoit_wolf", "single", 12),
                id="partial-template",
            ),
        ],
    )
    def test_pa_calibrate_input_forwards_settings_and_exports(
        monkeypatch, tmp_path, capsys, mapping_text, cli_options, expected_settings
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        importer = (
            importer_class.return_value
            if mapping_text is None
            else importer_class.from_template.return_value
        )
        importer.min_obs = 7 if mapping_text is None else 19
        importer.allow_below_min_obs = False
        frame = pd.DataFrame({"id": ["IDX"], "return": [0.01]})
        importer.load.return_value = frame
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        input_path = tmp_path / "returns.csv"
        output_path = tmp_path / "assets.yaml"
        args = [
            "calibrate",
            "--input",
            str(input_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--min-obs",
            "7",
        ]
        if mapping_text is not None:
            mapping_path = tmp_path / "mapping.yaml"
            mapping_path.write_text(mapping_text)
            args.extend(["--mapping", str(mapping_path)])
        pa_entry.main([*args, *cli_options])

        if mapping_text is None:
            importer_class.assert_called_once_with(min_obs=7, allow_below_min_obs=True)
            importer_class.from_template.assert_not_called()
        else:
            importer_class.assert_not_called()
            importer_class.from_template.assert_called_once_with(str(mapping_path))
            assert importer.allow_below_min_obs is True
        importer.load.assert_called_once_with(str(input_path))
        covariance_shrinkage, vol_regime, window = expected_settings
        calibrator_class.assert_called_once_with(
            min_obs=importer.min_obs,
            covariance_shrinkage=covariance_shrinkage,
            vol_regime=vol_regime,
            vol_regime_window=window,
        )
        calibrator = calibrator_class.return_value
        calibrator.calibrate.assert_called_once()
        assert calibrator.calibrate.call_args.args[0] is frame
        assert calibrator.calibrate.call_args.kwargs == {"index_id": "IDX"}
>       calibrator.to_yaml.assert_called_once_with(calibrator.calibrate.return_value, str(output_path))

tests/test_coverage_round_2381.py:631:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:998: in assert_called_once_with
    return self.assert_called_with(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <MagicMock name='mock().to_yaml' spec='function' id='132099100135440'>
args = (<MagicMock name='mock().calibrate()' id='132099100135104'>, '/tmp/pytest-of-runner/pytest-18/test_pa_calibrate_input_forwar0/assets.yaml')
kwargs = {}

    def assert_called_with(self, /, *args, **kwargs):
        """assert that the last call was made with the specified arguments.

        Raises an AssertionError if the args and keyword args passed in are
        different to the last call to the mock."""
        if self.call_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            actual = 'not called.'
            error_message = ('expected call not found.\nExpected: %s\n  Actual: %s'
                    % (expected, actual))
            raise AssertionError(error_message)

        def _error_message():
            msg = self._format_mock_failure_message(args, kwargs)
            return msg
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        actual = self._call_matcher(self.call_args)
        if actual != expected:
            cause = expected if isinstance(expected, Exception) else None
>           raise AssertionError(_error_message()) from cause
E           AssertionError: expected call not found.
E           Expected: to_yaml(<MagicMock name='mock().calibrate()' id='132099100135104'>, '/tmp/pytest-of-runner/pytest-18/test_pa_calibrate_input_forwar0/assets.yaml')
E             Actual: to_yaml(<MagicMock name='mock().calibrate()' id='132099100135104'>, 'wrong.yaml')

/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:986: AssertionError
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to /tmp/pytest-of-runner/pytest-18/test_pa_calibrate_input_forwar0/assets.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[defaults] - AssertionError: expected call not found.
Expected: to_yaml(<MagicMock name='mock().calibrate()' id='132099100135104'>, '/tmp/pytest-of-runner/pytest-18/test_pa_calibrate_input_forwar0/assets.yaml')
  Actual: to_yaml(<MagicMock name='mock().calibrate()' id='132099100135104'>, 'wrong.yaml')
1 failed, 13 warnings in 1.90s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.69s
```

## template-observations

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[nested-template]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[nested-template]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.068
- Green exit: 0; wall seconds: 3.852
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `b68bacafa290df5e41bfaeb9f277834bc5976af1d1e03afd292b39cd81534a06`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -455 +455 @@
-            min_obs=importer.min_obs,
+            min_obs=args.min_obs,
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
____ test_pa_calibrate_input_forwards_settings_and_exports[nested-template] ____

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7ea47c34fce0>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-20/test_pa_calibrate_input_forwar0')
capsys = <_pytest.capture.CaptureFixture object at 0x7ea47c382900>
mapping_text = "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n  vol_regime: two_state\n  vol_regime_window: '9'\n"
cli_options = [], expected_settings = ('ledoit_wolf', 'two_state', 9)

    @pytest.mark.parametrize(
        ("mapping_text", "cli_options", "expected_settings"),
        [
            pytest.param(None, [], ("none", "single", 12), id="defaults"),
            pytest.param(
                "covariance_shrinkage: ledoit_wolf\nvol_regime: two_state\nvol_regime_window: '5'\n",
                [],
                ("ledoit_wolf", "two_state", 5),
                id="flat-template",
            ),
            pytest.param(
                "vol_regime: single\ncalibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: '9'\n",
                [],
                ("ledoit_wolf", "two_state", 9),
                id="nested-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n"
                "  vol_regime: two_state\n  vol_regime_window: 9\n",
                ["--cov-shrinkage", "none", "--vol-regime", "single", "--vol-regime-window", "3"],
                ("none", "single", 3),
                id="cli-overrides-template",
            ),
            pytest.param(
                "calibration:\n  covariance_shrinkage: ledoit_wolf\n",
                [],
                ("ledoit_wolf", "single", 12),
                id="partial-template",
            ),
        ],
    )
    def test_pa_calibrate_input_forwards_settings_and_exports(
        monkeypatch, tmp_path, capsys, mapping_text, cli_options, expected_settings
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        importer = (
            importer_class.return_value
            if mapping_text is None
            else importer_class.from_template.return_value
        )
        importer.min_obs = 7 if mapping_text is None else 19
        importer.allow_below_min_obs = False
        frame = pd.DataFrame({"id": ["IDX"], "return": [0.01]})
        importer.load.return_value = frame
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        input_path = tmp_path / "returns.csv"
        output_path = tmp_path / "assets.yaml"
        args = [
            "calibrate",
            "--input",
            str(input_path),
            "--index-id",
            "IDX",
            "--output",
            str(output_path),
            "--min-obs",
            "7",
        ]
        if mapping_text is not None:
            mapping_path = tmp_path / "mapping.yaml"
            mapping_path.write_text(mapping_text)
            args.extend(["--mapping", str(mapping_path)])
        pa_entry.main([*args, *cli_options])

        if mapping_text is None:
            importer_class.assert_called_once_with(min_obs=7, allow_below_min_obs=True)
            importer_class.from_template.assert_not_called()
        else:
            importer_class.assert_not_called()
            importer_class.from_template.assert_called_once_with(str(mapping_path))
            assert importer.allow_below_min_obs is True
        importer.load.assert_called_once_with(str(input_path))
        covariance_shrinkage, vol_regime, window = expected_settings
>       calibrator_class.assert_called_once_with(
            min_obs=importer.min_obs,
            covariance_shrinkage=covariance_shrinkage,
            vol_regime=vol_regime,
            vol_regime_window=window,
        )

tests/test_coverage_round_2381.py:621:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:998: in assert_called_once_with
    return self.assert_called_with(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <MagicMock spec='CalibrationAgent' id='139244923046608'>, args = ()
kwargs = {'min_obs': 19, 'covariance_shrinkage': 'ledoit_wolf', 'vol_regime': 'two_state', 'vol_regime_window': 9}

    def assert_called_with(self, /, *args, **kwargs):
        """assert that the last call was made with the specified arguments.

        Raises an AssertionError if the args and keyword args passed in are
        different to the last call to the mock."""
        if self.call_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            actual = 'not called.'
            error_message = ('expected call not found.\nExpected: %s\n  Actual: %s'
                    % (expected, actual))
            raise AssertionError(error_message)

        def _error_message():
            msg = self._format_mock_failure_message(args, kwargs)
            return msg
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        actual = self._call_matcher(self.call_args)
        if actual != expected:
            cause = expected if isinstance(expected, Exception) else None
>           raise AssertionError(_error_message()) from cause
E           AssertionError: expected call not found.
E           Expected: mock(min_obs=19, covariance_shrinkage='ledoit_wolf', vol_regime='two_state', vol_regime_window=9)
E             Actual: mock(min_obs=7, covariance_shrinkage='ledoit_wolf', vol_regime='two_state', vol_regime_window=9)

/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:986: AssertionError
----------------------------- Captured stdout call -----------------------------
✓ Wrote asset library to /tmp/pytest-of-runner/pytest-20/test_pa_calibrate_input_forwar0/assets.yaml
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_input_forwards_settings_and_exports[nested-template] - AssertionError: expected call not found.
Expected: mock(min_obs=19, covariance_shrinkage='ledoit_wolf', vol_regime='two_state', vol_regime_window=9)
  Actual: mock(min_obs=7, covariance_shrinkage='ledoit_wolf', vol_regime='two_state', vol_regime_window=9)
1 failed, 13 warnings in 1.89s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.69s
```

## invalid-covariance-early-io

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-covariance]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-covariance]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.015
- Green exit: 0; wall seconds: 3.797
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `8aeb5dc0d855137a8bc7398b008026ef62a56215691f1287bd488760685ce8af`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -423,0 +424 @@
+        importer.load(args.input)
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-covariance] _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7cadc565fe10>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-22/test_pa_calibrate_rejects_inva0')
setting = 'covariance_shrinkage', bad_value = 'unsupported'
allowed = "['ledoit_wolf', 'none']"

    @pytest.mark.parametrize(
        ("setting", "bad_value", "allowed"),
        [
            ("covariance_shrinkage", "unsupported", "['ledoit_wolf', 'none']"),
            ("vol_regime", "unsupported", "['single', 'two_state']"),
        ],
        ids=["invalid-covariance", "invalid-regime"],
    )
    def test_pa_calibrate_rejects_invalid_template_before_loading_data(
        monkeypatch, tmp_path, setting, bad_value, allowed
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        mapping_path = tmp_path / "mapping.yaml"
        mapping_path.write_text(f"calibration:\n  {setting}: {bad_value}\n")
        with pytest.raises(ValueError) as exc_info:
            pa_entry.main(
                [
                    "calibrate",
                    "--input",
                    str(tmp_path / "missing.csv"),
                    "--index-id",
                    "IDX",
                    "--mapping",
                    str(mapping_path),
                ]
            )
        assert str(exc_info.value) == f"{setting} must be one of {allowed}"
>       importer_class.from_template.return_value.load.assert_not_called()

tests/test_coverage_round_2381.py:667:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <MagicMock name='mock.from_template().load' id='137085780369360'>

    def assert_not_called(self):
        """assert that the mock was never called.
        """
        if self.call_count != 0:
            msg = ("Expected '%s' to not have been called. Called %s times.%s"
                   % (self._mock_name or 'mock',
                      self.call_count,
                      self._calls_repr()))
>           raise AssertionError(msg)
E           AssertionError: Expected 'load' to not have been called. Called 1 times.
E           Calls: [call('/tmp/pytest-of-runner/pytest-22/test_pa_calibrate_rejects_inva0/missing.csv')].

/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:947: AssertionError
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-covariance] - AssertionError: Expected 'load' to not have been called. Called 1 times.
Calls: [call('/tmp/pytest-of-runner/pytest-22/test_pa_calibrate_rejects_inva0/missing.csv')].
1 failed, 13 warnings in 1.86s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.66s
```

## invalid-regime-early-io

Named test: `tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-regime]`.

Command: `python -m pytest 'tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-regime]' -m 'not slow and not live_llm' -q`

- Red exit: 1; wall seconds: 4.101
- Green exit: 0; wall seconds: 3.779
- SHA256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- SHA256 mutated: `8aeb5dc0d855137a8bc7398b008026ef62a56215691f1287bd488760685ce8af`
- SHA256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)

```diff
--- a/pa_core/pa.py
+++ b/pa_core/pa.py
@@ -423,0 +424 @@
+        importer.load(args.input)
```

Red output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-regime] _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x723340a77e10>
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-24/test_pa_calibrate_rejects_inva0')
setting = 'vol_regime', bad_value = 'unsupported'
allowed = "['single', 'two_state']"

    @pytest.mark.parametrize(
        ("setting", "bad_value", "allowed"),
        [
            ("covariance_shrinkage", "unsupported", "['ledoit_wolf', 'none']"),
            ("vol_regime", "unsupported", "['single', 'two_state']"),
        ],
        ids=["invalid-covariance", "invalid-regime"],
    )
    def test_pa_calibrate_rejects_invalid_template_before_loading_data(
        monkeypatch, tmp_path, setting, bad_value, allowed
    ) -> None:
        from pa_core import data

        importer_class = create_autospec(data.DataImportAgent)
        calibrator_class = create_autospec(data.CalibrationAgent)
        monkeypatch.setattr(data, "DataImportAgent", importer_class)
        monkeypatch.setattr(data, "CalibrationAgent", calibrator_class)
        mapping_path = tmp_path / "mapping.yaml"
        mapping_path.write_text(f"calibration:\n  {setting}: {bad_value}\n")
        with pytest.raises(ValueError) as exc_info:
            pa_entry.main(
                [
                    "calibrate",
                    "--input",
                    str(tmp_path / "missing.csv"),
                    "--index-id",
                    "IDX",
                    "--mapping",
                    str(mapping_path),
                ]
            )
        assert str(exc_info.value) == f"{setting} must be one of {allowed}"
>       importer_class.from_template.return_value.load.assert_not_called()

tests/test_coverage_round_2381.py:667:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <MagicMock name='mock.from_template().load' id='125564451086624'>

    def assert_not_called(self):
        """assert that the mock was never called.
        """
        if self.call_count != 0:
            msg = ("Expected '%s' to not have been called. Called %s times.%s"
                   % (self._mock_name or 'mock',
                      self.call_count,
                      self._calls_repr()))
>           raise AssertionError(msg)
E           AssertionError: Expected 'load' to not have been called. Called 1 times.
E           Calls: [call('/tmp/pytest-of-runner/pytest-24/test_pa_calibrate_rejects_inva0/missing.csv')].

/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:947: AssertionError
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2381.py::test_pa_calibrate_rejects_invalid_template_before_loading_data[invalid-regime] - AssertionError: Expected 'load' to not have been called. Called 1 times.
Calls: [call('/tmp/pytest-of-runner/pytest-24/test_pa_calibrate_rejects_inva0/missing.csv')].
1 failed, 13 warnings in 1.96s
```

Restored green output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 13 warnings in 1.66s
```
