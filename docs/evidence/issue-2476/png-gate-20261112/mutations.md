# PNG test production mutations

All mutations were temporary; original production bytes were restored in `finally`.

## omit-image-write

Test: `tests/test_cli.py::test_main_with_png`

Command: `timeout -k 5s 120s /opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pytest tests/test_cli.py::test_main_with_png -m 'not slow and not live_llm' -q`

```diff
--- a/pa_core/viz/export_backend.py
+++ b/pa_core/viz/export_backend.py
@@ -101,7 +101,7 @@
         write_opts: dict[str, Any] = dict(clean_opts)
         if format is not None:
             write_opts["format"] = format
-        fig.write_image(path, **write_opts)
+        return
         return
     image_format = format or Path(path).suffix.lstrip(".") or "png"
     Path(path).write_bytes(figure_to_image_bytes(fig, format=image_format, **clean_opts))
```

SHA256 before: `d7f5f9c9ec93ba23af09be71a6b765b38ae3c1ae6cd025ac4e71a8a28e4e608a`

SHA256 mutated: `517582c31839219d2e9f5abfcca207d582eb77c2e6202db26071885d46f0ec0b`

SHA256 restored: `d7f5f9c9ec93ba23af09be71a6b765b38ae3c1ae6cd025ac4e71a8a28e4e608a`

RED exit 1

```text
F                                                                        [100%]
=================================== FAILURES ===================================
______________________________ test_main_with_png ______________________________

tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-6/test_main_with_png0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x73a50fb5f6f0>

    def test_main_with_png(tmp_path, monkeypatch):
        """Run the CLI and image backend while isolating the external browser renderer."""
    
        def write_image(figure, path, **_kwargs):
            Path(path).write_bytes(b"rendered-png")
    
        renderer = create_autospec(go.Figure.write_image, side_effect=write_image)
        monkeypatch.setattr(go.Figure, "write_image", renderer)
        cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1, "financing_mode": "broadcast"}
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
        out_file = tmp_path / "out.xlsx"
        monkeypatch.chdir(tmp_path)
        main(
            [
                "--config",
                str(cfg_path),
                "--index",
                str(idx_csv),
                "--output",
                str(out_file),
                "--png",
            ]
        )
        assert out_file.exists()
>       renderer.assert_called_once()

/home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/tests/test_cli.py:56: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:244: in assert_called_once
    return mock.assert_called_once(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <MagicMock spec='function' id='127152768653296'>

    def assert_called_once(self):
        """assert that the mock was called only once.
        """
        if not self.call_count == 1:
            msg = ("Expected '%s' to have been called once. Called %s times.%s"
                   % (self._mock_name or 'mock',
                      self.call_count,
                      self._calls_repr()))
>           raise AssertionError(msg)
E           AssertionError: Expected 'mock' to have been called once. Called 0 times.

/opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/unittest/mock.py:965: AssertionError
----------------------------- Captured stdout call -----------------------------
[BACKEND] Using backend: numpy
----------------------------- Captured stderr call -----------------------------

sweep:   0%|          | 0/81 [00:00<?, ?it/s]
sweep:  49%|████▉     | 40/81 [00:00<00:00, 393.18it/s]
sweep: 100%|██████████| 81/81 [00:00<00:00, 412.70it/s]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/tests/test_cli.py:44: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/data/loaders.py:71: Pandas4Warning: A future version of pandas will return a BaseOffset object instead of a string from infer_freq. Use pd.set_option('future.infer_freq_returns_offset', True) to get the future behavior, or set to False to keep the old behavior and silence this warning. To preserve the string representation, use ``infer_freq(...).freqstr``.
    inferred = pd.infer_freq(date_values_sorted)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/cli.py:1302: UserWarning: Run executed without an explicit --seed (seed=None): results use a non-deterministic RNG and cannot be reproduced. Pass --seed to make the run reproducible.
    mw.write(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_cli.py::test_main_with_png - AssertionError: Expected 'mock' to have been called once. Called 0 times.
1 failed, 17 warnings in 3.18s
```

GREEN exit 0

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/tests/test_cli.py:44: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/data/loaders.py:71: Pandas4Warning: A future version of pandas will return a BaseOffset object instead of a string from infer_freq. Use pd.set_option('future.infer_freq_returns_offset', True) to get the future behavior, or set to False to keep the old behavior and silence this warning. To preserve the string representation, use ``infer_freq(...).freqstr``.
    inferred = pd.infer_freq(date_values_sorted)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/cli.py:1302: UserWarning: Run executed without an explicit --seed (seed=None): results use a non-deterministic RNG and cannot be reproduced. Pass --seed to make the run reproducible.
    mw.write(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 17 warnings in 2.97s
```

## wrong-png-destination

Test: `tests/test_cli.py::test_main_with_png`

Command: `timeout -k 5s 120s /opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pytest tests/test_cli.py::test_main_with_png -m 'not slow and not live_llm' -q`

```diff
--- a/pa_core/cli.py
+++ b/pa_core/cli.py
@@ -1386,7 +1386,7 @@
             if flags.png:
                 sweep_png_path = _export_path(".png")
                 _attempt_sweep_export(
-                    "PNG", sweep_png_path, lambda: write_figure_image(sweep_fig, sweep_png_path)
+                    "PNG", sweep_png_path, lambda: write_figure_image(sweep_fig, sweep_png_path.with_suffix(".pdf"))
                 )
             if flags.pdf:
                 sweep_pdf_path = _export_path(".pdf")
```

SHA256 before: `f3ffec1a90fe5888ca5ba357507e7542f4dc436117acea430348c87ab0b40e26`

SHA256 mutated: `3e23f3a0c883fab9ac7fbd664b5bbfd8d76e30a4558a478e052bccc9548fa851`

SHA256 restored: `f3ffec1a90fe5888ca5ba357507e7542f4dc436117acea430348c87ab0b40e26`

RED exit 1

```text
F                                                                        [100%]
=================================== FAILURES ===================================
______________________________ test_main_with_png ______________________________

tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-8/test_main_with_png0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7dd12463f950>

    def test_main_with_png(tmp_path, monkeypatch):
        """Run the CLI and image backend while isolating the external browser renderer."""
    
        def write_image(figure, path, **_kwargs):
            Path(path).write_bytes(b"rendered-png")
    
        renderer = create_autospec(go.Figure.write_image, side_effect=write_image)
        monkeypatch.setattr(go.Figure, "write_image", renderer)
        cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1, "financing_mode": "broadcast"}
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
        out_file = tmp_path / "out.xlsx"
        monkeypatch.chdir(tmp_path)
        main(
            [
                "--config",
                str(cfg_path),
                "--index",
                str(idx_csv),
                "--output",
                str(out_file),
                "--png",
            ]
        )
        assert out_file.exists()
        renderer.assert_called_once()
        figure, image_path = renderer.call_args.args
        assert isinstance(figure, go.Figure)
        assert figure.data
>       assert image_path == out_file.with_suffix(".png")
E       AssertionError: assert PosixPath('/tmp/pytest-of-runner/pytest-8/test_main_with_png0/out.pdf') == PosixPath('/tmp/pytest-of-runner/pytest-8/test_main_with_png0/out.png')
E        +  where PosixPath('/tmp/pytest-of-runner/pytest-8/test_main_with_png0/out.png') = with_suffix('.png')
E        +    where with_suffix = PosixPath('/tmp/pytest-of-runner/pytest-8/test_main_with_png0/out.xlsx').with_suffix

/home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/tests/test_cli.py:60: AssertionError
----------------------------- Captured stdout call -----------------------------
[BACKEND] Using backend: numpy
----------------------------- Captured stderr call -----------------------------

sweep:   0%|          | 0/81 [00:00<?, ?it/s]
sweep:  53%|█████▎    | 43/81 [00:00<00:00, 425.83it/s]
sweep: 100%|██████████| 81/81 [00:00<00:00, 437.39it/s]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/tests/test_cli.py:44: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/data/loaders.py:71: Pandas4Warning: A future version of pandas will return a BaseOffset object instead of a string from infer_freq. Use pd.set_option('future.infer_freq_returns_offset', True) to get the future behavior, or set to False to keep the old behavior and silence this warning. To preserve the string representation, use ``infer_freq(...).freqstr``.
    inferred = pd.infer_freq(date_values_sorted)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/cli.py:1302: UserWarning: Run executed without an explicit --seed (seed=None): results use a non-deterministic RNG and cannot be reproduced. Pass --seed to make the run reproducible.
    mw.write(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_cli.py::test_main_with_png - AssertionError: assert PosixPath('/tmp/pytest-of-runner/pytest-8/test_main_with_png0/out.pdf') == PosixPath('/tmp/pytest-of-runner/pytest-8/test_main_with_png0/out.png')
 +  where PosixPath('/tmp/pytest-of-runner/pytest-8/test_main_with_png0/out.png') = with_suffix('.png')
 +    where with_suffix = PosixPath('/tmp/pytest-of-runner/pytest-8/test_main_with_png0/out.xlsx').with_suffix
1 failed, 17 warnings in 3.11s
```

GREEN exit 0

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/tests/test_cli.py:44: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/data/loaders.py:71: Pandas4Warning: A future version of pandas will return a BaseOffset object instead of a string from infer_freq. Use pd.set_option('future.infer_freq_returns_offset', True) to get the future behavior, or set to False to keep the old behavior and silence this warning. To preserve the string representation, use ``infer_freq(...).freqstr``.
    inferred = pd.infer_freq(date_values_sorted)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/cli.py:1302: UserWarning: Run executed without an explicit --seed (seed=None): results use a non-deterministic RNG and cannot be reproduced. Pass --seed to make the run reproducible.
    mw.write(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 17 warnings in 2.66s
```

## empty-export-figure

Test: `tests/test_cli.py::test_main_with_png`

Command: `timeout -k 5s 120s /opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pytest tests/test_cli.py::test_main_with_png -m 'not slow and not live_llm' -q`

```diff
--- a/pa_core/cli.py
+++ b/pa_core/cli.py
@@ -1386,7 +1386,7 @@
             if flags.png:
                 sweep_png_path = _export_path(".png")
                 _attempt_sweep_export(
-                    "PNG", sweep_png_path, lambda: write_figure_image(sweep_fig, sweep_png_path)
+                    "PNG", sweep_png_path, lambda: write_figure_image(type(sweep_fig)(), sweep_png_path)
                 )
             if flags.pdf:
                 sweep_pdf_path = _export_path(".pdf")
```

SHA256 before: `f3ffec1a90fe5888ca5ba357507e7542f4dc436117acea430348c87ab0b40e26`

SHA256 mutated: `3c17ff3fa82c8f117a85f850317d07f5e87d5e3ad4c3d526c76886b9449d37c1`

SHA256 restored: `f3ffec1a90fe5888ca5ba357507e7542f4dc436117acea430348c87ab0b40e26`

RED exit 1

```text
F                                                                        [100%]
=================================== FAILURES ===================================
______________________________ test_main_with_png ______________________________

tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-10/test_main_with_png0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7e62e2dcf950>

    def test_main_with_png(tmp_path, monkeypatch):
        """Run the CLI and image backend while isolating the external browser renderer."""
    
        def write_image(figure, path, **_kwargs):
            Path(path).write_bytes(b"rendered-png")
    
        renderer = create_autospec(go.Figure.write_image, side_effect=write_image)
        monkeypatch.setattr(go.Figure, "write_image", renderer)
        cfg = {"N_SIMULATIONS": 2, "N_MONTHS": 1, "financing_mode": "broadcast"}
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
        out_file = tmp_path / "out.xlsx"
        monkeypatch.chdir(tmp_path)
        main(
            [
                "--config",
                str(cfg_path),
                "--index",
                str(idx_csv),
                "--output",
                str(out_file),
                "--png",
            ]
        )
        assert out_file.exists()
        renderer.assert_called_once()
        figure, image_path = renderer.call_args.args
        assert isinstance(figure, go.Figure)
>       assert figure.data
E       AssertionError: assert ()
E        +  where () = Figure({\n    'data': [], 'layout': {'template': '...'}\n}).data

/home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/tests/test_cli.py:59: AssertionError
----------------------------- Captured stdout call -----------------------------
[BACKEND] Using backend: numpy
----------------------------- Captured stderr call -----------------------------

sweep:   0%|          | 0/81 [00:00<?, ?it/s]
sweep:  54%|█████▍    | 44/81 [00:00<00:00, 432.89it/s]
sweep: 100%|██████████| 81/81 [00:00<00:00, 433.28it/s]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/tests/test_cli.py:44: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/data/loaders.py:71: Pandas4Warning: A future version of pandas will return a BaseOffset object instead of a string from infer_freq. Use pd.set_option('future.infer_freq_returns_offset', True) to get the future behavior, or set to False to keep the old behavior and silence this warning. To preserve the string representation, use ``infer_freq(...).freqstr``.
    inferred = pd.infer_freq(date_values_sorted)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/cli.py:1302: UserWarning: Run executed without an explicit --seed (seed=None): results use a non-deterministic RNG and cannot be reproduced. Pass --seed to make the run reproducible.
    mw.write(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_cli.py::test_main_with_png - AssertionError: assert ()
 +  where () = Figure({\n    'data': [], 'layout': {'template': '...'}\n}).data
1 failed, 17 warnings in 2.64s
```

GREEN exit 0

```text
.                                                                        [100%]
=============================== warnings summary ===============================
../../../../../opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: 13 warnings
  /opt/hostedtoolcache/Python/3.14.10/x64/lib/python3.14/site-packages/langsmith/_internal/_beta_decorator.py:92: DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated and slated for removal in Python 3.16; use inspect.iscoroutinefunction() instead
    if asyncio.iscoroutinefunction(func):

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/tests/test_cli.py:44: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/data/loaders.py:71: Pandas4Warning: A future version of pandas will return a BaseOffset object instead of a string from infer_freq. Use pd.set_option('future.infer_freq_returns_offset', True) to get the future behavior, or set to False to keep the old behavior and silence this warning. To preserve the string representation, use ``infer_freq(...).freqstr``.
    inferred = pd.infer_freq(date_values_sorted)

tests/test_cli.py::test_main_with_png
  /home/runner/work/Portable-Alpha-Extension-Model/Portable-Alpha-Extension-Model/pa_core/cli.py:1302: UserWarning: Run executed without an explicit --seed (seed=None): results use a non-deterministic RNG and cannot be reproduced. Pass --seed to make the run reproducible.
    mw.write(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 17 warnings in 2.50s
```
