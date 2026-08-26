# AfricaS2S — Test Specification

Tests are layered: fast unit tests with synthetic data, integration tests that run real methods on small real data, and end-to-end pipeline tests.

All tests use a common fixture set of small xarray arrays (synthetic or tiny real subsets) so the full unit suite runs in seconds.

---

## Fixtures

Shared synthetic data for unit tests. Defined once, used everywhere.

```
fixture: synthetic_gcm_hindcast
  xr.DataArray with dims (year: 10, member: 3, lat: 5, lon: 5)
  Coarse grid: 2° resolution over a small bbox
  Values: spatially correlated random field with a planted seasonal signal

fixture: synthetic_gcm_forecast
  xr.DataArray with dims (member: 3, lat: 5, lon: 5)
  Same grid as hindcast, single target season

fixture: synthetic_obs
  xr.DataArray with dims (year: 10, lat: 20, lon: 20)
  Fine grid: 0.5° resolution over same bbox
  Values: correlated with gcm signal + spatial detail + noise

fixture: perfect_forecast
  Obs data passed as forecast (should yield perfect skill)

fixture: climatology_forecast
  Uniform 1/3 tercile probabilities everywhere (should yield zero skill)
```

---

## 1. Unit tests — methods

### 1.1 Method base class

```
test_method_base_is_abstract
  Instantiating MethodBase directly raises TypeError.
  Subclass must implement fit() and predict().

test_method_registry_lookup
  registry.get_method("bcsd") returns BCSDMethod class.
  registry.get_method("cca") returns CCAMethod class.
  registry.get_method("nonexistent") raises KeyError.

test_register_method_decorator
  A class decorated with @register_method("test_m") is retrievable via registry.
```

### 1.2 BCSD method

```
test_bcsd_fit_stores_state
  method = BCSDMethod()
  method.fit(synthetic_gcm_hindcast, synthetic_obs)
  Assert method has learned parameters (quantile maps, spatial weights, etc.)

test_bcsd_predict_shape
  method.fit(hindcast, obs)
  result = method.predict(forecast)
  Assert result.dims == ("member", "lat", "lon")  [continuous]
  Assert result.lat matches obs.lat (fine resolution)
  Assert result.lon matches obs.lon

test_bcsd_predict_values_plausible
  Result values are within a physically plausible range.
  Result is not all NaN, not constant.

test_bcsd_respects_output_type
  output_type="tercile" → result.dims == ("tercile", "lat", "lon"), values sum to 1.0 per pixel
  output_type="continuous" → result.dims == ("member", "lat", "lon")
```

### 1.3 CCA method

```
test_cca_fit_stores_state
  CCA fits canonical correlation patterns. After fit(), method has modes/loadings.

test_cca_predict_shape
  Same shape assertions as BCSD.

test_cca_with_few_modes
  CCA with n_modes=2 on synthetic data doesn't crash, produces valid output.

test_cca_with_short_hindcast
  Hindcast with only 5 years: method still fits (may warn), produces output.
```

---

## 2. Unit tests — metrics

### 2.1 RPSS

```
test_rpss_perfect_forecast
  RPSS(perfect_tercile_forecast, obs) → 1.0 (or close to it).

test_rpss_climatology_forecast
  RPSS(uniform_1/3_everywhere, obs) → 0.0.

test_rpss_worse_than_climatology
  RPSS(inverted_forecast, obs) → negative value.

test_rpss_shape_spatial
  With spatial=True, returns xr.DataArray with (lat, lon) dims.
```

### 2.2 ROC area

```
test_roc_perfect_discrimination
  ROC area for perfect forecast → 1.0.

test_roc_no_discrimination
  ROC area for climatology → 0.5.

test_roc_per_tercile
  Returns separate ROC values for below-normal, normal, above-normal.
```

### 2.3 Correlation

```
test_pearson_perfect
  Pearson r between identical arrays → 1.0.

test_pearson_zero
  Pearson r between uncorrelated random arrays → ~0.0 (within tolerance).
```

---

## 3. Unit tests — cross-validation

```
test_loyo_yields_correct_folds
  LOYO on 10 years → 10 folds, each with 9 train years and 1 test year.
  Every year appears as test exactly once.

test_loyo_no_leakage
  For each fold, test year is not in training set.

test_blocked_cv_preserves_order
  Blocked CV with block_size=3 on 12 years → 4 folds.
  Train/test splits respect temporal ordering.
```

---

## 4. Unit tests — ensemble

```
test_uniform_ensemble
  3 forecast arrays with known values.
  Uniform ensemble = simple mean. Assert result ≈ mean of inputs.

test_skill_weighted_ensemble
  3 forecasts with known skill [0.5, 0.3, 0.1].
  Weighted ensemble weights proportional to skill.
  Assert result is closer to the high-skill forecast.

test_ensemble_output_shape
  Output has same spatial dims as inputs.
  Tercile output: (tercile, lat, lon) with probabilities summing to 1.

test_ensemble_single_model
  Ensemble of 1 model = that model's forecast (no crash, no change).
```

---

## 5. Unit tests — skill()

```
test_skill_returns_report
  report = skill(forecast, obs, metrics=["rpss"])
  Assert report.scores is a dict with "rpss" key.
  Assert isinstance(report.scores["rpss"], float).

test_skill_spatial_maps
  report = skill(forecast, obs, metrics=["rpss"], spatial=True)
  Assert "rpss" in report.spatial.
  Assert report.spatial["rpss"].dims == ("lat", "lon").

test_skill_multiple_metrics
  report = skill(forecast, obs, metrics=["rpss", "roc", "pearson_r"])
  Assert all three keys present in report.scores.

test_skill_compare
  comparison = skill_compare({"A": fcst_a, "B": fcst_b}, obs, metrics=["rpss"])
  Assert comparison has rows for "A" and "B" with rpss values.
```

---

## 6. Unit tests — tercile conversion

```
test_continuous_to_tercile
  Given a continuous ensemble (member, lat, lon) and obs climatology,
  convert to tercile probabilities (3, lat, lon).
  Assert probabilities in [0, 1] and sum to 1.0 per pixel.

test_tercile_uniform_from_climatology
  If forecast matches obs climatology perfectly, tercile probs ≈ (1/3, 1/3, 1/3).
```

---

## 7. Integration tests (real data, slow)

These use small real data subsets. Mark with `@pytest.mark.integration`.

### 7.1 BCSD on real CFSv2 + CHIRPS

```
test_bcsd_real_data
  Load a small CFSv2 hindcast (3 years, East Africa bbox) and CHIRPS obs.
  Run downscale(gcm, obs, method="bcsd").
  Assert:
    - Output resolution matches CHIRPS
    - No NaN-only grid cells within the land mask
    - Tercile probabilities sum to 1.0
```

### 7.2 CCA on real data

```
test_cca_real_data
  Same setup as BCSD test.
  Run downscale(gcm, obs, method="cca").
  Assert same shape/validity checks.
```

---

## 8. End-to-end pipeline tests

### 8.1 Single GCM, single method

```
test_e2e_single_gcm_single_method
  result = africas2s.downscale(gcm, obs, method="bcsd", ...)
  report = africas2s.skill(result, obs, metrics=["rpss"])
  Assert report.scores["rpss"] is a finite float.
  Assert result forecast is at obs resolution.
```

### 8.2 Single GCM, optimize across methods

```
test_e2e_optimize_single_gcm
  best = africas2s.optimize(
      gcm, obs,
      methods=["bcsd", "cca"],
      cv="loyo",
      primary_metric="rpss",
  )
  Assert best.method in ["bcsd", "cca"].
  Assert best.score is a finite float.
  Assert best.forecast has correct shape.
```

### 8.3 Multi-GCM ensemble

```
test_e2e_multi_gcm_ensemble
  # Optimize 2 GCMs independently
  best1 = africas2s.optimize(gcm1, obs, methods=["bcsd", "cca"], ...)
  best2 = africas2s.optimize(gcm2, obs, methods=["bcsd", "cca"], ...)

  # Combine
  mme = africas2s.ensemble([best1, best2], obs, strategy="uniform")

  # Verify
  report = africas2s.skill(mme, obs, metrics=["rpss", "roc"])

  Assert report.scores has "rpss" and "roc" keys.
  Assert mme forecast shape matches obs grid.
```

### 8.4 Full pipeline end-to-end

```
test_e2e_full_pipeline
  This is the "does the whole thing work" test.

  1. Load GCM hindcasts for 2 models (synthetic or small real)
  2. Load obs
  3. For each GCM: optimize(methods=["bcsd", "cca"], cv="loyo", metric="rpss")
  4. Ensemble the results with strategy="uniform"
  5. Run skill() on the MME
  6. Assert:
     - Final forecast is at obs resolution
     - Tercile probabilities sum to 1.0
     - Skill report contains RPSS, ROC
     - RPSS is not NaN
     - Pipeline completes without error
```

### 8.5 Skill of uniform terciles is zero

```
test_e2e_climatology_baseline
  Create a "forecast" that is uniform 1/3 everywhere.
  skill(climatology, obs, metrics=["rpss"]) → RPSS ≈ 0.0.
  This verifies the scoring engine's baseline behavior.
```

---

## 9. Plugin contract tests

Verify that new methods/metrics can be plugged in without breaking anything.

```
test_plugin_method_contract
  Define a trivial DummyMethod that returns obs mean as forecast.
  Register it with @register_method("dummy").
  Run africas2s.downscale(gcm, obs, method="dummy").
  Assert it produces valid output shape.

test_plugin_metric_contract
  Define a trivial DummyMetric that returns 0.5 always.
  Register it with @register_metric("dummy_metric").
  Run africas2s.skill(forecast, obs, metrics=["dummy_metric"]).
  Assert report.scores["dummy_metric"] == 0.5.
```

---

## Running tests

```bash
# Unit tests only (fast, synthetic data)
pytest africas2s/tests/ -m "not integration"

# All tests
pytest africas2s/tests/

# Just pipeline tests
pytest africas2s/tests/ -k "e2e"

# Just plugin contract tests
pytest africas2s/tests/ -k "plugin"
```
