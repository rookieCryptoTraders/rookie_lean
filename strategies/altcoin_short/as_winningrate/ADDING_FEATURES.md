# Adding New Features to the Altcoin Short Strategy

This document explains how the **custom data feature pipeline** works and how to add new features (from depth, quote, or other custom data) without coupling them to the core alpha logic.

---

## 1. Overview: Where Features Live

| Layer | Location | Role |
|-------|----------|------|
| **Algorithm** | `main.py` | Subscribes to securities and custom data (depth, quote), passes `Slice` to the framework. |
| **Alpha model** | `alpha.py` | `update()` receives `Slice`, captures custom data into `latest_*` caches; `_refresh_all_factors()` → `_calculate_all_factors()` computes factors and runs the **custom feature pipeline**. |
| **Feature pipeline** | `alpha.py` | List of `(get_bar_fn, compute_fn)` pairs. Each pair: get the bar for a symbol, run the calculator, merge the returned dict into factor data. |
| **Data loading** | `utils.py` | Defines `CryptoFutureDepthData`, `CryptoFutureQuoteData`, and file/API loading helpers. |

Data flow:

1. **main.py** `on_data(data: Slice)` → framework calls `alpha_model.update(algorithm, data)`.
2. **alpha.py** `update()`: for each `CryptoFutureDepthData` (and optionally quote, etc.) in `data`, maps base symbol → `latest_depth[symbol]` (and `latest_quote[symbol]` if you add it).
3. On full-hour bars, `_refresh_all_factors()` runs. For each symbol it calls `_calculate_all_factors(algorithm, symbol)`.
4. **Inside _calculate_all_factors**: base factors (volatility, CO, disposition, etc.) are computed from OHLCV history; then the **custom feature pipeline** runs:
   - For each `(get_bar_fn, compute_fn)` in `_custom_feature_calculators`:
     - `bar = get_bar_fn(algorithm, symbol)` (e.g. depth or quote snapshot).
     - `custom_features.update(compute_fn(bar))`.
   - Resulting keys (e.g. `depth_obi_l5`, `depth_micro_price_divergence`) are merged into `factor_dict` and used in selection weight and filters.

---

## 2. Adding a New **Depth-Derived** Feature

You already have depth snapshots in `latest_depth[symbol]` and a calculator `compute_depth_features(depth_bar)` in `alpha.py`.

### Option A: Add a new key inside the existing depth calculator

1. **Edit `compute_depth_features()` in `alpha.py`**  
   - Input: `depth_bar` (optional `CryptoFutureDepthData`).  
   - Output: dict with keys `depth_obi_l5`, `depth_spread`, `depth_micro_price_divergence` (and any new keys you add).  
   - Add your new metric (e.g. `depth_near_obi`, `depth_imbalance_2pct`) in the same function and return it in the dict.

2. **Use the new key in the alpha model**  
   - In `_calculate_all_factors`, the pipeline already does:
     - `depth_obi_l5 = custom_features.get("depth_obi_l5", 0.0)`  
     - and similar for `depth_spread`, `depth_micro_price_divergence`.  
   - Add a line for your new key, e.g.:
     - `depth_near_obi = custom_features.get("depth_near_obi", 0.0)`  
   - Add the key to `factor_dict` so it is stored in `coin_data` and available in `update()` for filters/tags:
     - `factor_dict["depth_near_obi"] = round(depth_near_obi, 4)`  
   - If the feature should affect selection weight, add it to `_compute_selection_weight()` (parameters and formula) and pass it from `factor_dict` when building the weight.

3. **Optional: use in filters or tags**  
   - In `update()`, you can filter on the new key (e.g. skip if `d.get("depth_near_obi", 0) > 0.9`) or add it to `insight.Tag` for debugging.

### Option B: Add a second depth-specific calculator

If you want a separate function (e.g. for a different depth aggregation):

1. Implement a new function, e.g. `compute_depth_extra_features(depth_bar) -> Dict[str, float]`, returning keys like `depth_imbalance_10level`.
2. Add a getter that returns the same bar: e.g. `_get_depth_bar` is already there; you can reuse it.
3. Append to the pipeline in `AltcoinShortAlphaModel.__init__`:
   - `self._custom_feature_calculators.append((self._get_depth_bar, compute_depth_extra_features))`
4. In `_calculate_all_factors`, after the pipeline loop, read the new keys from `custom_features` and add them to `factor_dict` (and to `_compute_selection_weight` if needed).

---

## 3. Adding a New **Custom Data Type** (e.g. Quote, Open Interest)

To add a **new** kind of custom data (e.g. L1 quote, open interest, funding rate):

### Step 1: Define or reuse the custom data type in `utils.py`

- If it’s a new type: subclass `PythonData`, implement `get_source()` and `reader()` (and optionally `GetSource` / `Reader` for the C# bridge), and put the type in `utils.py` (or a dedicated module).
- If it already exists (e.g. `CryptoFutureQuoteData`), skip.

### Step 2: Subscribe in `main.py`

- For each trading symbol, call `self.add_data(YourCustomData, f"{ticker}.SUFFIX", Resolution.MINUTE)` (or the resolution you need).
- Store the mapping, e.g. `self.quote_symbols[symbol] = quote_security.symbol` (you already have this for quote).
- No need to “register” with the alpha model for subscription; the alpha model only needs to know how to get the bar per symbol (Step 4).

### Step 3: Capture the custom data in `alpha.py` in `update()`

- In `update()`, after the loop that fills `latest_depth`, add a similar loop for your type, e.g.:

```python
for kvp in data.get(CryptoFutureQuoteData):
    quote_symbol = kvp.key
    quote_data = kvp.value
    base_ticker = quote_symbol.value.replace(".QUOTE", "")
    for s in self.coin_data.keys():
        if s.value == base_ticker:
            self.latest_quote[s] = quote_data
            break
```

- Add the cache dict in `__init__`: `self.latest_quote: Dict[Symbol, CryptoFutureQuoteData] = {}`.

### Step 4: Add a getter and a calculator in `alpha.py`

- **Getter** (provides the bar for the pipeline):

```python
def _get_quote_bar(self, algorithm: QCAlgorithm, symbol: Symbol) -> Optional[CryptoFutureQuoteData]:
    return self.latest_quote.get(symbol)
```

- **Calculator** (bar → feature dict):

```python
def compute_quote_features(quote_bar: Optional[CryptoFutureQuoteData]) -> Dict[str, float]:
    out = {"quote_spread_bps": 0.0, "quote_imbalance": 0.0}  # example keys
    if quote_bar is None:
        return out
    try:
        bid = getattr(quote_bar, "bid", 0.0) or 0.0
        ask = getattr(quote_bar, "ask", 0.0) or 0.0
        if ask > 0:
            out["quote_spread_bps"] = (ask - bid) / ask * 10000.0
        # ...
    except Exception:
        pass
    return out
```

### Step 5: Register the calculator in the pipeline

In `AltcoinShortAlphaModel.__init__`:

```python
self._custom_feature_calculators.append((self._get_quote_bar, compute_quote_features))
```

### Step 6: Use the new keys in `_calculate_all_factors`

- After the pipeline loop, `custom_features` will contain the new keys. Add:
  - `quote_spread_bps = custom_features.get("quote_spread_bps", 0.0)` (and similarly for `quote_imbalance`).
- Add these to `factor_dict`.
- Optionally use them in `_compute_selection_weight()` and in `update()` for filters or tags.

### Step 7: Initialise new keys in `coin_data` when symbols are added

In `on_securities_changed` and in `register_symbol()`, when you create the initial `coin_data[symbol]` dict, add default values for the new keys, e.g.:

- `"quote_spread_bps": 0.0`, `"quote_imbalance": 0.0`

so that `d.get("quote_spread_bps", 0.0)` in `update()` never breaks.

---

## 4. Checklist for a New Feature

- [ ] **Data source**: Custom data type in `utils.py` and subscription in `main.py`.
- [ ] **Capture**: In `alpha.update()`, fill a `latest_*` dict from `data.get(YourType)`.
- [ ] **Getter**: `_get_*_bar(algorithm, symbol)` returning the bar for that symbol.
- [ ] **Calculator**: `compute_*_features(bar) -> Dict[str, float]` with sensible defaults when `bar is None`.
- [ ] **Pipeline**: `self._custom_feature_calculators.append((self._get_*_bar, compute_*_features))` in `__init__`.
- [ ] **Factor dict**: In `_calculate_all_factors`, read new keys from `custom_features` and add them to `factor_dict`.
- [ ] **Selection / filters**: If the feature should affect weight or filtering, wire it in `_compute_selection_weight()` and in `update()` (filters and/or `insight.Tag`).
- [ ] **Initial state**: Default values for new keys in `coin_data` in `on_securities_changed` and `register_symbol()`.

---

## 5. File Reference

| File | Relevant parts |
|------|----------------|
| **alpha.py** | `compute_depth_features`, `_custom_feature_calculators`, `_get_depth_bar`, `update()` (capture loop), `_calculate_all_factors()` (pipeline loop + factor_dict), `_compute_selection_weight()`, `on_securities_changed` / `register_symbol()` (initial coin_data). |
| **main.py** | `add_data(..., Resolution.MINUTE)`, `depth_symbols` / `quote_symbols`, `on_data` (no need to pass custom data explicitly; it’s in `data` and the alpha model reads it in `update()`). |
| **utils.py** | `CryptoFutureDepthData`, `CryptoFutureQuoteData`, and any new `PythonData` subclass with `get_source` / `reader`. |

This keeps the depth (and any other custom data) calculation **reusable and decoupled**: add a calculator and a getter, register one pair in the pipeline, and merge the returned keys into factor data and selection logic.
