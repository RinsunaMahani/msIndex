# msIndex — parametric maize risk engine

msIndex is a parametric drought-insurance engine for maize in South Africa's maize triangle. It combines
satellite and climate data into one index of yield shortfall, turns that index into payouts on a standard
policy ladder, prices the cover against the capital its tail risk needs, and seals every payout decision
in a tamper-evident ledger.

It ships as a multi-page Streamlit app, with a data pipeline that rebuilds the dataset from public sources.

## Why

Traditional crop cover needs a loss adjuster on every farm, which often costs more than the crop is worth.
The cheap alternative is a rainfall index, but it pays the wrong farms: it cannot see rain that falls at the
wrong time, or heat that damages a crop that did get its rain. msIndex reads the crop itself by adding
evaporative demand, heat stress and cropland greenness to rainfall.

## What it does

| Step | What happens | Code |
|---|---|---|
| **Observe** | For 12 districts across the Free State, Mpumalanga and North West, from the 2000/01 season onward, pull rainfall, temperature, evapotranspiration and cropland NDVI | [src/pipeline/](src/pipeline/) |
| **Learn** | A ridge regression, with its penalty chosen by leave-one-season-out CV, maps 10 seasonal features to the provincial yield anomaly against trend. It is trained on seasons before 2018/19 and tested on everything after | [src/model.py](src/model.py) |
| **Trigger** | A payout ladder turns the index into a share of the sum insured (attachment, exhaustion, franchise) | [src/pricing.py](src/pricing.py) |
| **Price** | Expected loss plus a cost-of-capital charge on the 1-in-200 tail, with an aggregate stop-loss reinsurance split. The tail comes from a zero-inflated beta fitted to the seasonal losses, or the worst observed season when there are too few losses to fit one | [src/pricing.py](src/pricing.py) |
| **Seal and settle** | Each decision is chained with HMAC-SHA256 and carries the policy terms it was made under, the amount it commits to, and the model and dataset fingerprints. A signed payment batch pays the sealed amounts, and only from a ledger that still verifies | [src/trigger.py](src/trigger.py), [src/ledger.py](src/ledger.py) |
| **Brief** | An LLM (Claude or Gemini) writes an underwriting briefing strictly from a fact sheet the engine computed. Without an API key it falls back to a deterministic template | [src/briefing.py](src/briefing.py) |

The app also exports a one-page PDF term sheet ([src/termsheet.py](src/termsheet.py)).

### App pages

- **Overview**: the headline case and its limits
- **Portfolio risk**: a 3D satellite map with the season's cropland greenness anomaly and a risk column per district (see below)
- **Basis risk**: msIndex against a rainfall-only trigger at a matched payout budget, on out-of-sample seasons
- **Pricing & capital**: loss distribution, VaR/TVaR, technical and gross premium, reinsurance. Priced by
  default from every season with a final yield; switch to the holdout seasons for an out-of-sample view
- **Audit & settlement**: the hash-chained ledger, a tamper demo, and the signed payment batch
- **Underwriting briefing**: LLM-written season summary and Q&A
- **Data & methods**, **Security**, **Business model**

Policy terms (sum insured, attachment, exhaustion, area) are set in the sidebar, and every page prices and
settles against them.

### The map

The map is built with deck.gl (through `st.pydeck_chart`) and needs no Google, Mapbox or Cesium key.

- **Map or Globe view.** The map view is full-resolution satellite imagery with place names and borders,
  tilted in 3D. Globe wraps the imagery around the Earth, so you can spin it and zoom from space down to the
  region.
- **Three imagery sources:**
  - Esri World Imagery, the default and the sharpest (non-commercial use only)
  - Sentinel-2 cloudless (EOX, CC BY 4.0)
  - NASA MODIS true colour on a date you pick in the selected season (15 February by default), served by NASA
    GIBS (public domain), so you see what the fields actually looked like that summer
- **3D terrain** from the AWS Terrain Tiles, with adjustable relief exaggeration. The greenness overlay,
  district columns and labels are draped onto the terrain surface. Imagery and relief load down to zoom 15
  (about 5 m per pixel with Esri), one level finer than the view needs, so they stay sharp on high-DPI
  screens. That means more tiles, so 3D takes a moment longer to fill in.
- **Fly to** any district, and **play all seasons** to watch a drought spread across the region. The camera
  stays where you put it while the seasons change.
- **Map options** also toggle place names and district labels, set the greenness opacity and camera tilt, and
  pick the NASA image date.

Navigation: drag to move, scroll or double-click to zoom, right-drag (or Ctrl-drag) to tilt and turn. The
arrow keys work too.

#### How the imagery is wired

- **Flat map:** the imagery is the basemap itself, defined by the style files in
  [static/basemaps/](static/basemaps/). Streamlit's deck.gl build draws raster `TileLayer`s as nothing, so
  this is the route that renders tiles at full resolution.
- **Globe:** Streamlit draws no basemap behind a globe, so it is textured with whole images: NASA Blue
  Marble for the Earth and a sharp image of the region.
- **3D terrain:** `TerrainLayer` loads its own texture tiles.
- **Place names and boundaries** come from Esri's reference overlay, which needs no API key. CARTO's label
  tiles now watermark requests that don't carry a paid key, so they aren't used.

Attribution for whatever is on screen is shown under the map. The hosted Esri and EOX tile services have their
own fair-use terms, so check them (or self-host the tiles) before using this commercially.

If the map looks out of date after pulling changes, hard-refresh the page (Ctrl+F5): browsers cache the
basemap style files.

## Data sources

| Signal | Source |
|---|---|
| Rainfall | CHIRPS v2.0 daily, 0.05° (UCSB Climate Hazards Center) |
| Max temperature, reference ET | ERA5-Land via the Open-Meteo archive API |
| Vegetation | MODIS MOD13Q1/MYD13Q1 v6.1 250 m NDVI via Microsoft Planetary Computer |
| Cropland mask | ESA WorldCover 2021 v200 (cropland class) |
| Yields | Crop Estimates Committee (SAGIS) and HarvestStat Africa v1.2, commercial white + yellow maize by province |
| Yield cross-check | DAFF Abstract of Agricultural Statistics 2018 |

Feature definitions and the build date are recorded in [data/metadata.json](data/metadata.json).

## Getting started

Requires Python 3.12+ (developed on 3.14).

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Copy [.env.example](.env.example) to `.env` and fill it in:

| Variable | Purpose |
|---|---|
| `MSINDEX_HMAC_KEY` | Signs the decision ledger and payment batches |
| `MSINDEX_PARCEL_SALT` | Salts parcel pseudonyms. It must be random, because parcel IDs follow a guessable pattern |
| `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | Optional. Enables the LLM briefing (Claude is used if both are set) |

Generate each secret with `python -c "import secrets; print(secrets.token_hex(32))"`. If they're missing,
the app still runs with random per-process values, but signatures and pseudonyms won't survive a restart.

Run the app from the project root, so Streamlit picks up `.streamlit/config.toml`. That file sets the theme
and switches on the static file serving the map needs for its basemap styles and greenness overlays:

```bash
streamlit run app.py
```

The repository includes a built dataset ([data/district_seasons.csv](data/district_seasons.csv)), so the
app works without running the pipeline.

## Refreshing the data

```bash
python -m src.pipeline                      # every complete season since 2000
python -m src.pipeline --seasons 2020 2025  # a specific planting-year range
```

The pipeline downloads yields, ERA5, CHIRPS and MODIS, builds the district-season table and renders the
map overlays into `static/overlays/`. Downloads are cached in `data/raw/`, so later runs fetch only what is
new. The Crop Estimates Committee publishes monthly from February to August, so re-run after each release
to pick up the latest yield forecast.

A "season" is named by its planting year: season 2024 is the 2024/25 crop.

## Tests

```bash
pytest
```

The unit tests cover the pipeline features, labels and splits, pricing, the tail fit, the ledger and the
briefing. The app tests render every page headlessly and check the map's layers. They are skipped if the
dataset hasn't been built.

## Project layout

```text
app.py                  Streamlit entry point and navigation
.env.example            Template for the secrets and API keys in .env
app_pages/              One file per page
src/
  app_data.py           Shared data loading, scoring context and sidebar policy terms
  model.py              Yield anomaly labels, train/test split, ridge model
  trigger.py            Risk bands, trigger evaluation, HMAC audit chain
  pricing.py            Policy terms, payout ladder, risk stats, premium, basis risk
  ledger.py             On-disk ledger, fingerprints, signed payment batches
  briefing.py           Claude / Gemini briefing with template fallback
  termsheet.py          PDF term sheet
  pipeline/             Data download and build (python -m src.pipeline)
data/
  district_seasons.csv  Built dataset (12 districts x 26 seasons)
  metadata.json         Sources, definitions, build date
  legacy/               Earlier synthetic dataset, kept for reference
static/
  basemaps/             Map styles: satellite imagery with or without place names
  overlays/             NDVI anomaly map overlays, one per season (built by the pipeline)
tests/                  pytest suite
```

## Limitations

- **Provincial, not farm-level.** Yields are only published per province, so the four districts in a
  province share one yield label. This is a provincial index.
- **Few independent loss events.** The out-of-sample period (2018/19 onward) has few losses, most of them in
  the 2023/24 drought, so the comparison against a rainfall index rests on a small sample.
- **Deep droughts are understated.** The model tends to under-predict the most severe shortfalls, and part of
  the losses in the holdout are covered less than half. The Basis risk page reports these numbers directly.
- **Current-season yields are preliminary** until the Crop Estimates Committee's final figures.
- The 1-in-200 capital figure comes from a distribution fitted to about 25 seasons, so it is modelled, not
  measured.
- **The greenness overlay is about 440 m per pixel**, so it looks blocky when zoomed in to field level. MODIS
  itself is 250 m. The NASA dated imagery is also 250 m, coarser than Esri or Sentinel-2.

This is a research and demonstration project, not a regulated insurance product.
