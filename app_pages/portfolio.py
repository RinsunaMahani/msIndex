import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src.app_data import (DATA_PATH, GREENNESS_SCALE, LEGEND_FILE, OVERLAY_DIR, OVERLAY_URL, RISK_COLORS, RISK_RGB,
                          download_row, get_context, policy_sidebar, season_label)
from src.model import SPLIT_CURRENT, SPLIT_TEST, SPLIT_TRAIN
from src.pricing import apply_policy
from src.trigger import LOSS_THRESHOLD, TEST_START_SEASON

# Why the imagery is wired the way it is: Streamlit's deck.gl build draws TileLayer tiles as GeoJSON, so a
# raster TileLayer shows nothing and Streamlit's grey default basemap shows through. Instead:
# - Map view: the imagery is the basemap itself, through a style file in static/basemaps/ that the bundled
#   mapbox-gl 1.x (open source, no token) renders tile by tile at full resolution.
# - Globe view: Streamlit draws no basemap, so the globe is textured with whole images (NASA Blue Marble for
#   the Earth plus a sharp image of the region), which deck.gl's BitmapLayer wraps onto the sphere.
# - 3D terrain: TerrainLayer loads its own texture tiles, so it takes the tile template directly.
# Every source here can be shown without a Google, Mapbox or Cesium licence.
BASEMAP_URL = "app/static/basemaps"
IMAGERY = {
    "Esri World Imagery": {
        "style": "esri",
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "max_zoom": 19,
        "credit": "Esri, Maxar, Earthstar Geographics (Esri terms: non-commercial use)",
    },
    "Sentinel-2 cloudless": {
        "style": "sentinel2",
        "tiles": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg",
        "max_zoom": 15,
        "credit": "Sentinel-2 cloudless 2016 by EOX IT Services GmbH (s2maps.eu), CC BY 4.0, contains modified "
                  "Copernicus Sentinel data 2016",
    },
    "NASA MODIS, on a date this season": {
        "style": "sentinel2",  # backdrop outside the dated image, which covers the maize triangle
        "tiles": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/"
                 "default/{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
        "max_zoom": 9,
        "credit": "NASA EOSDIS GIBS / Worldview, MODIS Terra corrected reflectance (public domain)",
    },
}
NASA = "NASA MODIS, on a date this season"
# Esri's reference overlay is made to sit on World Imagery and, unlike CARTO's label tiles, needs no API key
LABELS_CREDIT = "boundaries and place names: Esri, HERE, Garmin (Esri terms: non-commercial use)"
BLUE_MARBLE = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
               "&LAYERS=BlueMarble_ShadedRelief_Bathymetry&STYLES=&SRS=EPSG:4326&BBOX=-180,-90,180,90"
               "&WIDTH=2048&HEIGHT=1024&FORMAT=image/jpeg")
BLUE_MARBLE_CREDIT = "Earth: NASA Blue Marble via GIBS (public domain)"
REGION_BOUNDS = (23.5, -29.5, 31.5, -24.0)  # west, south, east, north: the maize triangle with context around it
REGION_IMAGE_WIDTH = {"Esri World Imagery": 3072}  # pixels; WMS services get the default
DEFAULT_IMAGE_WIDTH = 2048
TERRAIN_TILES = "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"
TERRAIN_CREDIT = "terrain: Terrain Tiles on AWS Open Data (SRTM, GMTED2010, ETOPO1)"
# TerrainLayer fetches its texture at the same zoom as the elevation tile, so this caps imagery sharpness as well
# as relief detail. The AWS terrain tiles go to 15; beyond that deck.gl over-zooms.
TERRAIN_MAX_ZOOM = 15
TERRAIN_ZOOM_OFFSET = 1    # one level finer than the view needs: sharp on high-DPI screens, 4x the tiles
TERRAIN_MESH_ERROR_M = 1.5  # finer triangulation (default 4 m), which matters once relief is exaggerated
DRAPE = [{"@@type": "TerrainExtension"}]  # lays a layer onto the 3D terrain surface instead of at sea level
# On a tilted map a label's pixel offset puts its glyphs over ground that is nearer the camera than its anchor,
# so the flat overlays underneath hide it. Skip the depth test for labels. Both spellings are given because
# deck.gl 9 reads luma.gl v9 names while its WebGL path still honours the classic ones; each ignores the other.
ALWAYS_ON_TOP = {"depthCompare": "always", "depthWriteEnabled": False, "depthTest": False, "depthMask": False}
COLUMN_RADIUS_M = 6000
SEVERITY_SCALE = 150000  # metres of column height per unit of predicted yield shortfall
REGION_VIEW = {"latitude": -26.85, "longitude": 27.5, "zoom": 6.3}
GLOBE_ZOOM = 3.2
EARTH_RADIUS_M = 6378137.0
# Google Earth-like handling: gliding pans, smooth wheel zoom, right-drag or Ctrl-drag to tilt and turn
CONTROLLER = {"inertia": 600, "scrollZoom": {"smooth": True, "speed": 0.02}, "dragRotate": True,
              "touchRotate": True, "keyboard": True, "doubleClickZoom": True}
NAVIGATION_HINT = ("Drag to move, scroll or double-click to zoom, right-drag (or Ctrl-drag) to tilt and turn; "
                   "arrow keys work too.")
SPACE_BACKGROUND = "radial-gradient(ellipse at center, #1b2a41 0%, #0a0f1a 60%, #03050a 100%)"

ctx = get_context(TEST_START_SEASON, os.path.getmtime(DATA_PATH))
terms = policy_sidebar()


def terrarium_decoder(exaggeration: float) -> dict:
    """Terrarium tiles encode height as R*256 + G + B/256 - 32768 metres; scaling every term exaggerates relief."""
    k = exaggeration
    return {"rScaler": 256 * k, "gScaler": k, "bScaler": k / 256, "offset": -32768 * k}


def season_start(season: int) -> date:
    return date(season, 10, 1)


def tile_template(name: str, image_date: date) -> str:
    return IMAGERY[name]["tiles"].replace("{date}", image_date.isoformat())


def _mercator(lon: float, lat: float) -> tuple[float, float]:
    return (EARTH_RADIUS_M * np.radians(lon),
            EARTH_RADIUS_M * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


def region_image(name: str, image_date: date, globe: bool) -> str:
    """One image of the region from the imagery service's image endpoint.

    On the globe the image must be in plain longitude/latitude; on the flat map, in Web Mercator. Either
    way the output rectangle matches the requested bounds exactly, so the image sits where it should.
    """
    west, south, east, north = REGION_BOUNDS
    if globe:
        srs, (x0, y0), (x1, y1) = 4326, (west, south), (east, north)
    else:
        srs, (x0, y0), (x1, y1) = 3857, _mercator(west, south), _mercator(east, north)
    width = REGION_IMAGE_WIDTH.get(name, DEFAULT_IMAGE_WIDTH)
    height = int(round(width * (y1 - y0) / (x1 - x0)))
    bbox = f"{x0:.6f},{y0:.6f},{x1:.6f},{y1:.6f}"
    if name == "Esri World Imagery":
        return ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
                f"?bbox={bbox}&bboxSR={srs}&imageSR={srs}&size={width},{height}&format=jpg&f=image")
    wms = f"SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&STYLES=&SRS=EPSG:{srs}&BBOX={bbox}&WIDTH={width}&HEIGHT={height}&FORMAT=image/jpeg"
    if name == NASA:
        return (f"https://gibs.earthdata.nasa.gov/wms/epsg{srs}/best/wms.cgi?{wms}"
                f"&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&TIME={image_date.isoformat()}")
    return f"https://tiles.maps.eox.at/wms?{wms}&LAYERS=s2cloudless"


def image_layer(layer_id: str, url: str, bounds, opacity: float = 1.0, **kwargs) -> pdk.Layer:
    layer = pdk.Layer("BitmapLayer", id=layer_id, data=None, bounds=list(bounds), opacity=opacity, **kwargs)
    # Assigned after construction so pydeck keeps the URL instead of base64-embedding a local file
    layer.image = url
    return layer


def basemap_style(opts: dict) -> str:
    suffix = "_labels" if opts["place_names"] else ""
    return f"{BASEMAP_URL}/{IMAGERY[opts['imagery']]['style']}{suffix}.json"


def build_deck(season_df: pd.DataFrame, season: int, layer: str, opts: dict) -> pdk.Deck:
    """Satellite imagery (flat, draped over exaggerated terrain, or wrapped on a globe), this season's
    greenness image, and a labelled 3D column per district whose height is the predicted yield shortfall."""
    globe = opts["view"] == "Globe"
    terrain = opts["terrain_3d"] and not globe
    drape = {"extensions": DRAPE} if terrain else {}

    layers = []
    if globe:
        layers.append(image_layer("earth", BLUE_MARBLE, (-180, -90, 180, 90)))
        layers.append(image_layer("region-imagery", region_image(opts["imagery"], opts["image_date"], True),
                                  REGION_BOUNDS))
    elif terrain:
        layers.append(pdk.Layer("TerrainLayer", id="terrain",
                                elevation_decoder=terrarium_decoder(opts["exaggeration"]),
                                elevation_data=TERRAIN_TILES, texture=tile_template(opts["imagery"], opts["image_date"]),
                                operation="'terrain+draw'",  # quoted, or pydeck sends it as an expression
                                max_zoom=min(TERRAIN_MAX_ZOOM, IMAGERY[opts["imagery"]]["max_zoom"]),
                                zoom_offset=TERRAIN_ZOOM_OFFSET, mesh_max_error=TERRAIN_MESH_ERROR_M))
    elif opts["imagery"] == NASA:
        layers.append(image_layer("region-imagery", region_image(NASA, opts["image_date"], False), REGION_BOUNDS))

    if layer == "Vegetation greenness" and ctx.overlay_meta and os.path.exists(f"{OVERLAY_DIR}/greenness_{season}.png"):
        layers.append(image_layer("greenness", f"{OVERLAY_URL}/greenness_{season}.png", ctx.overlay_meta["bounds"],
                                  opacity=opts["overlay_opacity"], **drape))

    cols = season_df.assign(
        elevation=lambda d: (-d["pred_yield_anomaly"]).clip(lower=0) * SEVERITY_SCALE + 3000,
        color=lambda d: d["Risk band"].map(lambda b: RISK_RGB[b] + [210]),
        pred_pct=lambda d: (d["pred_yield_anomaly"] * 100).round(1),
        payout_pct=lambda d: (d["payout_share"] * 100).round(1),
        ndvi_txt=lambda d: d["ndvi_anomaly"].round(3),
        rain_txt=lambda d: d["rain_mm"].round(0),
    )
    records = cols.to_dict("records")
    layers.append(pdk.Layer("ColumnLayer", id="districts", data=records, get_position=["lon", "lat"],
                            get_elevation="elevation", get_fill_color="color", radius=COLUMN_RADIUS_M,
                            extruded=True, pickable=True, auto_highlight=True, elevation_scale=1, **drape))
    if opts["labels"]:
        layers.append(pdk.Layer("TextLayer", id="district-names", data=records, get_position=["lon", "lat"],
                                get_text="district", get_size=13, get_color=[255, 255, 255],
                                get_pixel_offset=[0, 18], font_weight=600, billboard=True,
                                font_settings={"sdf": True}, outline_width=3, outline_color=[20, 20, 20, 230],
                                parameters=ALWAYS_ON_TOP, **drape))

    focus = season_df[season_df["district"] == opts["focus"]]
    if len(focus):
        target = {"latitude": float(focus["lat"].iloc[0]), "longitude": float(focus["lon"].iloc[0]),
                  "zoom": 9.0 if globe else 10.0}
    else:
        target = {**REGION_VIEW, "zoom": GLOBE_ZOOM if globe else REGION_VIEW["zoom"]}

    return pdk.Deck(
        layers=layers,
        views=[pdk.View(type="_GlobeView" if globe else "MapView", controller=CONTROLLER)],
        initial_view_state=pdk.ViewState(**target, pitch=0 if globe else opts["tilt"], bearing=0,
                                         transition_duration=1500),
        # On the flat map the satellite imagery is the basemap; the globe has no basemap, only the layers above
        map_provider=None, map_style=None if globe else basemap_style(opts),
        tooltip={"html": "<b>{district}</b> ({province})<br/>Predicted yield: {pred_pct}%<br/>"
                         "Payout: {payout_pct}% of sum insured<br/>Greenness anomaly: {ndvi_txt}<br/>"
                         "Rain Oct–Mar: {rain_txt} mm<br/>Risk band: {Risk band}",
                 "style": {"backgroundColor": "rgba(30,30,30,0.9)", "color": "white", "fontSize": "12px"}},
    )


def map_credits(opts: dict) -> str:
    globe = opts["view"] == "Globe"
    credits = [IMAGERY[opts["imagery"]]["credit"]]
    if opts["imagery"] == NASA:
        credits[0] += f", image date {opts['image_date']:%d %b %Y}"
        if not globe:
            credits.append(IMAGERY["Sentinel-2 cloudless"]["credit"] + " (outside the dated image)")
    if globe:
        credits.append(BLUE_MARBLE_CREDIT)
    elif opts["place_names"]:
        credits.append(LABELS_CREDIT)
    if opts["terrain_3d"] and not globe:
        credits.append(TERRAIN_CREDIT)
    credits.append("greenness overlay: MODIS NDVI via Microsoft Planetary Computer, ESA WorldCover cropland mask")
    return "Imagery: " + " · ".join(credits)


def render_season_view(slider_season, animate, map_layer, opts: dict):
    """Everything that depends on the selected season, run as a fragment so "Play all seasons"
    can advance without rerunning the rest of the page."""
    if animate:
        st.session_state.anim_index = (st.session_state.get("anim_index", ctx.seasons.index(slider_season)) + 1) % len(ctx.seasons)
        season = ctx.seasons[st.session_state.anim_index]
    else:
        st.session_state.anim_index = ctx.seasons.index(slider_season)
        season = slider_season

    season_df = apply_policy(ctx.df[ctx.df["season"] == season], terms)
    prev_df = ctx.df[ctx.df["season"] == season - 1]
    split = season_df["Data split"].iloc[0]

    def delta(cur, prev, fmt):
        return None if prev is None else format(cur - prev, fmt)

    has_prev = not prev_df.empty
    cur_ndvi, cur_rain = season_df["ndvi_anomaly"].mean(), season_df["rain_mm"].mean()
    cur_loss = int((season_df["pred_yield_anomaly"] <= LOSS_THRESHOLD).sum())
    prev_ndvi = prev_df["ndvi_anomaly"].mean() if has_prev else None
    prev_rain = prev_df["rain_mm"].mean() if has_prev else None
    prev_loss = int((prev_df["pred_yield_anomaly"] <= LOSS_THRESHOLD).sum()) if has_prev else None

    if animate:
        st.caption(f":material/play_circle: Playing — {season_label(season)}")

    with st.container(horizontal=True):
        st.metric("Mean greenness anomaly (NDVI)", f"{cur_ndvi:+.3f}", delta(cur_ndvi, prev_ndvi, "+.3f"), border=True,
                  help="Peak cropland NDVI against each district's 2000–2014 norm. Negative = browner than usual.")
        st.metric("Mean Oct–Mar rainfall", f"{cur_rain:,.0f} mm", delta(cur_rain, prev_rain, "+,.0f"), border=True)
        st.metric("Districts with predicted loss", f"{cur_loss} / {len(season_df)}",
                  None if prev_loss is None else f"{cur_loss - prev_loss:+d}", delta_color="inverse", border=True,
                  help=f"Predicted yield at least {abs(LOSS_THRESHOLD):.0%} below the provincial trend.")
        st.metric("Payout this season", f"{season_df['payout_share'].mean():.1%} of cover", border=True,
                  help="Average payout across districts under the policy terms in the sidebar.")

    if split == SPLIT_TEST:
        st.badge("Test season: out-of-sample predictions", icon=":material/check_circle:", color="green")
    elif split == SPLIT_TRAIN:
        st.badge("Training season: in-sample predictions, accuracy is overstated", icon=":material/warning:", color="orange")
    else:
        st.badge("Current season: forecast from a model trained on all final seasons; reported yield is preliminary",
                 icon=":material/schedule:", color="blue")

    # The dated image keeps its place in the season as seasons change, so playback compares like with like
    opts = {**opts, "image_date": season_start(season) + timedelta(days=opts["image_offset"])}
    # The season is deliberately not in the key: the chart updates in place, so the camera stays where the
    # user put it while seasons change or play. Switching view or imagery remounts it.
    view_key = "_".join(str(opts[k]) for k in ("view", "imagery", "place_names", "terrain_3d"))

    if opts["view"] == "Globe":
        # The globe has no basemap behind it, so give it space rather than the page's sand colour
        st.html(f"<style>.st-key-map_frame {{background: {SPACE_BACKGROUND};}}</style>")
    with st.container(border=True, key="map_frame"):
        st.pydeck_chart(build_deck(season_df, season, map_layer, opts), height=640, key=f"risk_map_{view_key}")
    legend_col, text_col = st.columns([1, 4], vertical_alignment="center")
    if map_layer == "Vegetation greenness" and ctx.overlay_meta:
        if os.path.exists(LEGEND_FILE):
            legend_col.image(LEGEND_FILE, width="stretch")
            legend_col.caption(f"−{ctx.overlay_meta['anomaly_limit']:.0%}  ·  normal  ·  +{ctx.overlay_meta['anomaly_limit']:.0%}")
        text_col.caption(
            f"Cropland greenness in {season_label(season)} against the "
            f"{ctx.overlay_meta['baseline_seasons'][0]}–{ctx.overlay_meta['baseline_seasons'][1]} normal, from "
            "MODIS imagery: brown = drought-stressed, green = healthier than usual. Only cropland is painted. "
            f"Columns are districts, height = predicted loss severity. {NAVIGATION_HINT}"
        )
    else:
        text_col.caption(f"Columns are districts: height = predicted loss severity, colour = risk band. {NAVIGATION_HINT}")
    st.caption(map_credits(opts))

    col1, col2 = st.columns([3, 2])
    with col1:
        with st.container(border=True):
            st.markdown("**District risk summary**")
            table = season_df[["district", "province", "rain_mm", "water_balance_mm", "ndvi_anomaly",
                               "pred_yield_anomaly", "payout_share", "Risk band"]].sort_values("pred_yield_anomaly")
            st.dataframe(
                table, hide_index=True, height=250,
                column_config={
                    "district": "District", "province": "Province",
                    "rain_mm": st.column_config.NumberColumn("Rain (mm)", format="%.0f"),
                    "water_balance_mm": st.column_config.NumberColumn("Water bal. (mm)", format="%.0f"),
                    "ndvi_anomaly": st.column_config.NumberColumn("Greenness", format="%+.3f"),
                    "pred_yield_anomaly": st.column_config.NumberColumn("Pred. yield", format="percent"),
                    "payout_share": st.column_config.NumberColumn("Payout", format="percent"),
                },
            )
            download_row(table, f"msindex_districts_{season}.csv")
    with col2:
        with st.container(border=True):
            st.markdown("**Provincial yield: predicted vs reported** (commercial maize)")
            prov = (season_df.groupby("province")
                    .agg(pred=("pred_yield_tha", "mean"), reported=("yield_tha", "first"),
                         trend=("yield_trend", "first"), status=("yield_status", "first")).reset_index())
            st.dataframe(prov, hide_index=True, column_config={
                "province": "Province",
                "pred": st.column_config.NumberColumn("Predicted t/ha", format="%.2f"),
                "reported": st.column_config.NumberColumn("Reported t/ha", format="%.2f"),
                "trend": st.column_config.NumberColumn("Trend t/ha", format="%.2f"),
                "status": "Yield status"})

    # Regional greenness through time, so the selected season is seen in context
    ndvi_lim = float(ctx.df["ndvi_anomaly"].abs().quantile(0.98))
    trend_df = ctx.df.groupby("season", as_index=False)[["ndvi_anomaly", "yield_anomaly"]].mean()
    trend_fig = px.bar(trend_df, x="season", y="ndvi_anomaly", color="ndvi_anomaly",
                       color_continuous_scale=GREENNESS_SCALE, range_color=[-ndvi_lim, ndvi_lim],
                       labels={"season": "Season (planting year)", "ndvi_anomaly": "Mean greenness anomaly",
                               "yield_anomaly": "Reported yield anomaly"},
                       hover_data={"yield_anomaly": ":+.1%"}, height=260)
    trend_fig.update_traces(marker_line_color=["#111" if s == season else "rgba(0,0,0,0)" for s in trend_df["season"]],
                            marker_line_width=2.5)
    trend_fig.add_vline(x=TEST_START_SEASON - 0.5, line_dash="dash", line_color="grey",
                        annotation_text="Test period →", annotation_position="top right")
    trend_fig.add_vline(x=ctx.current_season - 0.5, line_dash="dot", line_color="grey",
                        annotation_text="Current →", annotation_position="bottom right")
    trend_fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False, xaxis=dict(dtick=2))
    with st.container(border=True):
        st.markdown("**Regional greenness by season** (selected season outlined)")
        st.plotly_chart(trend_fig, key=f"greenness_trend_{season}")


st.subheader("Maize triangle vegetation & production risk", anchor=False)

ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 1], vertical_alignment="bottom")
slider_season = ctrl1.select_slider("Agricultural season", options=ctx.seasons, value=ctx.current_season,
                                    format_func=season_label, key="season", bind="query-params")
map_layer = ctrl2.segmented_control("Map layer", options=["Vegetation greenness", "Predicted yield risk"],
                                    default="Vegetation greenness", required=True, key="layer", bind="query-params")
animate = ctrl3.toggle("Play all seasons",
                       help="Step through every season so droughts are visible as brown spreading across the region.")
v1, v2, v3 = st.columns([2, 3, 2], vertical_alignment="bottom")
view = v1.segmented_control("View", options=["Map", "Globe"], default="Map", required=True, key="view",
                            help="Globe wraps the imagery around the Earth, Google Earth style: drag to spin, scroll "
                                 "to zoom from space down to the region.")
imagery = v2.selectbox("Satellite imagery", list(IMAGERY), key="imagery",
                       help="All three can be shown without a Google or Mapbox licence. Esri is the sharpest; "
                            "Sentinel-2 cloudless is openly licensed; the NASA option is the actual satellite picture "
                            "on a date you choose in the selected season.")
focus = v3.selectbox("Fly to", ["Whole region"] + sorted(ctx.df["district"].unique()), key="focus")

with st.expander("Map options", icon=":material/tune:"):
    o1, o2, o3 = st.columns(3)
    terrain_3d = o1.toggle("3D terrain", value=False, disabled=view == "Globe",
                           help="Drape the imagery, greenness and columns over real elevation. The highveld is "
                                "flat, so relief is exaggerated to make it readable.")
    exaggeration = o1.slider("Relief exaggeration", 1.0, 8.0, 4.0, step=0.5, disabled=not terrain_3d or view == "Globe")
    overlay_opacity = o2.slider("Greenness opacity", 0.0, 1.0, 0.7, step=0.05)
    place_names = o2.toggle("Place names", value=True, disabled=view == "Globe",
                            help="Town and country names from OpenStreetMap, drawn over the imagery.")
    labels = o2.toggle("District labels", value=True)
    tilt = o3.slider("Camera tilt", 0, 60, 45, step=5, disabled=view == "Globe", help="0 = straight down.")
    image_offset = 137  # 15 February: peak canopy, before harvest
    if imagery == NASA:
        image_offset = o3.select_slider(
            "Image date", options=list(range(0, 212)), value=image_offset, key="image_offset",
            format_func=lambda d: f"{season_start(2001) + timedelta(days=d):%d %b}",
            help="Day in the season (1 Oct to 30 Apr). The summer rainy season is often cloudy: move a few days "
                 "either way for a clear view.")

map_opts = {"view": view, "imagery": imagery, "focus": focus, "terrain_3d": terrain_3d and view != "Globe",
            "exaggeration": exaggeration, "overlay_opacity": overlay_opacity, "labels": labels, "tilt": tilt,
            "image_offset": image_offset, "place_names": place_names}

# Auto-advancing rerun of just this section while "Play all seasons" is on
st.fragment(run_every=2.0 if animate else None)(render_season_view)(slider_season, animate, map_layer, map_opts)

if ctx.metadata:
    with st.expander("Data sources & definitions", icon=":material/database:"):
        st.caption(f"Dataset built {ctx.metadata['built_at']} · seasons {season_label(ctx.metadata['seasons'][0])}–"
                   f"{season_label(ctx.metadata['seasons'][1])} · refresh with `python -m src.pipeline`")
        st.table(pd.DataFrame([(k.replace("_", " ").capitalize(), v) for k, v in ctx.metadata["sources"].items()],
                              columns=["Input", "Source"]).set_index("Input"))
        st.table(pd.DataFrame(list(ctx.metadata["definitions"].items()),
                              columns=["Field", "Definition"]).set_index("Field"))
