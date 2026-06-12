# Umpire Analytics Dashboard
# Author: Christian Lafter
# Date: 6/12/26
#
# This dashboard is designed for Trackman based pitching data umpire evaluation. It ingests
# Trackman pitch-tracking CSV data — either uploaded directly or pulled from the
# Trackman FTP server — and produces accuracy metrics, a filterable missed-call
# table, and an interactive strike zone chart. A static PNG report can also be
# exported as a ZIP for distribution to supervisors or umpires.
import io
import datetime
import zipfile
from ftplib import FTP, error_perm
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


# Sidebar messages are stored in session state with an expiry timestamp so they
# disappear automatically after a configurable number of seconds rather than
# persisting across every rerun.

def set_sidebar_message(text, kind="success", duration_seconds=20):
    """Write a timed status message into session state for the sidebar to render."""
    expiry = datetime.datetime.now().timestamp() + float(duration_seconds)
    st.session_state["sidebar_message"] = {
        "text": text,
        "kind": kind,
        "expiry": expiry,
    }

def render_sidebar_message():
    """Read the pending sidebar message from session state and display it if not expired."""
    msg = st.session_state.get("sidebar_message")
    if msg:
        if msg.get("expiry", 0) > datetime.datetime.now().timestamp():
            kind = msg.get("kind", "info")
            text = msg.get("text", "")
            if kind == "success":
                st.sidebar.success(text)
            elif kind == "warning":
                st.sidebar.warning(text)
            elif kind == "error":
                st.sidebar.error(text)
            else:
                st.sidebar.info(text)
        else:
            del st.session_state["sidebar_message"]

render_sidebar_message()

st.set_page_config(page_title="Umpire Dashboard", layout="wide")
st.title("Umpire Analytics Dashboard")
st.markdown("Upload a Trackman CSV to begin:")


# Date detection helpers. Trackman files use several different date column names
# depending on the export version, so we try the most common ones in priority order.

def _get_date_column(df):
    """Return the name of the date column in df, or None if not found."""
    exact_date = [col for col in df.columns if col.lower() == "date"]
    if exact_date:
        return exact_date[0]
    exact_date_like = [col for col in df.columns if col.lower() in {"game date", "gamedate", "date game", "date"}]
    if exact_date_like:
        return exact_date_like[0]
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None

def _to_game_date(df, column_name):
    """Parse column_name to datetime and write a date-only GameDate column. Returns True on success."""
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
    if df[column_name].notna().any():
        df["GameDate"] = df[column_name].dt.date
        return True
    return False


# FTP helpers. The Trackman server stores one CSV per game named with team codes
# and a date stamp. These functions handle connecting, walking the directory tree,
# and downloading individual files.

def _connect_ftp(host, port, username, password):
    """Open and authenticate an FTP connection. Raises on failure."""
    try:
        ftp = FTP(timeout=30)
        ftp.connect(host, int(port))
        ftp.login(username, password)
        return ftp
    except Exception as e:
        raise Exception(f"FTP connection failed: {e}")

def _retrieve_path(ftp, path):
    """Download a single file from the FTP server into an in-memory BytesIO buffer."""
    bio = io.BytesIO()
    ftp.retrbinary(f"RETR {path}", bio.write)
    bio.seek(0)
    return bio

def _download_latest_csv_from_directory(ftp, remote_dir):
    """
    Given a directory path on the FTP server, return the most recently modified
    CSV file as a BytesIO buffer. Uses the MDTM command to compare modification times.
    """
    entries = ftp.nlst(remote_dir)
    csv_files = [f for f in entries if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in FTP directory: {remote_dir}")
    def _get_mdtm(filepath):
        try:
            resp = ftp.sendcmd(f"MDTM {filepath}")
            return datetime.datetime.strptime(resp[4:], "%Y%m%d%H%M%S")
        except Exception:
            return datetime.datetime.min
    latest = max(csv_files, key=_get_mdtm)
    return _retrieve_path(ftp, latest)

def _download_csv_from_ftp(host, port, username, password, remote_path):
    """
    Download a CSV from the FTP server by path. If remote_path is a directory,
    the most recently modified CSV inside it is returned instead.
    Always closes the FTP connection in the finally block.
    """
    ftp = _connect_ftp(host, port, username, password)
    try:
        if remote_path.lower().endswith(".csv"):
            return _retrieve_path(ftp, remote_path)
        return _download_latest_csv_from_directory(ftp, remote_path)
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()

def _is_ftp_directory(ftp, path):
    """
    Test whether a given FTP path is a directory by attempting to cd into it.
    Used as a fallback when the server does not support MLSD.
    """
    current = ftp.pwd()
    try:
        ftp.cwd(path)
        ftp.cwd(current)
        return True
    except error_perm:
        return False
    except Exception:
        return False

def _scan_ftp_for_csv(ftp, remote_path=".", max_depth=8):
    """
    Recursively walk the FTP directory tree starting at remote_path and return
    a sorted list of all .csv file paths found. Tries MLSD first (modern servers)
    and falls back to NLST if MLSD is not supported. Depth is capped at max_depth
    to avoid runaway traversal on large servers.
    """
    csv_files = []
    def _walk(path, depth):
        if depth > max_depth:
            return
        try:
            entries = list(ftp.mlsd(path))
        except Exception:
            try:
                entries = [(name, None) for name in ftp.nlst(path)]
            except Exception:
                return
        for name, facts in entries:
            if name in (".", ".."):
                continue
            candidate = name if path in (".", "/") else f"{path.rstrip('/')}/{name}"
            if facts is not None:
                is_dir = facts.get("type") == "dir"
            else:
                is_dir = _is_ftp_directory(ftp, candidate)
            if is_dir:
                _walk(candidate, depth + 1)
            elif candidate.lower().endswith(".csv"):
                csv_files.append(candidate)
    _walk(remote_path or ".", 0)
    return sorted(set(csv_files))

def _filter_ftp_candidates(paths):
    """
    From the full list of CSV paths on the server, return only those that are
    likely to be pitch data files. Files matching known non-pitch keywords
    (e.g. playerpositioning) are excluded first, then paths containing common
    pitch-data keywords are preferred. If no keyword matches are found the full
    filtered list is returned so the user still sees something.
    """
    keywords = ["trackman", "baseball", "game", "umpire", "pitch", "report", "events", "stats"]
    excluded = "playerpositioning"
    filtered = [path for path in paths if excluded not in path.lower()]
    matches = [path for path in filtered if any(keyword in path.lower() for keyword in keywords)]
    return sorted(matches or filtered)

def _extract_date_from_filename(filename):
    """
    Try to pull a game date out of a Trackman filename. Trackman typically
    embeds the date as YYYYMMDD or YYYY-MM-DD. Returns a date object or None.
    """
    import re
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            pass
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if match:
        try:
            return datetime.datetime.strptime(match.group(0), "%Y-%m-%d").date()
        except ValueError:
            pass
    return None

def _organize_files_by_date(file_list):
    """Group FTP file paths into a dict keyed by date, for the date-picker UI."""
    files_by_date = {}
    for filepath in file_list:
        date = _extract_date_from_filename(filepath)
        if date:
            if date not in files_by_date:
                files_by_date[date] = []
            files_by_date[date].append(filepath)
    return files_by_date


# Trackman exports team identifiers as short codes. This map translates them to
# the full Frontier League team names used throughout the UI and in report filenames.

team_map = {
    "EVA_OTT": "Evansville Otters",
    "FLO_Y'A": "Florence Y'alls",
    "LAK_ERI24": "Lake Erie Crushers",
    "SCH_BOO": "Schaumburg Boomers",
    "WIN_CIT29": "Windy City ThunderBolts",
    "TRO_AIG": "Trois-Rivières Aigles",
    "MIS_MUD": "Mississippi Mud Monsters",
    "GAT_GRI": "Gateway Grizzlies",
    "OTT_TIT": "Ottawa Titans",
    "QUE_CAP": "Québec Capitales",
    "SUS_COU1": "Sussex County Miners",
    "DOW_EAS1": "Down East Bird Dawgs",
    "NEW_YOR13": "New York Boulders",
    "NEW_ENG23": "Brockton Rox",
    "WAS_WIL3": "Washington Wild Things",
    "TRI_VAL": "Tri-City ValleyCats",
    "NEW_JER6": "New Jersey Jackals",
    "JOL_SLA": "Joliet Slammers",
}

def clean_team(team):
    """Translate a Trackman team code to a full name. Passes through unknown codes unchanged."""
    if pd.isna(team):
        return team
    return team_map.get(str(team), str(team))

def _extract_team_codes_from_filename(filename):
    """
    Scan a filename for Trackman team codes and return them in order of appearance.
    The first code found is treated as the away team, the second as the home team,
    which matches the convention used in Trackman's file naming scheme.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    search_text = basename.lower()
    team_map_lower = {key.lower(): key for key in team_map.keys()}
    found = []
    for code_lower, original_key in team_map_lower.items():
        idx = search_text.find(code_lower)
        if idx >= 0:
            found.append((idx, original_key))
    found.sort(key=lambda item: item[0])
    return [original_key for _, original_key in found]

def _get_ftp_display_label(remote_path):
    """
    Build a human-readable label for an FTP file, e.g. "Evansville Otters @ Gateway Grizzlies".
    Appends "(Unverified)" if the filename contains that word. Falls back to the
    bare filename if no known team codes are found.
    """
    basename = os.path.splitext(os.path.basename(remote_path))[0]
    lower_name = basename.lower()
    unverified = "unverified" in lower_name
    team_codes = _extract_team_codes_from_filename(basename)
    if len(team_codes) >= 2:
        away = clean_team(team_codes[0])
        home = clean_team(team_codes[1])
        label = f"{away} @ {home}"
    elif len(team_codes) == 1:
        away = clean_team(team_codes[0])
        label = f"{away} @ Unknown"
    else:
        label = basename
    if unverified:
        label += " (Unverified)"
    return label

def _build_ftp_display_labels(paths):
    """Return a list of display labels for a list of FTP paths, one label per path."""
    return [_get_ftp_display_label(path) for path in paths]


# Strike zone geometry. All measurements are in feet to match Trackman's coordinate system.
# The zone edges represent the rulebook strike zone; the buffer box defines the close-call
# region by extending one ball radius plus 1.69 inches outward from each edge.

ZONE_LEFT   = -0.83   # left edge of the plate in feet
ZONE_RIGHT  =  0.83   # right edge of the plate in feet
ZONE_BOTTOM =  1.5    # lower boundary of the strike zone in feet above the ground
ZONE_TOP    =  3.5    # upper boundary of the strike zone in feet above the ground

BASEBALL_DIAMETER = 2.94 / 12   # official baseball diameter converted to feet
BASEBALL_RADIUS   = BASEBALL_DIAMETER / 2
CC_BUFFER         = 1.69 / 12   # close-call buffer width in feet (1.69 inches)

# The outer dotted box drawn on the chart. It extends one ball radius plus the
# close-call buffer beyond each zone edge, representing the outermost position
# at which a pitch can still be considered a close call.
BUFFER_LEFT   = (ZONE_LEFT   - BASEBALL_RADIUS) - CC_BUFFER
BUFFER_RIGHT  = (ZONE_RIGHT  + BASEBALL_RADIUS) + CC_BUFFER
BUFFER_BOTTOM = (ZONE_BOTTOM - BASEBALL_RADIUS) - CC_BUFFER
BUFFER_TOP    = (ZONE_TOP    + BASEBALL_RADIUS) + CC_BUFFER

# The inner cutoff box. A pitch whose center is more than CC_BUFFER inside every
# zone edge is considered "deep in the zone" and is not a close call regardless of
# the overlap fraction check below.
INNER_LEFT   = ZONE_LEFT   + CC_BUFFER
INNER_RIGHT  = ZONE_RIGHT  - CC_BUFFER
INNER_BOTTOM = ZONE_BOTTOM + CC_BUFFER
INNER_TOP    = ZONE_TOP    - CC_BUFFER

# A pitch is a close call only if the fraction of the ball's area that overlaps the
# strike zone is no greater than this threshold. At 1/2, the ball's center can sit
# right on the zone edge and still qualify; pitches with the majority of the ball
# inside the zone are not considered close calls.
CC_MAX_ZONE_OVERLAP = 1 / 2


def _circle_rect_overlap_frac(cx, cy, r, rx0, ry0, rx1, ry1):
    """
    Estimate what fraction of each circle's area overlaps a rectangle.

    cx, cy are 1-D numpy arrays of pitch center coordinates. r is the ball radius
    (scalar). The rectangle spans (rx0, ry0) to (rx1, ry1).

    The method samples 512 evenly-spaced points around the circle's circumference
    and checks what fraction of them fall inside the rectangle. For convex shapes
    this is a good approximation of the true area fraction. Analytic shortcuts are
    applied for circles that are entirely inside or entirely outside the rectangle.

    Returns a 1-D float array of overlap fractions in [0, 1].
    """
    x = np.asarray(cx, dtype=float)
    y = np.asarray(cy, dtype=float)

    # Distance from each center to the nearest point on the rectangle boundary.
    # If this distance exceeds r the circle does not touch the rectangle at all.
    clamp_x = np.clip(x, rx0, rx1)
    clamp_y = np.clip(y, ry0, ry1)
    dist_sq = (x - clamp_x) ** 2 + (y - clamp_y) ** 2
    entirely_outside = dist_sq > r ** 2

    # A circle is fully inside the rectangle if its bounding box fits within it.
    entirely_inside  = (x - r >= rx0) & (x + r <= rx1) & (y - r >= ry0) & (y + r <= ry1)

    N = 512
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Boundary sample points for all pitches at once: shape (n_pitches, N)
    px = x[:, None] + r * np.cos(theta)[None, :]
    py = y[:, None] + r * np.sin(theta)[None, :]
    inside = (px >= rx0) & (px <= rx1) & (py >= ry0) & (py <= ry1)
    frac = inside.mean(axis=1)

    frac = np.where(entirely_outside, 0.0, frac)
    frac = np.where(entirely_inside,  1.0, frac)
    return frac


def _close_call_mask(x_arr, y_arr):
    """
    Return a boolean array marking which pitches qualify as close calls.

    A pitch is a close call when both of the following are true:
      1. The ball center falls within the buffer band — i.e. inside the outer
         dotted box but not so far inside the zone that it is past the inner cutoff.
      2. No more than half (1/2) of the ball's cross-sectional area overlaps the
         strike zone rectangle.

    The overlap check is the key rule: it allows a pitch that barely clips the zone
    corner to be a close call, while a pitch whose center is well inside the zone
    (so most of the ball is in) is excluded. The near-edge pre-filter means the
    expensive overlap calculation only runs on candidate pitches, not every row.
    """
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)

    within_buffer = (
        (x >= BUFFER_LEFT)  & (x <= BUFFER_RIGHT)
        & (y >= BUFFER_BOTTOM) & (y <= BUFFER_TOP)
    )
    deep_inside = (
        (x > INNER_LEFT) & (x < INNER_RIGHT)
        & (y > INNER_BOTTOM) & (y < INNER_TOP)
    )
    near_edge = within_buffer & ~deep_inside

    if not near_edge.any():
        return near_edge

    candidates = np.where(near_edge)[0]
    frac = _circle_rect_overlap_frac(
        x[candidates], y[candidates], BASEBALL_RADIUS,
        ZONE_LEFT, ZONE_BOTTOM, ZONE_RIGHT, ZONE_TOP,
    )

    result = near_edge.copy()
    result[candidates] = near_edge[candidates] & (frac <= CC_MAX_ZONE_OVERLAP)
    return result


def _compute_metrics(game_df):
    """
    Compute the standard umpire accuracy statistics for a dataframe of called pitches.
    Returns a plain dict so callers can use the values in both the chart annotation
    and the PNG report without duplicating the logic.
    """
    total     = len(game_df)
    missed    = int(game_df["MissedCall"].sum())
    correct   = total - missed
    accuracy  = (correct / total * 100) if total > 0 else 0.0

    is_cs  = game_df["PitchCall"] == "StrikeCalled"
    is_cb  = game_df["PitchCall"] == "BallCalled"
    in_zone = game_df["InZone"]
    is_cc  = game_df["CloseCall"]

    # Called strike accuracy: of all called strikes, how many were actually in the zone.
    cs_acc = ((is_cs & in_zone).sum() / is_cs.sum() * 100) if is_cs.sum() > 0 else 0.0
    # Called ball accuracy: of all called balls, how many were actually outside the zone.
    cb_acc = ((is_cb & ~in_zone).sum() / is_cb.sum() * 100) if is_cb.sum() > 0 else 0.0
    # Ball-called-as-strike percentage: fraction of called balls that were in the zone.
    ball_str = ((is_cb & in_zone).sum() / is_cb.sum() * 100) if is_cb.sum() > 0 else 0.0
    cc_count = int(is_cc.sum())
    cc_correct_pct = ((is_cc & ~game_df["MissedCall"]).sum() / cc_count * 100) if cc_count > 0 else 0.0

    return dict(
        total=total, missed=missed, accuracy=accuracy,
        cs_acc=cs_acc, cb_acc=cb_acc, ball_str=ball_str,
        cc_count=cc_count, cc_correct_pct=cc_correct_pct,
    )


def _build_game_png(game_df, away_team, home_team, game_date):
    """
    Render one game's strike zone chart to PNG bytes using matplotlib.

    The output matches the visual layout of the interactive Plotly chart: green dots
    for correct calls, red for missed calls, with "CC" text labels on close calls.
    The solid rectangle is the rulebook zone; the dotted rectangle is the close-call
    buffer. Stats are printed in the title. The figure is sized to preserve a 1:1
    aspect ratio so the zone does not appear distorted.
    """
    m = _compute_metrics(game_df)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor("white")

    correct_df = game_df[~game_df["MissedCall"]]
    missed_df  = game_df[game_df["MissedCall"]]
    close_df   = game_df[game_df["CloseCall"]]

    ax.scatter(
        correct_df["PlateLocSide"], correct_df["PlateLocHeight"],
        color="#2ca02c", edgecolors="black", linewidths=0.6,
        s=200, alpha=0.85, zorder=3, label="Correct Call",
    )
    ax.scatter(
        missed_df["PlateLocSide"], missed_df["PlateLocHeight"],
        color="#d62728", edgecolors="black", linewidths=0.6,
        s=200, alpha=0.85, zorder=4, label="Missed Call",
    )

    for _, row in close_df.iterrows():
        ax.text(
            row["PlateLocSide"], row["PlateLocHeight"],
            "CC", ha="center", va="center",
            fontsize=6.5, color="white", fontweight="bold", zorder=5,
        )

    zone_rect = Rectangle(
        (ZONE_LEFT, ZONE_BOTTOM), ZONE_RIGHT - ZONE_LEFT, ZONE_TOP - ZONE_BOTTOM,
        linewidth=3, edgecolor="black", facecolor="none", zorder=6,
    )
    ax.add_patch(zone_rect)

    buf_rect = Rectangle(
        (BUFFER_LEFT, BUFFER_BOTTOM), BUFFER_RIGHT - BUFFER_LEFT, BUFFER_TOP - BUFFER_BOTTOM,
        linewidth=1.5, edgecolor="black", facecolor="none",
        linestyle="dotted", zorder=6,
    )
    ax.add_patch(buf_rect)

    all_x = game_df["PlateLocSide"].dropna()
    all_y = game_df["PlateLocHeight"].dropna()
    x_pad, y_pad = 0.35, 0.35

    max_x = max(abs(all_x.min()), abs(all_x.max())) + x_pad if len(all_x) else 2.0
    y_lo  = max(0.0, float(all_y.min()) - y_pad) if len(all_y) else 0.0
    y_hi  = float(all_y.max()) + y_pad if len(all_y) else 4.5

    x_lo, x_hi = -max_x, max_x
    x_span = x_hi - x_lo
    y_span = y_hi - y_lo

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")

    # Scale the figure dimensions to match the data aspect ratio so the physical
    # zone rectangle looks correct when printed or shared as an image.
    base_size, margin_in = 8.0, 2.0
    if x_span >= y_span:
        fig_w = base_size + margin_in
        fig_h = base_size * (y_span / x_span) + margin_in
    else:
        fig_h = base_size + margin_in
        fig_w = base_size * (x_span / y_span) + margin_in
    fig.set_size_inches(fig_w, fig_h)

    ax.set_xlabel("Horizontal Location", fontsize=12)
    ax.set_ylabel("Vertical Location", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)

    stats_line = (
        f"Total Called: {m['total']} | Missed: {m['missed']} | "
        f"Accuracy: {m['accuracy']:.1f}% | CS%: {m['cs_acc']:.1f}% | "
        f"CB%: {m['cb_acc']:.1f}% | Ball to Strike%: {m['ball_str']:.1f}% | "
        f"CC: {m['cc_count']} | CC Correct%: {m['cc_correct_pct']:.1f}%"
    )
    ax.set_title(stats_line, fontsize=9, pad=8)

    date_str = game_date.strftime("%B %d, %Y") if hasattr(game_date, "strftime") else str(game_date)
    fig.suptitle(
        f"{away_team}  @  {home_team}\n{date_str}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _safe_filename_part(name):
    """Strip characters that are not safe to use in file or archive entry names."""
    import re
    return re.sub(r"[^\w\s\-]", "", str(name)).strip().replace(" ", "_")


def _infer_games_from_df(day_df):
    """
    Partition a single-day dataframe into individual games based on which teams
    appear as both pitcher and batter teams. Returns a list of
    (away_team, home_team, game_slice_df) tuples. Used only in the upload path
    where a single CSV may contain multiple games from the same day.
    """
    games = []
    if "PitcherTeam" not in day_df.columns or "BatterTeam" not in day_df.columns:
        return [("Unknown", "Unknown", day_df)]

    teams_as_pitcher = set(day_df["PitcherTeam"].dropna().unique())
    teams_as_batter  = set(day_df["BatterTeam"].dropna().unique())
    all_teams        = teams_as_pitcher | teams_as_batter

    seen_pairs = set()
    matchups   = []
    for t1 in sorted(all_teams):
        for t2 in sorted(all_teams):
            if t1 == t2:
                continue
            pair = tuple(sorted([t1, t2]))
            if pair in seen_pairs:
                continue
            t1_pitches_to_t2 = ((day_df["PitcherTeam"] == t1) & (day_df["BatterTeam"] == t2)).any()
            t2_pitches_to_t1 = ((day_df["PitcherTeam"] == t2) & (day_df["BatterTeam"] == t1)).any()
            if t1_pitches_to_t2 and t2_pitches_to_t1:
                seen_pairs.add(pair)
                matchups.append((t1, t2))

    if not matchups:
        return [("Unknown", "Unknown", day_df)]

    for away, home in matchups:
        mask = (
            ((day_df["PitcherTeam"] == away) & (day_df["BatterTeam"] == home))
            | ((day_df["PitcherTeam"] == home) & (day_df["BatterTeam"] == away))
        )
        game_slice = day_df[mask]
        if len(game_slice) > 0:
            games.append((away, home, game_slice))

    return games if games else [("Unknown", "Unknown", day_df)]


def generate_day_zip(day_df, selected_date):
    """
    Generate one PNG per game in day_df and bundle them into a ZIP archive.
    Used by the upload code path. Games are inferred from the team columns
    via _infer_games_from_df rather than from filenames.
    Returns (zip_bytes, list_of_filenames).
    """
    games = _infer_games_from_df(day_df)

    if hasattr(selected_date, "month"):
        date_prefix = f"{selected_date.month}_{selected_date.day}_{selected_date.year}"
    else:
        d = datetime.datetime.strptime(str(selected_date), "%Y-%m-%d").date()
        date_prefix = f"{d.month}_{d.day}_{d.year}"

    zip_buf   = io.BytesIO()
    filenames = []

    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for away, home, game_slice in games:
            away_safe = _safe_filename_part(away)
            home_safe = _safe_filename_part(home)
            fname     = f"{date_prefix} {away_safe}@{home_safe}.png"
            png_bytes = _build_game_png(game_slice, away, home, selected_date)
            zf.writestr(fname, png_bytes)
            filenames.append(fname)

    zip_buf.seek(0)
    return zip_buf.getvalue(), filenames


def _read_trackman_csv(bio):
    """
    Read a Trackman CSV from a BytesIO buffer. Some Trackman exports include a
    metadata header row before the column names, so if PitchNo is not found in
    the first read we retry with skiprows=1.
    """
    try:
        tmp = pd.read_csv(bio)
        if "PitchNo" not in tmp.columns:
            bio.seek(0)
            tmp = pd.read_csv(bio, skiprows=1)
        return tmp
    except Exception:
        return None


def _process_raw_df(raw_df):
    """
    Apply the full processing pipeline to a raw Trackman dataframe. This mirrors
    the main-body pipeline so FTP-path PNG reports use identical logic to the
    interactive chart. Steps: filter to called pitches, filter to Frontier League,
    clean names and team codes, compute InZone, CloseCall, and MissedCall columns.
    """
    raw_df = raw_df[raw_df["PitchCall"].str.contains("Called", case=False, na=False)].copy()

    if "League" in raw_df.columns:
        raw_df = raw_df[raw_df["League"].str.upper().str.strip() == "FRONT"]
    if len(raw_df) == 0:
        return raw_df

    if "Pitcher" in raw_df.columns:
        raw_df["Pitcher"] = raw_df["Pitcher"].apply(clean_name)
    if "Batter" in raw_df.columns:
        raw_df["Batter"] = raw_df["Batter"].apply(clean_name)
    if "PitcherTeam" in raw_df.columns:
        raw_df["PitcherTeam"] = raw_df["PitcherTeam"].apply(clean_team)
    if "BatterTeam" in raw_df.columns:
        raw_df["BatterTeam"] = raw_df["BatterTeam"].apply(clean_team)

    _x = raw_df["PlateLocSide"].astype(float)
    _y = raw_df["PlateLocHeight"].astype(float)

    # InZone: the ball physically overlaps the strike zone rectangle. We expand
    # each zone edge by one ball radius so the check is based on the ball touching
    # the zone, not the center of the ball being inside it.
    raw_df["InZone"] = (
        (_x >= (ZONE_LEFT   - BASEBALL_RADIUS)) & (_x <= (ZONE_RIGHT  + BASEBALL_RADIUS))
        & (_y >= (ZONE_BOTTOM - BASEBALL_RADIUS)) & (_y <= (ZONE_TOP    + BASEBALL_RADIUS))
    )

    # CloseCall: near a zone edge with no more than half the ball inside the zone.
    raw_df["CloseCall"] = _close_call_mask(_x.values, _y.values)

    # MissedCall: an in-zone pitch called ball, or an out-of-zone pitch called strike.
    raw_df["MissedCall"] = (
        (raw_df["InZone"] & (raw_df["PitchCall"] == "BallCalled"))
        | (~raw_df["InZone"] & (raw_df["PitchCall"] == "StrikeCalled"))
    )
    return raw_df


# Data source selector. The user can either upload a CSV directly or connect to the
# Trackman FTP server and download a file by date. Both paths ultimately produce the
# same uploaded_file BytesIO object that the rest of the script consumes.

data_source = st.sidebar.radio("Data Source", ["Upload CSV", "FTP Download"])
uploaded_file = None

if data_source == "FTP Download":
    with st.sidebar.expander("FTP Settings", expanded=True):
        ftp_host = st.text_input(
            "FTP Host", value="ftp.trackmanbaseball.com", disabled=True, key="ftp_host"
        )
        ftp_port = st.number_input(
            "Port", value=21, min_value=1, max_value=65535, key="ftp_port"
        )
        ftp_username = "Frontier League"
        ftp_password = "VHq3wDSmJr"

        st.text_input("Username", value=ftp_username, disabled=True, key="ftp_username_display")
        st.text_input("Password", value="********", type="password", disabled=True, key="ftp_password_display")

        ftp_scan_base = st.text_input(
            "FTP scan start directory", value="",
            help="Leave blank to scan the FTP root for CSV files.", key="ftp_scan_base"
        )
        exclude_unverified = st.checkbox(
            "Exclude 'Unverified' files", value=False, key="ftp_exclude_unverified",
            help="When checked, files with 'unverified' in the filename will be ignored during the scan."
        )

        if st.button("Scan FTP for CSV files", key="ftp_scan_button"):
            try:
                ftp = _connect_ftp(ftp_host, ftp_port, ftp_username, ftp_password)
                try:
                    scan_path = ftp_scan_base or "."
                    csv_files = _scan_ftp_for_csv(ftp, scan_path)
                    excluded_unverified_count = 0
                    if exclude_unverified:
                        pre_count = len(csv_files)
                        csv_files = [f for f in csv_files if "unverified" not in f.lower()]
                        excluded_unverified_count = pre_count - len(csv_files)
                    year_counts = {}
                    undated = []
                    for p in csv_files:
                        d = _extract_date_from_filename(p)
                        if d:
                            year_counts[d.year] = year_counts.get(d.year, 0) + 1
                        else:
                            undated.append(p)
                    st.session_state["ftp_scan_diag"] = {
                        "total": len(csv_files),
                        "year_counts": year_counts,
                        "undated": undated,
                        "excluded_unverified": excluded_unverified_count,
                    }
                    st.session_state["ftp_scan_results"] = _filter_ftp_candidates(csv_files)
                    if not st.session_state["ftp_scan_results"]:
                        st.sidebar.warning("No CSV files found during scan.")
                    else:
                        set_sidebar_message(
                            f"Found {len(st.session_state['ftp_scan_results'])} candidate CSV files.",
                            kind="success", duration_seconds=20,
                        )
                finally:
                    try:
                        ftp.quit()
                    except Exception:
                        try:
                            ftp.close()
                        except Exception:
                            pass
            except Exception as exc:
                st.sidebar.error(f"FTP scan failed: {exc}")

    scan_results = st.session_state.get("ftp_scan_results", [])

    if scan_results:
        files_by_date = _organize_files_by_date(scan_results)

        if files_by_date:
            with st.sidebar.expander("Filter by Date", expanded=False):
                available_dates = sorted(files_by_date.keys())
                selected_date = st.date_input(
                    "Select Game Date",
                    value=available_dates[0] if available_dates else datetime.date.today(),
                    min_value=available_dates[0] if available_dates else datetime.date.today(),
                    max_value=available_dates[-1] if available_dates else datetime.date.today(),
                    key="ftp_date_select",
                )
                files_for_date = files_by_date.get(selected_date, [])

                if files_for_date:
                    if len(files_for_date) == 1:
                        ftp_remote_path = files_for_date[0]
                        st.info(f"Selected: {_get_ftp_display_label(ftp_remote_path)}")
                    else:
                        st.write(f"**{len(files_for_date)} files found for this date**")
                        display_labels = _build_ftp_display_labels(files_for_date)
                        selected_label = st.selectbox("Choose file", display_labels, key="ftp_csv_select")
                        selected_index = display_labels.index(selected_label)
                        ftp_remote_path = files_for_date[selected_index]

                    if st.button("Download selected FTP CSV", key="ftp_download_button"):
                        try:
                            ftp_file = _download_csv_from_ftp(
                                ftp_host, ftp_port, ftp_username, ftp_password, ftp_remote_path
                            )
                            st.session_state["ftp_file_bytes"]  = ftp_file.getvalue()
                            st.session_state["ftp_selected_date"] = selected_date
                            set_sidebar_message("FTP CSV downloaded successfully.", kind="success", duration_seconds=20)
                        except Exception as exc:
                            st.error(f"FTP download failed: {exc}")

                    st.divider()
                    st.markdown("**PNG Report**")
                    st.caption(
                        f"Generates one strike zone PNG per game on "
                        f"{selected_date.strftime('%B %d, %Y')} and downloads as a ZIP."
                    )
                    if st.button("Download Day PNG Report", key="png_report_button_ftp"):
                        st.session_state["png_report_requested"] = True
                        st.session_state["png_report_date"]      = selected_date
                        st.session_state["png_report_all_files"] = files_for_date

                else:
                    st.warning(f"No files found for {selected_date.strftime('%Y-%m-%d')}. Try another date.")
                    ftp_remote_path = None
        else:
            with st.sidebar.expander("Filter by Date", expanded=False):
                st.warning("No dates found in filenames. Showing all files.")
                display_labels = _build_ftp_display_labels(scan_results)
                selected_label = st.selectbox("Choose file", display_labels, key="ftp_csv_select")
                selected_index = display_labels.index(selected_label)
                ftp_remote_path = scan_results[selected_index]

                if st.button("Download selected FTP CSV", key="ftp_download_button"):
                    try:
                        ftp_file = _download_csv_from_ftp(
                            ftp_host, ftp_port, ftp_username, ftp_password, ftp_remote_path
                        )
                        st.session_state["ftp_file_bytes"] = ftp_file.getvalue()
                        set_sidebar_message("FTP CSV downloaded successfully.", kind="success", duration_seconds=20)
                    except Exception as exc:
                        st.error(f"FTP download failed: {exc}")
    else:
        ftp_remote_path = st.sidebar.text_input(
            "Remote file or directory", value="", key="ftp_remote_path"
        )
        st.sidebar.info("Scan the FTP server to list CSVs automatically, or enter a file/directory path manually.")

    if st.session_state.get("ftp_file_bytes") is not None:
        uploaded_file = io.BytesIO(st.session_state["ftp_file_bytes"])
        set_sidebar_message("FTP CSV loaded from session.", kind="success", duration_seconds=20)
    else:
        st.sidebar.info("After choosing a file, click Download selected FTP CSV.")

else:
    st.sidebar.header("Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file or download one from FTP.")
    st.stop()


# CSV loading. Trackman sometimes prepends a metadata row above the column headers,
# so if PitchNo is missing on the first read we retry with skiprows=1.

try:
    df = pd.read_csv(uploaded_file)
    if "PitchNo" not in df.columns:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, skiprows=1)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Only rows where the umpire made a call (StrikeCalled or BallCalled) are relevant.
df = df[df["PitchCall"].str.contains("Called", case=False, na=False)]

# This dashboard is built for Frontier League data. The League column value for
# Frontier League games is "FRONT". Other leagues (e.g. MLB, affiliated ball)
# present in a multi-league export are dropped here.
if "League" in df.columns:
    df = df[df["League"].str.upper().str.strip() == "FRONT"]
    if len(df) == 0:
        st.error("No rows with League == 'FRONT' found in this file.")
        st.stop()

def clean_name(name):
    """Convert "Last, First" format to "First Last" for display."""
    if pd.isna(name):
        return name
    if "," in str(name):
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return name

if "Pitcher" in df.columns:
    df["Pitcher"] = df["Pitcher"].apply(clean_name)
else:
    st.error(f"Missing column: Pitcher. Available: {list(df.columns)}")
    st.stop()

if "Batter" in df.columns:
    df["Batter"] = df["Batter"].apply(clean_name)
else:
    st.error(f"Missing column: Batter. Available: {list(df.columns)}")
    st.stop()

df["PitcherTeam"] = df["PitcherTeam"].apply(clean_team)
df["BatterTeam"]  = df["BatterTeam"].apply(clean_team)

# Build a "Balls-Strikes" count string for each pitch so the count filter works.
if "Balls" in df.columns and "Strikes" in df.columns:
    df["Count"] = (
        df["Balls"].fillna(0).astype(int).astype(str)
        + "-"
        + df["Strikes"].fillna(0).astype(int).astype(str)
    )
else:
    df["Count"] = np.nan


# Strike zone classification. All three derived columns use the constants and
# helper functions defined earlier in this file.

_x = df["PlateLocSide"].astype(float)
_y = df["PlateLocHeight"].astype(float)

# InZone: the ball physically overlaps the zone rectangle (center within one
# ball radius of any zone edge counts as touching).
df["InZone"] = (
    (_x >= (ZONE_LEFT   - BASEBALL_RADIUS))
    & (_x <= (ZONE_RIGHT  + BASEBALL_RADIUS))
    & (_y >= (ZONE_BOTTOM - BASEBALL_RADIUS))
    & (_y <= (ZONE_TOP    + BASEBALL_RADIUS))
)

# CloseCall: the pitch is near a zone edge and no more than half the ball
# overlaps the zone. See _close_call_mask for the full definition.
df["CloseCall"] = _close_call_mask(_x.values, _y.values)

# MissedCall: a ball-in-zone called ball, or a ball-out-of-zone called strike.
df["MissedCall"] = (
    (df["InZone"] & (df["PitchCall"] == "BallCalled"))
    | (~df["InZone"] & (df["PitchCall"] == "StrikeCalled"))
)
df["CallResult"] = np.where(df["MissedCall"], "Missed Call", "Correct Call")


# Date detection and filtering. If the CSV contains a recognisable date column we
# expose a date picker so the user can restrict the view to a single day or range.
# For the FTP path the date was already selected before download so no picker is shown.

date_column = _get_date_column(df)
upload_selected_date = None

if date_column is not None and _to_game_date(df, date_column):
    game_dates = sorted(df["GameDate"].dropna().unique())
    if game_dates:
        ftp_loaded = (data_source == "FTP Download" and st.session_state.get("ftp_file_bytes") is not None)
        if not ftp_loaded:
            with st.sidebar.expander("Filter by Date", expanded=False):
                if len(game_dates) == 1:
                    upload_selected_date = st.date_input(
                        "Select Game Date",
                        value=game_dates[0],
                        min_value=game_dates[0],
                        max_value=game_dates[0],
                    )
                    df = df[df["GameDate"] == upload_selected_date]
                else:
                    selected_range = st.date_input(
                        "Select Game Date Range",
                        value=(game_dates[0], game_dates[-1]),
                        min_value=game_dates[0],
                        max_value=game_dates[-1],
                    )
                    if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
                        start_date, end_date = selected_range
                        df = df[(df["GameDate"] >= start_date) & (df["GameDate"] <= end_date)]
                        upload_selected_date = start_date
                    else:
                        df = df[df["GameDate"] == selected_range]
                        upload_selected_date = selected_range

                st.sidebar.caption(f"Detected date field: {date_column}")

                if upload_selected_date is not None and len(df) > 0:
                    st.divider()
                    st.markdown("**PNG Report**")
                    st.caption(
                        f"Generates one strike zone PNG per game on "
                        f"{upload_selected_date.strftime('%B %d, %Y')} and downloads as a ZIP."
                    )
                    if st.button("Download Day PNG Report", key="png_report_button_upload"):
                        st.session_state["png_report_requested"] = True
                        st.session_state["png_report_date"]      = upload_selected_date
                        st.session_state.pop("png_report_all_files", None)
else:
    st.sidebar.caption("No date field detected. Upload an FTP/CSV with a date column.")


# PNG report generation. This block runs when the user clicks either of the
# "Download Day PNG Report" buttons. The FTP path downloads each game file
# individually and processes it; the upload path partitions the already-loaded
# dataframe by team matchup.

if st.session_state.get("png_report_requested"):
    report_date       = st.session_state.get("png_report_date")
    all_files_for_day = st.session_state.get("png_report_all_files")

    if all_files_for_day and data_source == "FTP Download":
        zip_buf   = io.BytesIO()
        filenames = []
        date_prefix = (
            f"{report_date.month}_{report_date.day}_{report_date.year}"
            if hasattr(report_date, "month")
            else str(report_date)
        )

        progress = st.progress(0, text="Downloading game files...")

        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for i, fpath in enumerate(all_files_for_day):
                progress.progress(
                    int((i / len(all_files_for_day)) * 90),
                    text=f"Downloading {i + 1}/{len(all_files_for_day)}: {os.path.basename(fpath)}",
                )
                try:
                    ftp_bio = _download_csv_from_ftp(
                        ftp_host, ftp_port, ftp_username, ftp_password, fpath
                    )
                except Exception as exc:
                    st.warning(f"Could not download {fpath}: {exc}")
                    continue

                raw_df = _read_trackman_csv(ftp_bio)
                if raw_df is None or len(raw_df) == 0:
                    st.warning(f"Empty or unreadable file: {fpath}")
                    continue

                game_df = _process_raw_df(raw_df)
                if len(game_df) == 0:
                    st.warning(f"No called pitches found in: {fpath}")
                    continue

                # Prefer team names extracted from the filename since that is the most
                # reliable source. Fall back to the team columns in the data if needed.
                team_codes = _extract_team_codes_from_filename(fpath)
                if len(team_codes) >= 2:
                    away = clean_team(team_codes[0])
                    home = clean_team(team_codes[1])
                elif "PitcherTeam" in game_df.columns and "BatterTeam" in game_df.columns:
                    teams = list(game_df["PitcherTeam"].dropna().unique())
                    away  = teams[0] if len(teams) > 0 else "Unknown"
                    home  = teams[1] if len(teams) > 1 else "Unknown"
                else:
                    away, home = "Unknown", "Unknown"

                away_safe = _safe_filename_part(away)
                home_safe = _safe_filename_part(home)
                fname     = f"{date_prefix} {away_safe}@{home_safe}.png"

                # Guard against duplicate archive entry names when two FTP files
                # resolve to the same matchup label (e.g. a verified and unverified
                # copy of the same game).
                existing = {info.filename for info in zf.infolist()}
                counter  = 2
                while fname in existing:
                    fname   = f"{date_prefix} {away_safe}@{home_safe}_{counter}.png"
                    counter += 1

                png_bytes = _build_game_png(game_df, away, home, report_date)
                zf.writestr(fname, png_bytes)
                filenames.append(fname)

        progress.progress(100, text="Packaging ZIP...")
        zip_buf.seek(0)
        progress.empty()

        date_label = (
            report_date.strftime("%m_%d_%Y")
            if hasattr(report_date, "strftime") else str(report_date)
        )
        if filenames:
            st.download_button(
                label=f"Download ZIP - {len(filenames)} game(s) - {date_label}",
                data=zip_buf.getvalue(),
                file_name=f"umpire_report_{date_label}.zip",
                mime="application/zip",
                key="png_zip_download",
            )
            st.success(f"Ready. {len(filenames)} PNG(s) packaged in ZIP:")
            for fn in filenames:
                st.caption(f"  {fn}")
        else:
            st.error("No valid game files could be processed for the selected date.")

    else:
        report_df = df.copy()
        if "GameDate" in report_df.columns and report_date is not None:
            report_df = report_df[report_df["GameDate"] == report_date]

        if len(report_df) == 0:
            st.warning("No data available for the selected date to generate a PNG report.")
        else:
            with st.spinner("Generating PNG report..."):
                try:
                    zip_bytes, filenames = generate_day_zip(report_df, report_date)
                    date_label = (
                        report_date.strftime("%m_%d_%Y")
                        if hasattr(report_date, "strftime") else str(report_date)
                    )
                    st.download_button(
                        label=f"Download ZIP - {len(filenames)} game(s) - {date_label}",
                        data=zip_bytes,
                        file_name=f"umpire_report_{date_label}.zip",
                        mime="application/zip",
                        key="png_zip_download",
                    )
                    st.success(f"Ready. {len(filenames)} PNG(s) packaged in ZIP:")
                    for fn in filenames:
                        st.caption(f"  {fn}")
                except Exception as exc:
                    st.error(f"PNG report generation failed: {exc}")

    st.session_state["png_report_requested"] = False
    st.session_state.pop("png_report_all_files", None)


# Sidebar filters. Filters are applied in cascading stages so that each dropdown
# only shows options that are valid given what has already been selected above it.
# Stage order: pitcher team -> inning -> pitcher -> batter -> pitch type -> count.

with st.sidebar.expander("Filters", expanded=True):
    pitcher_teams = st.multiselect("Pitcher Team", sorted(df["PitcherTeam"].dropna().unique()))
    innings       = st.multiselect("Inning", sorted(df["Inning"].dropna().unique()))
    close_call_only = st.toggle("Only Close Calls", value=False)

    filter_mask = pd.Series([True] * len(df), index=df.index)
    if pitcher_teams:
        filter_mask &= df["PitcherTeam"].isin(pitcher_teams)
    if innings:
        filter_mask &= df["Inning"].isin(innings)
    filtered_stage1 = df[filter_mask]

    pitchers     = st.multiselect("Pitcher", sorted(filtered_stage1["Pitcher"].dropna().unique()))
    filter_mask2 = pd.Series([True] * len(filtered_stage1), index=filtered_stage1.index)
    if pitchers:
        filter_mask2 &= filtered_stage1["Pitcher"].isin(pitchers)
    filtered_stage2 = filtered_stage1[filter_mask2]

    batters      = st.multiselect("Batter", sorted(filtered_stage2["Batter"].dropna().unique()))
    filter_mask3 = pd.Series([True] * len(filtered_stage2), index=filtered_stage2.index)
    if batters:
        filter_mask3 &= filtered_stage2["Batter"].isin(batters)
    filtered_stage3 = filtered_stage2[filter_mask3]

    pitch_types = st.multiselect("Pitch Type", sorted(filtered_stage3["TaggedPitchType"].dropna().unique()))
    counts      = st.multiselect("Count", sorted(filtered_stage3["Count"].dropna().unique()))

    filter_mask_final = pd.Series([True] * len(filtered_stage3), index=filtered_stage3.index)
    if pitch_types:
        filter_mask_final &= filtered_stage3["TaggedPitchType"].isin(pitch_types)
    if close_call_only:
        filter_mask_final &= filtered_stage3["CloseCall"]
    if counts:
        filter_mask_final &= filtered_stage3["Count"].isin(counts)

    filtered = filtered_stage3[filter_mask_final]


# Chart axis scaling. We base the scale on the pre-pitch-type-filtered data so that
# zooming in via pitch type or count does not cause the axes to jump around.

scale_source = filtered_stage3 if (filtered_stage3 is not None and len(filtered_stage3) > 0) else filtered
if scale_source is not None and len(scale_source) > 0:
    x_min, x_max = scale_source["PlateLocSide"].min(), scale_source["PlateLocSide"].max()
    y_min, y_max = scale_source["PlateLocHeight"].min(), scale_source["PlateLocHeight"].max()
else:
    x_min, x_max = ZONE_LEFT, ZONE_RIGHT
    y_min, y_max = ZONE_BOTTOM, ZONE_TOP

x_padding, y_padding = 0.35, 0.35
max_x   = max(abs(x_min), abs(x_max)) + x_padding
x_range = [-max_x, max_x]
y_range = [max(0, y_min - y_padding), y_max + y_padding]


# Summary metrics. These are computed from the filtered dataset so they update
# live as the user adjusts the sidebar filters.

st.subheader("Umpire Stats")

total_pitches    = len(filtered)
missed_calls     = filtered["MissedCall"].sum()
correct_calls    = total_pitches - missed_calls
overall_accuracy = (correct_calls / total_pitches * 100) if total_pitches > 0 else 0

is_called_strike = filtered["PitchCall"] == "StrikeCalled"
is_called_ball   = filtered["PitchCall"] == "BallCalled"
is_in_zone       = filtered["InZone"]
is_close_call    = filtered["CloseCall"]

called_strikes = filtered[is_called_strike]
called_balls   = filtered[is_called_ball]

# Called strike accuracy: percentage of called strikes that were correctly in the zone.
called_strike_accuracy = (
    (is_called_strike & is_in_zone).sum() / is_called_strike.sum() * 100
    if is_called_strike.sum() > 0 else 0
)
# Called ball accuracy: percentage of called balls that were correctly out of the zone.
called_ball_accuracy = (
    (is_called_ball & ~is_in_zone).sum() / is_called_ball.sum() * 100
    if is_called_ball.sum() > 0 else 0
)
# Ball-to-strike rate: percentage of called balls that were actually in the zone.
called_ball_strike_pct = (
    (is_called_ball & is_in_zone).sum() / is_called_ball.sum() * 100
    if is_called_ball.sum() > 0 else 0
)
close_calls = filtered[is_close_call]
# CC correct rate: of all close calls, what percentage did the umpire get right.
close_call_accuracy = (
    (is_close_call & ~filtered["MissedCall"]).sum() / is_close_call.sum() * 100
    if is_close_call.sum() > 0 else 0
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Called Pitches", total_pitches)
col2.metric("Missed Calls", int(missed_calls))
col3.metric("Overall Accuracy %", f"{overall_accuracy:.1f}%")

col4, col5, col6, col7, col8 = st.columns(5)
col4.metric("Called Strike Accuracy",           f"{called_strike_accuracy:.1f}%")
col5.metric("Called Ball Accuracy",             f"{called_ball_accuracy:.1f}%")
col6.metric("% Called Balls That Were Strikes", f"{called_ball_strike_pct:.1f}%")
col7.metric("Close Calls",                      len(close_calls))
col8.metric("CC Correct %",                     f"{close_call_accuracy:.1f}%")

with st.expander("Umpire Accuracy by Count", expanded=False):
    if "Count" in filtered.columns:
        count_summary = (
            filtered.groupby("Count")
            .agg(Total=("PitchCall", "count"), Missed=("MissedCall", "sum"))
            .assign(Accuracy=lambda d: (1 - d["Missed"] / d["Total"]) * 100)
            .reset_index()
            .sort_values(by="Total", ascending=False)
        )
        st.dataframe(count_summary, use_container_width=True)
    else:
        st.write("Count data not available in this dataset.")


# Missed calls table. Clicking a row highlights that pitch on the strike zone chart
# below, which is useful for reviewing specific calls in context.

st.subheader("Missed Calls")

missed_calls_df = filtered[filtered["MissedCall"]].copy()
display_columns = ["Pitcher", "PitcherTeam", "Batter", "BatterTeam"]
if "GameDate" in missed_calls_df.columns:
    display_columns.append("GameDate")
display_columns.extend([
    "Inning", "TaggedPitchType", "PitchCall",
    "PlateLocSide", "PlateLocHeight", "RelSpeed", "SpinRate", "CloseCall",
])

event = st.dataframe(
    missed_calls_df[display_columns],
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_pitch = None
if event.selection.rows:
    idx = event.selection.rows[0]
    selected_pitch = missed_calls_df.index[idx]


# Interactive strike zone chart. Green points are correct calls, red are missed.
# Close calls are labelled "CC" in white text. Selecting a row in the missed calls
# table above highlights that pitch in yellow. The solid rectangle is the rulebook
# zone; the dotted rectangle is the outer edge of the close-call buffer band.

st.subheader("Strike Zone")

fig = px.scatter(
    filtered,
    x="PlateLocSide",
    y="PlateLocHeight",
    color="CallResult",
    color_discrete_map={"Correct Call": "green", "Missed Call": "red"},
    hover_data={
        "Pitcher": True, "PitcherTeam": True, "Batter": True, "BatterTeam": True,
        "TaggedPitchType": True, "PitchCall": True, "Balls": True, "Strikes": True,
        "Inning": True, "RelSpeed": True, "SpinRate": True,
        "MissedCall": True, "InZone": True, "CloseCall": True,
    },
    height=850,
)
fig.update_traces(marker=dict(size=18, line=dict(width=1, color="black"), opacity=0.85))

if is_close_call.any():
    close_call_points = filtered[is_close_call]
    fig.add_trace(go.Scatter(
        x=close_call_points["PlateLocSide"],
        y=close_call_points["PlateLocHeight"],
        mode="text",
        text=["CC"] * len(close_call_points),
        textposition="middle center",
        textfont=dict(color="white", size=10),
        showlegend=False,
        hoverinfo="skip",
    ))

if selected_pitch is not None and selected_pitch in filtered.index:
    p = filtered.loc[selected_pitch]
    fig.add_trace(go.Scatter(
        x=[p["PlateLocSide"]], y=[p["PlateLocHeight"]],
        mode="markers",
        marker=dict(size=34, color="yellow", line=dict(width=4, color="black")),
        showlegend=False,
        hovertemplate=(
            f"<b>Pitcher:</b> {p['Pitcher']}<br>"
            f"<b>Batter:</b> {p['Batter']}<br>"
            f"<b>Type:</b> {p['TaggedPitchType']}<br>"
            f"<b>Call:</b> {p['PitchCall']}<br><extra></extra>"
        ),
    ))

fig.add_shape(type="rect", x0=ZONE_LEFT, y0=ZONE_BOTTOM, x1=ZONE_RIGHT, y1=ZONE_TOP,
              line=dict(width=4, color="black"))
fig.add_shape(type="rect", x0=BUFFER_LEFT, y0=BUFFER_BOTTOM, x1=BUFFER_RIGHT, y1=BUFFER_TOP,
              line=dict(width=2, color="black", dash="dot"))

fig.update_layout(
    xaxis_title="Horizontal Location",
    yaxis_title="Vertical Location",
    hovermode="closest",
    plot_bgcolor="white",
    margin=dict(t=140, b=40, l=40, r=40),
)

stats_text = (
    f"Total Called: {total_pitches} | Missed: {int(missed_calls)} | "
    f"Accuracy: {overall_accuracy:.1f}% | CS%: {called_strike_accuracy:.1f}% | "
    f"CB%: {called_ball_accuracy:.1f}% | Ball to Strike%: {called_ball_strike_pct:.1f}% | "
    f"CC: {len(close_calls)} | CC Correct%: {close_call_accuracy:.1f}%"
)
fig.add_annotation(
    text=stats_text, xref="paper", yref="paper",
    x=0.5, y=1.15, showarrow=False, font=dict(size=14),
)
fig.update_yaxes(scaleanchor="x", scaleratio=1, range=y_range)
fig.update_xaxes(range=x_range)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Full Dataset"):
    st.dataframe(filtered, use_container_width=True)