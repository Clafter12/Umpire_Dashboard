# Umpire Analytics Dashboard
# Author: Christian Lafter
# Date: 6/2/26
# Description: A Streamlit dashboard for analyzing umpire performance using Trackman CSV data. 
# Supports direct CSV upload or FTP download from the Trackman server. Provides metrics, 
# strike zone visualization, and missed call analysis.
import io
import datetime
from ftplib import FTP, error_perm
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Configuration

st.set_page_config(
    page_title="Umpire Dashboard",
    layout="wide"
)

st.title("Umpire Analytics Dashboard")

st.markdown("""
Upload a Trackman CSV to begin:
""")

# FTP download helpers

def _get_date_column(df):
    # Prefer an explicit Date column, then any exact date-like column name, then any column containing "date".
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
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
    if df[column_name].notna().any():
        df["GameDate"] = df[column_name].dt.date
        return True
    return False


def _connect_ftp(host, port, username, password):
    try:
        ftp = FTP(timeout=30)
        ftp.connect(host, int(port))
        ftp.login(username, password)
        return ftp
    except Exception as e:
        raise Exception(f"FTP connection failed: {e}")


def _retrieve_path(ftp, path):
    bio = io.BytesIO()
    ftp.retrbinary(f"RETR {path}", bio.write)
    bio.seek(0)
    return bio


def _download_latest_csv_from_directory(ftp, remote_dir):
    entries = ftp.nlst(remote_dir)
    csv_files = [f for f in entries if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in FTP directory: {remote_dir}"
        )

    def _get_mdtm(filepath):
        try:
            resp = ftp.sendcmd(f"MDTM {filepath}")
            return datetime.datetime.strptime(resp[4:], "%Y%m%d%H%M%S")
        except Exception:
            return datetime.datetime.min

    latest = max(csv_files, key=_get_mdtm)
    return _retrieve_path(ftp, latest)


def _download_csv_from_ftp(host, port, username, password, remote_path):
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
    keywords = [
        "trackman",
        "baseball",
        "game",
        "umpire",
        "pitch",
        "report",
        "events",
        "stats"
    ]
    excluded = "playerpositioning"

    # Do not filter by year anymore — only exclude unwanted filename patterns
    filtered = [
        path for path in paths
        if excluded not in path.lower()
    ]
    matches = [
        path for path in filtered
        if any(keyword in path.lower() for keyword in keywords)
    ]
    return sorted(matches or filtered)


def _extract_date_from_filename(filename):
    """Extract date from filename. Supports common date formats like YYYYMMDD or YYYY-MM-DD."""
    import re
    
    # Try YYYYMMDD format
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            date_str = match.group(1)
            return datetime.datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            pass
    
    # Try YYYY-MM-DD format
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if match:
        try:
            return datetime.datetime.strptime(match.group(0), "%Y-%m-%d").date()
        except ValueError:
            pass
    
    return None


def _organize_files_by_date(file_list):
    """Organize files by extracted date, return dict with dates as keys."""
    files_by_date = {}
    for filepath in file_list:
        date = _extract_date_from_filename(filepath)
        if date:
            if date not in files_by_date:
                files_by_date[date] = []
            files_by_date[date].append(filepath)
    return files_by_date


# Team mapping

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
    "JOL_SLA": "Joliet Slammers"
}


def clean_team(team):

    if pd.isna(team):
        return team

    return team_map.get(str(team), str(team))


def _extract_team_codes_from_filename(filename):
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
    return [_get_ftp_display_label(path) for path in paths]


# Data source selection

data_source = st.sidebar.radio(
    "Data Source",
    ["Upload CSV", "FTP Download"]
)

uploaded_file = None

if data_source == "FTP Download":
    st.sidebar.header("FTP Settings")

    ftp_host = st.sidebar.text_input(
        "FTP Host",
        value="ftp.trackmanbaseball.com",
        disabled=True,
        key="ftp_host"
    )

    ftp_port = st.sidebar.number_input(
        "Port",
        value=21,
        min_value=1,
        max_value=65535,
        key="ftp_port"
    )

    # FTP credentials - Direct credentials used for login
    ftp_username = "Frontier League"
    ftp_password = "VHq3wDSmJr"

    # Display only (does not overwrite actual credentials)
    st.sidebar.text_input(
        "Username",
        value=ftp_username,
        disabled=True,
        key="ftp_username_display"
    )

    st.sidebar.text_input(
        "Password",
        value="********",
        type="password",
        disabled=True,
        key="ftp_password_display"
    )

    ftp_scan_base = st.sidebar.text_input(
        "FTP scan start directory",
        value="",
        help="Leave blank to scan the FTP root for CSV files.",
        key="ftp_scan_base"
    )

    exclude_unverified = st.sidebar.checkbox(
        "Exclude 'Unverified' files",
        value=False,
        key="ftp_exclude_unverified",
        help="When checked, files with 'unverified' in the filename will be ignored during the scan."
    )

    if st.sidebar.button("Scan FTP for CSV files", key="ftp_scan_button"):
        try:
            ftp = _connect_ftp(ftp_host, ftp_port, ftp_username, ftp_password)
            try:
                scan_path = ftp_scan_base or "."
                csv_files = _scan_ftp_for_csv(ftp, scan_path)

                # Optionally remove files that include 'unverified' in the filename
                excluded_unverified_count = 0
                if exclude_unverified:
                    pre_count = len(csv_files)
                    csv_files = [f for f in csv_files if "unverified" not in f.lower()]
                    excluded_unverified_count = pre_count - len(csv_files)

                # Build diagnostics for scanned files: counts per year and undated files
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
                    "excluded_unverified": excluded_unverified_count if 'excluded_unverified_count' in locals() else 0,
                }

                st.session_state["ftp_scan_results"] = _filter_ftp_candidates(csv_files)
                
                if not st.session_state["ftp_scan_results"]:
                    st.sidebar.warning("No CSV files found during scan.")
                else:
                    st.sidebar.success(
                        f"Found {len(st.session_state['ftp_scan_results'])} candidate CSV files."
                    )
            finally:
                try:
                    ftp.quit()
                except Exception:
                    ftp.close()
        except Exception as exc:
            st.sidebar.error(f"FTP scan failed: {exc}")

    scan_results = st.session_state.get("ftp_scan_results", [])

    # (Diagnostics removed) The FTP scan diagnostics panel was removed per user request.

    if scan_results:
        # Organize files by date for filtering
        files_by_date = _organize_files_by_date(scan_results)
        
        if files_by_date:
            st.sidebar.subheader("Filter by Date")
            
            # Calendar picker for date selection
            available_dates = sorted(files_by_date.keys())
            selected_date = st.sidebar.date_input(
                "Select Game Date",
                value=available_dates[0] if available_dates else datetime.date.today(),
                min_value=available_dates[0] if available_dates else datetime.date.today(),
                max_value=available_dates[-1] if available_dates else datetime.date.today(),
                key="ftp_date_select"
            )
            
            # Get files for selected date
            files_for_date = files_by_date.get(selected_date, [])
            
            if files_for_date:
                if len(files_for_date) == 1:
                    ftp_remote_path = files_for_date[0]
                    st.sidebar.info(
                        f"Selected: {_get_ftp_display_label(ftp_remote_path)}"
                    )
                else:
                    st.sidebar.write(f"**{len(files_for_date)} files found for this date**")
                    display_labels = _build_ftp_display_labels(files_for_date)
                    selected_label = st.sidebar.selectbox(
                        "Choose file",
                        display_labels,
                        key="ftp_csv_select"
                    )
                    selected_index = display_labels.index(selected_label)
                    ftp_remote_path = files_for_date[selected_index]
                
                if st.sidebar.button("Download selected FTP CSV", key="ftp_download_button"):
                    try:
                        ftp_file = _download_csv_from_ftp(
                            ftp_host,
                            ftp_port,
                            ftp_username,
                            ftp_password,
                            ftp_remote_path
                        )
                        st.session_state["ftp_file_bytes"] = ftp_file.getvalue()
                        st.sidebar.success("FTP CSV downloaded successfully.")
                    except Exception as exc:
                        st.sidebar.error(f"FTP download failed: {exc}")
            else:
                st.sidebar.warning(f"No files found for {selected_date.strftime('%Y-%m-%d')}. Try another date.")
                ftp_remote_path = None
        else:
            st.sidebar.warning("No dates found in filenames. Showing all files.")
            display_labels = _build_ftp_display_labels(scan_results)
            selected_label = st.sidebar.selectbox(
                "Choose file",
                display_labels,
                key="ftp_csv_select"
            )
            selected_index = display_labels.index(selected_label)
            ftp_remote_path = scan_results[selected_index]
            
            if st.sidebar.button("Download selected FTP CSV", key="ftp_download_button"):
                try:
                    ftp_file = _download_csv_from_ftp(
                        ftp_host,
                        ftp_port,
                        ftp_username,
                        ftp_password,
                        ftp_remote_path
                    )
                    st.session_state["ftp_file_bytes"] = ftp_file.getvalue()
                    st.sidebar.success("FTP CSV downloaded successfully.")
                except Exception as exc:
                    st.sidebar.error(f"FTP download failed: {exc}")
    else:
        ftp_remote_path = st.sidebar.text_input(
            "Remote file or directory",
            value="",
            key="ftp_remote_path"
        )
        st.sidebar.info(
            "Scan the FTP server to list CSVs automatically, or enter a file/directory path manually."
        )

    if st.session_state.get("ftp_file_bytes") is not None:
        uploaded_file = io.BytesIO(st.session_state["ftp_file_bytes"])
        st.sidebar.success("FTP CSV loaded from session.")
    else:
        st.sidebar.info("After choosing a file, click Download selected FTP CSV.")

else:
    st.sidebar.header("Upload CSV")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

if uploaded_file is None:
    st.info("Please upload a CSV file or download one from FTP.")
    st.stop()

# Load CSV

try:

    df = pd.read_csv(uploaded_file)

    # If PitchNo is missing,
    # retry while skipping first row

    if "PitchNo" not in df.columns:

        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file,
            skiprows=1
        )

except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

# Keep only called pitches

df = df[df["PitchCall"].str.contains("Called", case=False, na=False)]

# Clean names

def clean_name(name):

    if pd.isna(name):
        return name

    if "," in str(name):
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"

    return name

# --- SAFE NAME CLEANING (Pitcher/Batter) ---

if "Pitcher" in df.columns:  # line ~235
    df["Pitcher"] = df["Pitcher"].apply(clean_name)
else:
    st.error(f"Missing column: Pitcher. Available: {list(df.columns)}")
    st.stop()

if "Batter" in df.columns:  # line ~241
    df["Batter"] = df["Batter"].apply(clean_name)
else:
    st.error(f"Missing column: Batter. Available: {list(df.columns)}")
    st.stop()

df["PitcherTeam"] = df["PitcherTeam"].apply(clean_team)
df["BatterTeam"] = df["BatterTeam"].apply(clean_team)

# Strike zone constants

ZONE_LEFT = -0.83
ZONE_RIGHT = 0.83
ZONE_BOTTOM = 1.5
ZONE_TOP = 3.5

# Baseball dimensions

BASEBALL_DIAMETER = 2.94 / 12
BASEBALL_RADIUS = BASEBALL_DIAMETER / 2

# Close-call buffer

CC_BUFFER = 0.5 * BASEBALL_DIAMETER

# Buffer zone dimensions
# Visualizes the full close-call boundary

BUFFER_LEFT = (ZONE_LEFT - BASEBALL_RADIUS) - CC_BUFFER
BUFFER_RIGHT = (ZONE_RIGHT + BASEBALL_RADIUS) + CC_BUFFER
BUFFER_BOTTOM = (ZONE_BOTTOM - BASEBALL_RADIUS) - CC_BUFFER
BUFFER_TOP = (ZONE_TOP + BASEBALL_RADIUS) + CC_BUFFER

# Strike zone logic
# Use vectorized boolean operations instead of per-row apply for performance.
_x = df["PlateLocSide"]
_y = df["PlateLocHeight"]
df["InZone"] = (
    (_x >= (ZONE_LEFT - BASEBALL_RADIUS))
    & (_x <= (ZONE_RIGHT + BASEBALL_RADIUS))
    & (_y >= (ZONE_BOTTOM - BASEBALL_RADIUS))
    & (_y <= (ZONE_TOP + BASEBALL_RADIUS))
)

# Close-call logic

def is_close_call(row):
    x = row["PlateLocSide"]
    y = row["PlateLocHeight"]

    # Entire baseball boundaries
    ball_left = x - BASEBALL_RADIUS
    ball_right = x + BASEBALL_RADIUS
    ball_bottom = y - BASEBALL_RADIUS
    ball_top = y + BASEBALL_RADIUS

    # Distances from entire baseball to strike zone edges
    left_distance = abs(ball_right - ZONE_LEFT)
    right_distance = abs(ZONE_RIGHT - ball_left)
    bottom_distance = abs(ball_top - ZONE_BOTTOM)
    top_distance = abs(ZONE_TOP - ball_bottom)

    closest_edge = min(left_distance, right_distance, bottom_distance, top_distance)

    # Ball must be within expanded close-call boundary
    within_buffer = (BUFFER_LEFT <= x <= BUFFER_RIGHT and BUFFER_BOTTOM <= y <= BUFFER_TOP)

    return within_buffer and closest_edge <= CC_BUFFER

df["CloseCall"] = df.apply(is_close_call, axis=1)

# Missed-call logic
# Missed if ball was called when in zone, or strike was called when out of zone.
df["MissedCall"] = (
    (df["InZone"] & (df["PitchCall"] == "BallCalled"))
    | ((~df["InZone"]) & (df["PitchCall"] == "StrikeCalled"))
)

# Call-result labeling
df["CallResult"] = np.where(df["MissedCall"], "Missed Call", "Correct Call")

# Detect and filter by game date

date_column = _get_date_column(df)
if date_column is not None and _to_game_date(df, date_column):
    game_dates = sorted(df["GameDate"].dropna().unique())
    if game_dates:
        if len(game_dates) == 1:
            selected_date = st.sidebar.date_input(
                "Select Game Date",
                value=game_dates[0],
                min_value=game_dates[0],
                max_value=game_dates[0]
            )
            df = df[df["GameDate"] == selected_date]
        else:
            selected_range = st.sidebar.date_input(
                "Select Game Date Range",
                value=(game_dates[0], game_dates[-1]),
                min_value=game_dates[0],
                max_value=game_dates[-1]
            )
            if (
                isinstance(selected_range, (list, tuple))
                and len(selected_range) == 2
            ):
                start_date, end_date = selected_range
                df = df[
                    (df["GameDate"] >= start_date)
                    & (df["GameDate"] <= end_date)
                ]
            else:
                df = df[df["GameDate"] == selected_range]
        st.sidebar.caption(f"Detected date field: {date_column}")
else:
    st.sidebar.caption(
        "No date field detected. Upload an FTP/CSV with a date column."
    )

# Filters
st.sidebar.header("Filters")

pitcher_teams = st.sidebar.multiselect(
    "Pitcher Team",
    sorted(df["PitcherTeam"].dropna().unique())
)

innings = st.sidebar.multiselect(
    "Inning",
    sorted(df["Inning"].dropna().unique())
)

# Close-call only toggle
close_call_only = st.sidebar.toggle("Only Close Calls", value=False)

# Build filter mask efficiently
filter_mask = pd.Series([True] * len(df), index=df.index)

if pitcher_teams:
    filter_mask &= df["PitcherTeam"].isin(pitcher_teams)

if innings:
    filter_mask &= df["Inning"].isin(innings)

# Apply first-stage filters
filtered_stage1 = df[filter_mask]

pitchers = st.sidebar.multiselect(
    "Pitcher",
    sorted(filtered_stage1["Pitcher"].dropna().unique())
)

filter_mask2 = pd.Series([True] * len(filtered_stage1), index=filtered_stage1.index)

if pitchers:
    filter_mask2 &= filtered_stage1["Pitcher"].isin(pitchers)

filtered_stage2 = filtered_stage1[filter_mask2]

batters = st.sidebar.multiselect(
    "Batter",
    sorted(filtered_stage2["Batter"].dropna().unique())
)

filter_mask3 = pd.Series([True] * len(filtered_stage2), index=filtered_stage2.index)

if batters:
    filter_mask3 &= filtered_stage2["Batter"].isin(batters)

filtered_stage3 = filtered_stage2[filter_mask3]

pitch_types = st.sidebar.multiselect(
    "Pitch Type",
    sorted(filtered_stage3["TaggedPitchType"].dropna().unique())
)

# Final filter with close-call toggle
filter_mask_final = pd.Series([True] * len(filtered_stage3), index=filtered_stage3.index)

if pitch_types:
    filter_mask_final &= filtered_stage3["TaggedPitchType"].isin(pitch_types)

if close_call_only:
    filter_mask_final &= filtered_stage3["CloseCall"]

filtered = filtered_stage3[filter_mask_final]

# Dynamic chart scaling - Calculate once
# Use the stage3 filtered dataset (before the close-call-only toggle) as the
# default scale source so toggling "Only Close Calls" doesn't aggressively
# shrink the axes. Fall back to `filtered` or strike-zone defaults when empty.
scale_source = filtered_stage3 if (filtered_stage3 is not None and len(filtered_stage3) > 0) else filtered

if scale_source is not None and len(scale_source) > 0:
    x_min, x_max = scale_source["PlateLocSide"].min(), scale_source["PlateLocSide"].max()
    y_min, y_max = scale_source["PlateLocHeight"].min(), scale_source["PlateLocHeight"].max()
else:
    # sensible defaults around the strike zone
    x_min, x_max = ZONE_LEFT, ZONE_RIGHT
    y_min, y_max = ZONE_BOTTOM, ZONE_TOP

# Visual padding
x_padding = 0.35
y_padding = 0.35

# Symmetrical horizontal scaling
max_x = max(abs(x_min), abs(x_max)) + x_padding

x_range = [-max_x, max_x]
y_range = [max(0, y_min - y_padding), y_max + y_padding]

# Metrics - Optimized to reduce redundant filtering
st.subheader("Umpire Stats")

total_pitches = len(filtered)
missed_calls = filtered["MissedCall"].sum()
correct_calls = total_pitches - missed_calls

overall_accuracy = (correct_calls / total_pitches * 100) if total_pitches > 0 else 0

# Cache boolean masks to avoid repeated filtering
is_called_strike = filtered["PitchCall"] == "StrikeCalled"
is_called_ball = filtered["PitchCall"] == "BallCalled"
is_in_zone = filtered["InZone"]
is_close_call = filtered["CloseCall"]

called_strikes = filtered[is_called_strike]
called_balls = filtered[is_called_ball]

called_strike_accuracy = (
    is_called_strike.sum() / is_called_strike.astype(int).sum() * 100
    if is_called_strike.sum() > 0 else 0
)

called_ball_accuracy = (
    (is_called_ball & ~is_in_zone).sum() / is_called_ball.sum() * 100
    if is_called_ball.sum() > 0 else 0
)

called_ball_strike_pct = (
    (is_called_ball & is_in_zone).sum() / is_called_ball.sum() * 100
    if is_called_ball.sum() > 0 else 0
)

# Close-call metrics
close_calls = filtered[is_close_call]
close_call_accuracy = (
    ((is_close_call & is_called_strike).sum()) / is_close_call.sum() * 100
    if is_close_call.sum() > 0 else 0
)

# Display metrics

col1, col2, col3 = st.columns(3)

col1.metric("Total Called Pitches", total_pitches)
col2.metric("Missed Calls", int(missed_calls))
col3.metric("Overall Accuracy %", f"{overall_accuracy:.1f}%")

col4, col5, col6, col7, col8 = st.columns(5)

col4.metric(
    "Called Strike Accuracy",
    f"{called_strike_accuracy:.1f}%"
)

col5.metric(
    "Called Ball Accuracy",
    f"{called_ball_accuracy:.1f}%"
)

col6.metric(
    "% Called Balls That Were Strikes",
    f"{called_ball_strike_pct:.1f}%"
)

col7.metric(
    "Close Calls",
    len(close_calls)
)

col8.metric(
    "CC Correct %",
    f"{close_call_accuracy:.1f}%"
)

# Missed calls table

st.subheader("Missed Calls")

missed_calls_df = filtered[
    filtered["MissedCall"]
].copy()

display_columns = [
    "Pitcher",
    "PitcherTeam",
    "Batter",
    "BatterTeam",
]

if "GameDate" in missed_calls_df.columns:
    display_columns.append("GameDate")

display_columns.extend([
    "Inning",
    "TaggedPitchType",
    "PitchCall",
    "PlateLocSide",
    "PlateLocHeight",
    "RelSpeed",
    "SpinRate",
    "CloseCall"
])

event = st.dataframe(
    missed_calls_df[display_columns],
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row"
)

# Current selected pitch

selected_pitch = None

if event.selection.rows:
    idx = event.selection.rows[0]
    selected_pitch = missed_calls_df.index[idx]

# Strike zone chart

st.subheader("Strike Zone")

fig = px.scatter(
    filtered,
    x="PlateLocSide",
    y="PlateLocHeight",
    color="CallResult",
    color_discrete_map={
        "Correct Call": "green",
        "Missed Call": "red"
    },
    hover_data={
        "Pitcher": True,
        "PitcherTeam": True,
        "Batter": True,
        "BatterTeam": True,
        "TaggedPitchType": True,
        "PitchCall": True,
        "Balls": True,
        "Strikes": True,
        "Inning": True,
        "RelSpeed": True,
        "SpinRate": True,
        "MissedCall": True,
        "InZone": True,
        "CloseCall": True
    },
    height=850
)

fig.update_traces(
    marker=dict(
        size=18,
        line=dict(width=1, color="black"),
        opacity=0.85
    )
)

# Add CC labels for close-call points
if is_close_call.any():
    close_call_points = filtered[is_close_call]
    fig.add_trace(
        go.Scatter(
            x=close_call_points["PlateLocSide"],
            y=close_call_points["PlateLocHeight"],
            mode="text",
            text=["CC"] * len(close_call_points),
            textposition="middle center",
            textfont=dict(color="white", size=10),
            showlegend=False,
            hoverinfo="skip"
        )
    )

# Highlight selected pitch

if selected_pitch is not None:

    if selected_pitch in filtered.index:

        p = filtered.loc[selected_pitch]

        fig.add_trace(
            go.Scatter(
                x=[p["PlateLocSide"]],
                y=[p["PlateLocHeight"]],
                mode="markers",
                marker=dict(
                    size=34,
                    color="yellow",
                    line=dict(
                        width=4,
                        color="black"
                    )
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>Pitcher:</b> {p['Pitcher']}<br>"
                    f"<b>Batter:</b> {p['Batter']}<br>"
                    f"<b>Type:</b> {p['TaggedPitchType']}<br>"
                    f"<b>Call:</b> {p['PitchCall']}<br>"
                    f"<extra></extra>"
                )
            )
        )

# Strike zone

fig.add_shape(
    type="rect",
    x0=ZONE_LEFT,
    y0=ZONE_BOTTOM,
    x1=ZONE_RIGHT,
    y1=ZONE_TOP,
    line=dict(
        width=4,
        color="black"
    )
)

# Buffer zone
# Dotted black line represents
# the full close-call boundary

fig.add_shape(
    type="rect",
    x0=BUFFER_LEFT,
    y0=BUFFER_BOTTOM,
    x1=BUFFER_RIGHT,
    y1=BUFFER_TOP,
    line=dict(
        width=2,
        color="black",
        dash="dot"
    )
)

fig.update_layout(
    xaxis_title="Horizontal Location",
    yaxis_title="Vertical Location",
    hovermode="closest",
    plot_bgcolor="white",
    margin=dict(
        t=140,
        b=40,
        l=40,
        r=40
    )
)

stats_text = (
    f"Total Called: {total_pitches} | "
    f"Missed: {int(missed_calls)} | "
    f"Accuracy: {overall_accuracy:.1f}% | "
    f"CS%: {called_strike_accuracy:.1f}% | "
    f"CB%: {called_ball_accuracy:.1f}% | "
    f"Ball→Strike%: {called_ball_strike_pct:.1f}% | "
    f"CC: {len(close_calls)} | "
    f"CC Correct%: {close_call_accuracy:.1f}%"
)

fig.add_annotation(
    text=stats_text,
    xref="paper",
    yref="paper",
    x=0.5,
    y=1.15,
    showarrow=False,
    font=dict(size=14)
)

# True-scale chart axes

fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
    range=y_range
)

fig.update_xaxes(
    range=x_range
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Full dataset

with st.expander("Full Dataset"):
    st.dataframe(
        filtered,
        use_container_width=True
    )

# CSV export

st.subheader("Download Cleaned Report")

csv = filtered.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download CSV Report",
    data=csv,
    file_name="umpire_report.csv",
    mime="text/csv"
)