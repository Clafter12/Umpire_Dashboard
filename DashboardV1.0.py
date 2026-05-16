import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration

st.set_page_config(
    page_title="Umpire Dashboard",
    layout="wide"
)

st.title("Umpire Analytics Dashboard")

st.markdown("""
Upload a Trackman CSV to begin:
""")

# CSV upload

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file.")
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

df["Pitcher"] = df["Pitcher"].apply(clean_name)
df["Batter"] = df["Batter"].apply(clean_name)

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
    "NEW_JER": "New Jersey Jackals",
    "JOL_SLA": "Joliet Slammers"
}

def clean_team(team):

    if pd.isna(team):
        return team

    return team_map.get(str(team), str(team))

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

CC_BUFFER = 0.99 * BASEBALL_DIAMETER

# Strike zone logic

def is_in_zone(row):

    return (
        (ZONE_LEFT - BASEBALL_RADIUS)
        <= row["PlateLocSide"]
        <= (ZONE_RIGHT + BASEBALL_RADIUS)

        and

        (ZONE_BOTTOM - BASEBALL_RADIUS)
        <= row["PlateLocHeight"]
        <= (ZONE_TOP + BASEBALL_RADIUS)
    )

df["InZone"] = df.apply(is_in_zone, axis=1)

# Close-call logic

def is_close_call(row):

    if not row["InZone"]:
        return False

    left_distance = abs(
        row["PlateLocSide"] - (ZONE_LEFT - BASEBALL_RADIUS)
    )

    right_distance = abs(
        (ZONE_RIGHT + BASEBALL_RADIUS) - row["PlateLocSide"]
    )

    bottom_distance = abs(
        row["PlateLocHeight"] - (ZONE_BOTTOM - BASEBALL_RADIUS)
    )

    top_distance = abs(
        (ZONE_TOP + BASEBALL_RADIUS) - row["PlateLocHeight"]
    )

    closest_edge = min(
        left_distance,
        right_distance,
        bottom_distance,
        top_distance
    )

    return closest_edge <= CC_BUFFER

df["CloseCall"] = df.apply(is_close_call, axis=1)

# Missed-call logic

def is_missed_call(row):

    if row["InZone"] and row["PitchCall"] == "BallCalled":
        return True

    if not row["InZone"] and row["PitchCall"] == "StrikeCalled":
        return True

    return False

df["MissedCall"] = df.apply(is_missed_call, axis=1)

# Call-result labeling

def classify_call(row):

    if row["MissedCall"]:
        return "Missed Call"

    return "Correct Call"

df["CallResult"] = df.apply(classify_call, axis=1)

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

filtered_stage1 = df.copy()

if pitcher_teams:
    filtered_stage1 = filtered_stage1[
        filtered_stage1["PitcherTeam"].isin(pitcher_teams)
    ]

if innings:
    filtered_stage1 = filtered_stage1[
        filtered_stage1["Inning"].isin(innings)
    ]

pitchers = st.sidebar.multiselect(
    "Pitcher",
    sorted(filtered_stage1["Pitcher"].dropna().unique())
)

filtered_stage2 = filtered_stage1.copy()

if pitchers:
    filtered_stage2 = filtered_stage2[
        filtered_stage2["Pitcher"].isin(pitchers)
    ]

batters = st.sidebar.multiselect(
    "Batter",
    sorted(filtered_stage2["Batter"].dropna().unique())
)

filtered_stage3 = filtered_stage2.copy()

if batters:
    filtered_stage3 = filtered_stage3[
        filtered_stage3["Batter"].isin(batters)
    ]

pitch_types = st.sidebar.multiselect(
    "Pitch Type",
    sorted(filtered_stage3["TaggedPitchType"].dropna().unique())
)

filtered = filtered_stage3.copy()

if pitch_types:
    filtered = filtered[
        filtered["TaggedPitchType"].isin(pitch_types)
    ]

# Dynamic chart scaling

x_min = filtered["PlateLocSide"].min()
x_max = filtered["PlateLocSide"].max()

y_min = filtered["PlateLocHeight"].min()
y_max = filtered["PlateLocHeight"].max()

# Visual padding

x_padding = 0.35
y_padding = 0.35

# Symmetrical horizontal scaling

max_x = max(abs(x_min), abs(x_max)) + x_padding

x_range = [
    -max_x,
    max_x
]

y_range = [
    max(0, y_min - y_padding),
    y_max + y_padding
]

# Metrics

st.subheader("Umpire Stats")

total_pitches = len(filtered)
missed_calls = filtered["MissedCall"].sum()
correct_calls = total_pitches - missed_calls

overall_accuracy = (
    correct_calls / total_pitches * 100
    if total_pitches > 0 else 0
)

called_strikes = filtered[
    filtered["PitchCall"] == "StrikeCalled"
]

correct_called_strikes = called_strikes[
    called_strikes["InZone"]
]

called_strike_accuracy = (
    len(correct_called_strikes)
    / len(called_strikes) * 100
    if len(called_strikes) > 0 else 0
)

called_balls = filtered[
    filtered["PitchCall"] == "BallCalled"
]

correct_called_balls = called_balls[
    called_balls["InZone"] == False
]

called_ball_accuracy = (
    len(correct_called_balls)
    / len(called_balls) * 100
    if len(called_balls) > 0 else 0
)

called_balls_that_were_strikes = called_balls[
    called_balls["InZone"]
]

called_ball_strike_pct = (
    len(called_balls_that_were_strikes)
    / len(called_balls) * 100
    if len(called_balls) > 0 else 0
)

# Close-call metrics

close_calls = filtered[
    filtered["CloseCall"]
]

correct_close_calls = close_calls[
    close_calls["PitchCall"] == "StrikeCalled"
]

close_call_accuracy = (
    len(correct_close_calls)
    / len(close_calls) * 100
    if len(close_calls) > 0 else 0
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
    "Inning",
    "TaggedPitchType",
    "PitchCall",
    "PlateLocSide",
    "PlateLocHeight",
    "RelSpeed",
    "SpinRate",
    "CloseCall"
]

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

# Add CC labels

close_call_points = filtered[
    filtered["CloseCall"]
]

fig.add_trace(
    go.Scatter(
        x=close_call_points["PlateLocSide"],
        y=close_call_points["PlateLocHeight"],
        mode="text",
        text=["CC"] * len(close_call_points),
        textposition="middle center",
        textfont=dict(
            color="white",
            size=10
        ),
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