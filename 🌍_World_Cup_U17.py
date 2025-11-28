import pandas as pd
import re
import math
import numpy as np
from scipy import stats
import streamlit as st
from mplsoccer import PyPizza, FontManager
import matplotlib.pyplot as plt
from highlight_text import fig_text
import textwrap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from mplsoccer import Pitch
import subprocess
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mplsoccer import Pitch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from mplsoccer import VerticalPitch




# Layout setup
st.set_page_config(page_title='World Cup U17', layout='centered', page_icon='🌍')

# Tabs
tab1, tab2, tab3 = st.tabs(["Player Total Stats", "Player p90 Stats","Player Maps"])

# ------------------------- TAB 1 -------------------------
with tab1:
    st.subheader("📊 Player Total Stats")

    # =========================================
    # LOAD DATA ONCE (no league dropdown)
    # =========================================
    if "df_stats" not in st.session_state:
        st.session_state["df_stats"] = pd.read_excel(
            "Player Season Stats - World Cup U17.xlsx"
        )

    df = st.session_state["df_stats"]

    # Rename for display
    df = df.rename(columns={"shirt_number": "Shirt Number"})

    # Separate total & percentile columns
    total_columns = [
        col for col in df.columns
        if 'p90' not in col and 'Percentile' not in col and col != 'Last Updated'
    ]
    percentile_columns = [
        col for col in df.columns
        if 'Percentile' in col and 'p90' not in col
    ]

    stat_base_names = sorted([
        col for col in total_columns if col not in [
            'index', 'Player Id', 'Full Name', 'Match Name', 'Team Name',
            'Team Id', '90s', 'Most Played Position',
            'Positions Played', 'Number of Positions Played',
            'Position Group', 'position', 'Time Played', "Shirt Number"
        ]
    ])

    # Order stats: stat, then its percentile (if exists)
    ordered_cols = []
    for stat in stat_base_names:
        ordered_cols.append(stat)
        perc = f"{stat} Percentile"
        if perc in percentile_columns:
            ordered_cols.append(perc)

    identifying_cols = [
        'Match Name', 'Team Name', 'Position Group',
        'Time Played', '90s', "Shirt Number"
    ]

    full_df = df[identifying_cols + ordered_cols].copy()

    # ==============================
    # LEAGUE-LEVEL RANKING GROUP
    # ==============================

    # 1) Position group filter (defines ranking group within the league)
    position_groups = sorted(
        set(pos for sub in full_df['Position Group'].dropna().str.split(', ')
            for pos in sub)
    )
    pos = st.selectbox("Select Position Group", position_groups, key="position3")

    # 👉 This subset is used for RANKING (league + position group)
    league_pos_df = full_df[full_df['Position Group'].str.contains(pos, na=False)].copy()

    # ==============================
    # RANKS (computed once on league_pos_df)
    # ==============================

    negative_stats = [
        'Minutes Per Goal', 'Aerial Duels lost', 'Dispossessed', 'Duels lost', 'Ground Duels lost',
        'Handballs conceded', 'Hit Woodwork', 'Offsides', 'Overruns', 'Foul Attempted Tackle',
        'GK Unsuccessful Distribution', 'Goals Conceded', 'Goals Conceded Outside Box',
        'Goals Conceded Inside Box', 'Own Goal Scored', 'Penalties Conceded',
        'Red Cards - 2nd Yellow', 'Straight Red Cards', 'Total Red Cards',
        'Total Unsuccessful Passes ( Excl Crosses & Corners )',
        'Unsuccessful Corners into Box', 'Unsuccessful Crosses & Corners',
        'Unsuccessful Crosses open play', 'Unsuccessful Dribbles', 'Unsuccessful Launches',
        'Unsuccessful Long Passes', 'Unsuccessful Passes Opposition Half',
        'Unsuccessful Passes Own Half', 'Unsuccessful Short Passes', 'Unsuccessful lay-offs',
        'Yellow Cards', 'Substitute Off', 'Tackles Lost', 'Total Fouls Conceded',
        'Total Losses Of Possession', 'Shots Off Target (inc woodwork)', 'Shots Per Goal'
    ]

    # Select stats (on league-level, but user chooses which to show)
    available_stats = [col for col in ordered_cols if 'Percentile' not in col]
    selected_stats = st.multiselect(
        "Select Stats to Display",
        available_stats,
        default=available_stats,
        key="stats3"
    )

    # Compute ranks once on the league+position group data
    ranked_df = league_pos_df.copy()
    for stat in selected_stats:
        if stat not in ranked_df.columns:
            continue
        rank_col = f"{stat} Rank"
        ascending = stat in negative_stats  # lower is better for negative stats
        ranked_df[rank_col] = ranked_df[stat].rank(
            ascending=ascending,
            method='min'
        ).astype("Int64")

    # ==============================
    # DISPLAY FILTERS (Team & Player)
    # these DO NOT affect ranking group
    # ==============================

    # Team list based on league_pos_df (not ranked_df subset)
    teams = sorted(
        set(t for sub in league_pos_df['Team Name'].dropna().str.split(', ')
            for t in sub)
    )
    selected_team = st.selectbox("Select Team", ["All"] + teams, key="team3")

    display_df = ranked_df.copy()
    if selected_team != "All":
        display_df = display_df[
            display_df['Team Name'].str.contains(selected_team, na=False)
        ]

    # Player list based on already team-filtered display_df
    players = sorted(display_df['Match Name'].unique())
    selected_player = st.selectbox("Select Player", ["All"] + players, key="player3")
    if selected_player != "All":
        display_df = display_df[display_df['Match Name'] == selected_player]

    # Keep only identifying + selected stat columns (+ ranks + percentiles)
    cols_to_show = identifying_cols.copy()
    for stat in selected_stats:
        if stat in display_df.columns:
            cols_to_show.append(stat)
        perc = f"{stat} Percentile"
        if perc in display_df.columns:
            cols_to_show.append(perc)
        rank_col = f"{stat} Rank"
        if rank_col in display_df.columns:
            cols_to_show.append(rank_col)

    display_df = display_df[cols_to_show]

    # ==========================
    # STYLING & DISPLAY (fixed)
    # ==========================

    # Custom light gradient colormap from red to green
    light_red_to_green = LinearSegmentedColormap.from_list(
        "custom_redgreen",
        ["#f25b5b", "#f6f675", "#1fff4c"]
    )

    # Get global min/max rank (league level)
    rank_columns = [f"{stat} Rank" for stat in selected_stats if f"{stat} Rank" in ranked_df.columns]
    global_min = ranked_df[rank_columns].min().min()
    global_max = ranked_df[rank_columns].max().max()

    def apply_custom_gradient(s):
        # Use global league-level range for normalization
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
        cmap = light_red_to_green.reversed()
        return [f'background-color: {mcolors.to_hex(cmap(norm(val)))}' for val in s]

    if st.button("Show Total Stats", key="show_stats3"):
        styled_df = display_df.style
        for stat in selected_stats:
            rank_col = f"{stat} Rank"
            if rank_col in display_df.columns:
                styled_df = styled_df.apply(apply_custom_gradient, subset=[rank_col])
        st.dataframe(styled_df, height=750)

    with st.expander("Metric Glossary"):
         st.write("""
    - **Overrun**: Heavy touch in a dribble.
    - **Progressive Passes**: A pass that moves the ball closer to the opponent goal by 25% & at least 5 m vertically.
    - **Lay-off**: A pass by a striker who has received the ball with back to goal.
    - **Dispossessed**: Losing the ball under pressure.
    - **GK Distribution**: Successful goalkeeper passes.
    - **GK Launches**: Long balls launched forward.
    - **Other Goals**: Goals not scored with foot or head.""")  


#--------------------------TAB 2 -------------------------
 
with tab2:
    st.subheader("⏱️ Player p90 Stats")

    # Load data once
    if "df_p90" not in st.session_state:
        st.session_state["df_p90"] = pd.read_excel(
            "Player Season Stats - World Cup U17.xlsx"
        )

    df = st.session_state["df_p90"]

    # Basic setup
    df = df.rename(columns={"shirt_number": "Shirt Number"})
    df = df.loc[:, ~df.columns.str.endswith("p90 p90")]  # clean any double suffix

    identifying_cols = ['Match Name', 'Team Name', 'Position Group', 'Time Played', '90s', "Shirt Number"]
    p90_columns = [col for col in df.columns if col.endswith('p90')]

    # ==============================
    # LEAGUE-LEVEL RANKING GROUP
    # ==============================

    # Position group filter (defines ranking group within the league)
    position_groups = sorted(
        set(pos for sub in df['Position Group'].dropna().str.split(', ')
            for pos in sub)
    )
    pos = st.selectbox("Select Position Group", position_groups, key="position5")

    # 👉 This subset is used for RANKING (league + position group)
    league_pos_df = df[df['Position Group'].str.contains(pos, na=False)].copy()

    # p90 stats available for selection
    selected_stats = st.multiselect(
        "Select p90 Stats to Display",
        p90_columns,
        default=p90_columns,
        key="stats5"
    )

    # Negative stats definition
    negative_stats = [
        'Minutes Per Goal', 'Aerial Duels lost', 'Dispossessed', 'Duels lost', 'Ground Duels lost',
        'Handballs conceded', 'Hit Woodwork', 'Offsides', 'Overruns', 'Foul Attempted Tackle',
        'GK Unsuccessful Distribution', 'Goals Conceded', 'Goals Conceded Outside Box',
        'Goals Conceded Inside Box', 'Own Goal Scored', 'Penalties Conceded',
        'Red Cards - 2nd Yellow', 'Straight Red Cards', 'Total Red Cards',
        'Total Unsuccessful Passes ( Excl Crosses & Corners )',
        'Unsuccessful Corners into Box', 'Unsuccessful Crosses & Corners',
        'Unsuccessful Crosses open play', 'Unsuccessful Dribbles', 'Unsuccessful Launches',
        'Unsuccessful Long Passes', 'Unsuccessful Passes Opposition Half',
        'Unsuccessful Passes Own Half', 'Unsuccessful Short Passes', 'Unsuccessful lay-offs',
        'Yellow Cards', 'Substitute Off', 'Tackles Lost', 'Total Fouls Conceded',
        'Total Losses Of Possession', 'Shots Off Target (inc woodwork)', 'Shots Per Goal'
    ]

    # Extend negatives to p90
    negative_stats_p90 = [f"{stat} p90" for stat in negative_stats]

    # ==============================
    # RANKS (computed once on league_pos_df)
    # ==============================

    ranked_df = league_pos_df.copy()

    # Start display DF from identifiers
    for col in identifying_cols:
        if col not in ranked_df.columns:
            ranked_df[col] = None  # safety if any missing

    for stat in selected_stats:
        if stat not in ranked_df.columns:
            continue
        rank_col = f"{stat} Rank"
        ascending = stat in negative_stats_p90  # lower is better for negative stats
        ranked_df[rank_col] = ranked_df[stat].rank(
            ascending=ascending,
            method='min'
        ).astype("Int64")

    # ==============================
    # DISPLAY FILTERS (Team & Player)
    # these DO NOT affect ranking group
    # ==============================

    # Team list based on league_pos_df
    teams = sorted(
        set(t for sub in league_pos_df['Team Name'].dropna().str.split(', ')
            for t in sub)
    )
    selected_team = st.selectbox("Select Team", ["All"] + teams, key="team5")

    display_df = ranked_df.copy()
    if selected_team != "All":
        display_df = display_df[
            display_df['Team Name'].str.contains(selected_team, na=False)
        ]

    # Player list based on already team-filtered display_df
    players = sorted(display_df['Match Name'].unique())
    selected_player = st.selectbox("Select Player", ["All"] + players, key="player5")
    if selected_player != "All":
        display_df = display_df[display_df['Match Name'] == selected_player]

    # Final columns to show
    cols_to_show = identifying_cols.copy()
    for stat in selected_stats:
        if stat in display_df.columns:
            cols_to_show.append(stat)
        rank_col = f"{stat} Rank"
        if rank_col in display_df.columns:
            cols_to_show.append(rank_col)

    display_df = display_df[cols_to_show]

    # ==========================
    # STYLING & DISPLAY (league-wide gradient)
    # ==========================
    # Custom light gradient colormap from red to green
    light_red_to_green = LinearSegmentedColormap.from_list(
        "custom_redgreen",
        ["#f25b5b", "#f6f675", "#1fff4c"]
    )

    # League-wide rank min/max for all selected p90 stats
    rank_columns = [f"{stat} Rank" for stat in selected_stats if f"{stat} Rank" in ranked_df.columns]
    if rank_columns:
        global_min = ranked_df[rank_columns].min().min()
        global_max = ranked_df[rank_columns].max().max()
    else:
        global_min, global_max = 0, 1  # fallback

    def apply_custom_gradient(s):
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
        cmap = light_red_to_green.reversed()  # rank 1 = green
        return [f'background-color: {mcolors.to_hex(cmap(norm(val)))}' for val in s]

    if st.button("Show p90 Stats", key="show_p90_stats"):
        styled_df = display_df.style
        for stat in selected_stats:
            rank_col = f"{stat} Rank"
            if rank_col in display_df.columns:
                styled_df = styled_df.apply(apply_custom_gradient, subset=[rank_col])
        st.dataframe(styled_df, height=750)

    with st.expander("Metric Glossary"):
         st.write("""
    - **Overrun**: Heavy touch in a dribble.
    - **Progressive Passes**: A pass that moves the ball closer to the opponent goal by 25% & at least 5 m vertically.
    - **Lay-off**: A pass by a striker who has received the ball with back to goal.
    - **Dispossessed**: Losing the ball under pressure.
    - **GK Distribution**: Successful goalkeeper passes.
    - **GK Launches**: Long balls launched forward.
    - **Other Goals**: Goals not scored with foot or head.""") 
# ------------------------- TAB 3 -------------------------
with tab3:
    st.subheader("🗺️ Player Maps")
    league = "World Cup U17"

    @st.cache_data
    def load_data_maps():
        file_path = "World Cup U17.xlsx"
        df = pd.read_excel(file_path)

        df['endX'] = pd.to_numeric(df['endX'], errors='coerce')
        df['endY'] = pd.to_numeric(df['endY'], errors='coerce')
        df['goalMouthZ'] = pd.to_numeric(df['goalMouthZ'], errors='coerce')
        df['goalMouthY'] = pd.to_numeric(df['goalMouthY'], errors='coerce')
        df = df.sort_values(['id']).reset_index(drop=True)

        return df
    
    df = load_data_maps()

    # Team filter based on loaded data
    teams = sorted(df['teamName'].dropna().unique())
    selected_team = st.selectbox("Select Team", teams, key="team4")

    # Player filter based on selected team
    team_df = df[df['teamName'] == selected_team]
    players = sorted(team_df['playerName'].dropna().unique())
    selected_player = st.selectbox("Select Player", players, key="player4")

    # Map type selection
    map_type = st.selectbox(
        "Select Map Type",
        [
            "Passes Received",
            "Dribbles Map",
            "Key & Progressive Passes Map",
            "Shots Map",
            "Defensive Actions Map",
            "Tackles Map",
            "Recoveries & Interceptions Map",
        ],
        key="map_type4"
    )

    ############ PASSES RECEIVED MAP ##################
    def create_passes_received_map(df, player_name, team_name):
        
        all_events = df[df['teamName'] == team_name].copy()
        all_events['recipient name'] = all_events['playerName'].shift(-1)
        all_events['recipient id'] = all_events['playerId'].shift(-1)

        passes = all_events[
            (all_events['type/displayName'] == 'Pass') &
            (all_events['outcomeType/displayName'] == 'Successful')
        ]

        received = passes[passes['recipient name'] == player_name]
        #match_counts = received.groupby('matchDescription').size().reset_index(name='Passes Received')
        #st.table(match_counts)
        
        fig, ax = plt.subplots(figsize=(10.8, 10.8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        pitch_length, pitch_width = 105, 68
        pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
                    pitch_color='white', line_color='black', line_alpha=0.4, stripe=False)
        pitch.draw(ax=ax)

        # Logo
        logo = Image.open("Afican football analytics logo.png")
        logo = logo.resize((150, 150))
        ax_logo = fig.add_axes([0.70, 0.7, 0.20, 0.3])
        ax_logo.imshow(logo)
        ax_logo.axis('off')

        scaled_x = received.endX * pitch_length / 100
        scaled_y = received.endY * pitch_width / 100

        pitch.scatter(scaled_x, scaled_y, s=80, color='white', edgecolors='black', linewidth=1.5,
                    alpha=0.3, ax=ax, zorder=3)
        #heat map
        bin_stat = pitch.bin_statistic(scaled_x, scaled_y, statistic='count', bins=(5, 5))
        masked_bin_stat = bin_stat.copy()
        masked_bin_stat['statistic'] = np.ma.masked_where(bin_stat['statistic'] == 0, bin_stat['statistic'])
        hm = pitch.heatmap(masked_bin_stat, ax=ax, cmap='RdGy_r', alpha=0.9, zorder=1)

        # Horizontal colorbar
        cax = ax.inset_axes([0.02, 1.15, 0.26, 0.03])
        cbar = fig.colorbar(hm, cax=cax, orientation='horizontal')
        ax.text(2, pitch_width + 14, 'Number of Passes', fontsize=12,
                ha='left', va='center', color='black', family='monospace')

        # Grid
        x_step = pitch_length / 5
        y_step = pitch_width / 5
        for x in np.arange(x_step, pitch_length, x_step):
            ax.plot([x, x], [0, pitch_width], color='black', linewidth=0.9, alpha=0.8, zorder=2)
        for y in np.arange(y_step, pitch_width, y_step):
            ax.plot([0, pitch_length], [y, y], color='black', linewidth=0.9, alpha=0.8, zorder=2)

        # Zone counts
        for i in range(5):
            for j in range(5):
                x_start = i * x_step
                y_start = j * y_step
                count = ((scaled_x >= x_start) & (scaled_x < x_start + x_step) &
                        (scaled_y >= y_start) & (scaled_y < y_start + y_step)).sum()
                if count > 0:
                    ax.text(x_start + x_step / 2, y_start + y_step / 2, str(count),
                            color='black', size=20, weight='bold', ha='center', va='center', zorder=4)

        
        # Stats
        total = len(received)
        in_box = len(received[(received.endX > 84) & (received.endX < 100) &
                            (received.endY > 19) & (received.endY < 81)])

        ax.text(2, pitch_width +25, "RECEIVED PASSES", color='black', size=25,
                ha='left', va='top', weight='bold', family='monospace')
        ax.text(2, pitch_width + 20, f"{player_name} - {league}", color='black', size=16,
                ha='left', va='top', family='monospace')
        ax.text(2, pitch_width + 6,
                f"{total} Passes Received     {in_box} Inside The Box",
                color='black', size=14, ha='left', va='top',
                family='monospace', weight='bold')
        
        # Add pitch direction arrow
        
        ax.text(35, pitch_width - 73, '----------➤ Attacking Direction', va='center', ha='left',
                fontsize=15, color='black', family='monospace')

        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.axis('off')

        return fig
    


    ######################### Dribbles Map #####################################
    def create_dribbles_map(df, player_name, team_name):
    # Get team events
        all_events = df[df['teamName'] == team_name].copy()
        
        # Filter dribbles
        dribbles = all_events[
            (all_events['type/displayName'] == 'Take on') &
            (all_events['playerName'] == player_name)
        ]
        #successful = dribbles['outcomeType/displayName'] == 'Successful'
        # Create figure
        fig, ax = plt.subplots(figsize=(10.8, 10.8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        pitch_length, pitch_width = 105, 68
        pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
                    pitch_color='white', line_color='black', line_alpha=0.4, stripe=False)
        pitch.draw(ax=ax)

        # Logo
        logo = Image.open("Afican football analytics logo.png")
        logo = logo.resize((150, 150))
        ax_logo = fig.add_axes([0.70, 0.7, 0.20, 0.3])
        ax_logo.imshow(logo)
        ax_logo.axis('off')

        # Scale coordinates
        scaled_x = dribbles.x * pitch_length / 100
        scaled_y = dribbles.y * pitch_width / 100

        # Scatter plot with different colors for successful/unsuccessful
        successful = dribbles['outcomeType/displayName'] == 'Successful'
        pitch.scatter(scaled_x[successful], scaled_y[successful], s=200, 
                    color='#3fcf5d', edgecolors='black', linewidth=2.5,
                    alpha=1, ax=ax, zorder=3, label='Successful')
        pitch.scatter(scaled_x[~successful], scaled_y[~successful], s=200,
                    color='#EC313A', edgecolors='black', linewidth=2.5,
                    alpha=1, ax=ax, zorder=3, label='Unsuccessful')

    

        

        # Stats
        total = len(dribbles)
        successful = dribbles[dribbles['outcomeType/displayName'] == 'Successful']
        if len(dribbles) > 0:
            dribbles_success_rate = round((len(successful) / len(dribbles)) * 100, 2)
        else:
            dribbles_success_rate = 0.0  # or None, or 'N/A' depending on your use case        #st.write(successful)
        #st.write(len(dribbles))
        # Title and stats text
        ax.text(2, pitch_width + 25, "DRIBBLES MAP", color='black', size=25,
                ha='left', va='top', weight='bold', family='monospace')
        ax.text(2, pitch_width + 20, f"{player_name} - {league}" , color='black', size=16,
                ha='left', va='top', family='monospace')
        ax.text(2, pitch_width + 14, 'Dribble Outcome', fontsize=12,
                ha='left', va='center', color='black', family='monospace')
        
        ax.text(2.2, pitch_width + 6,
                f"{total} Attempted Dribbles",
                color='black', size=14, ha='left', va='top',
                family='monospace', weight='bold')
    

        ax.text(36, pitch_width + 6,
                f"{dribbles_success_rate} % Dribbles Success %",
                color='black', size=14, ha='left', va='top',
                family='monospace', weight='bold')
        
        
        
        # Add pitch direction arrow
        
        ax.text(35, pitch_width - 73, '----------➤ Attacking Direction', va='center', ha='left',
                fontsize=15, color='black', family='monospace')

        # Legend
        legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1.2),
                        ncol=2, frameon=False, fontsize=12)
        for text in legend.get_texts():
            text.set_color('black')
            text.set_family('monospace')

        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.axis('off')

        return fig


    ############################################# Key Passes #######################################################
    def create_key_passes_map(df, player_name, team_name):
        # Get player's key passes
        df = df[df['teamName'] == team_name].copy()

        key_passes = df[
            (df['playerName'] == player_name) &
            (df['keyPass'] == 1)
        ].copy()
        
        # Handle NaN values
        key_passes.loc[key_passes['endX'].isna(), 'endX'] = key_passes.loc[key_passes['endX'].isna(), 'x'] + 2
        key_passes.loc[key_passes['endY'].isna(), 'endY'] = key_passes.loc[key_passes['endY'].isna(), 'y'] + 2
        
        # Find assists
        events_df = df.copy()
        events_df['is_goal'] = events_df['type/displayName'] == 'Goal'
        events_df['next_is_goal'] = events_df['is_goal'].shift(-1)
        
        assists = events_df[
            (events_df['next_is_goal'] == True) &
            (events_df['type/displayName'] == 'Pass') &
            (events_df['outcomeType/displayName'] == 'Successful') &
            (events_df['playerName'] == player_name)
        ].copy()
        
        # Get progressive passes
        successful_passes = df[
            (df['type/displayName'] == 'Pass') &
            (df['outcomeType/displayName'] == 'Successful') &
            (df['playerName'] == player_name)
        ].copy()
        
        # Remove set pieces
        set_pieces = df[df.apply(lambda x: x.str.contains('Corner taken|Free kick taken', na=False).any(), axis=1)]
        successful_passes = successful_passes[~successful_passes['id'].isin(set_pieces['id'])]
        
        # Calculate progressive passes
        successful_passes['ppbegin'] = np.sqrt((100 - successful_passes['x'])**2 + 
                                            (50 - successful_passes['y'])**2)
        successful_passes['ppend'] = np.sqrt((100 - successful_passes['endX'])**2 + 
                                        (50 - successful_passes['endY'])**2)
        successful_passes['ratio'] = successful_passes['ppend'] / successful_passes['ppbegin']
        successful_passes['vertical_distance'] = abs(successful_passes['endY'] - successful_passes['y'])
        
        progressive_passes = successful_passes[
            (successful_passes['ratio'] < 0.75) &
            (successful_passes['vertical_distance'] >= 5)
        ]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10.8, 10.8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        pitch_length, pitch_width = 105, 68
        pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
                    pitch_color='white', line_color='black', line_alpha=0.4, stripe=False)
        pitch.draw(ax=ax)

        # Logo
        logo = Image.open("Afican football analytics logo.png")
        logo = logo.resize((150, 150))
        ax_logo = fig.add_axes([0.67, 0.685, 0.20, 0.3])
        ax_logo.imshow(logo)
        ax_logo.axis('off')

        # Scale coordinates
        for df in [progressive_passes, key_passes, assists]:
            df['scaled_x'] = df['x'] * pitch_length / 100
            df['scaled_y'] = df['y'] * pitch_width / 100
            df['scaled_endX'] = df['endX'] * pitch_length / 100
            df['scaled_endY'] = df['endY'] * pitch_width / 100

        # Draw arrows
        pitch.arrows(progressive_passes['scaled_x'], progressive_passes['scaled_y'],
                    progressive_passes['scaled_endX'], progressive_passes['scaled_endY'],
                    color='#FFCC00', width=4, alpha=1, zorder=3, ax=ax)

        pitch.arrows(key_passes['scaled_x'], key_passes['scaled_y'],
                    key_passes['scaled_endX'], key_passes['scaled_endY'],
                    color='#0F70BF', width=4, alpha=1, zorder=4, ax=ax)

        pitch.arrows(assists['scaled_x'], assists['scaled_y'],
                    assists['scaled_endX'], assists['scaled_endY'],
                    color='#1ADA89', width=4, alpha=1, zorder=5, ax=ax)

        # Add final third line
        ax.axvline(x=pitch_length*2/3, color='#9f9aa4', linestyle='--', alpha=0.7)

        # Add stats text
        ax.text(pitch_length *.137, pitch_width + 2, f"{len(key_passes) + len(assists)} Key Passes",
                color='#3d85c6', size=18, ha='center', family='monospace', weight='bold')
        ax.text(pitch_length *.105 , pitch_width + 7, f"{len(assists)} Assists",
                color='#3fcf5d', size=18, ha='center', family='monospace', weight='bold')
        ax.text(pitch_length*0.21, pitch_width + 12, f"{len(progressive_passes)} Progressive Passes",
                color='#FFCC00', size=18, ha='center', family='monospace', weight='bold')

        # Title
        ax.text(2, pitch_width + 25, "KEY & PROGRESSIVE PASSES", color='black', size=25,
                ha='left', va='top', weight='bold', family='monospace')
        ax.text(2, pitch_width + 19, f"{player_name} - {league}" , color='black', size=16,
                ha='left', va='top', family='monospace')
        
        # Add pitch direction arrow
        
        ax.text(35, pitch_width - 73, '----------➤ Attacking Direction', va='center', ha='left',
                fontsize=15, color='black', family='monospace')

        ax.set_ylim(0, pitch_width)
        ax.axis('off')

        return fig
    
    


    ######################################### Shots Map #################################################
    
    
    def create_shots_map(df, player_name, team_name):
        # --- Filter player shots ---
        shots = df[
            (df["playerName"] == player_name) &
            (df["teamName"] == team_name) &
            (df["type/displayName"].isin(["Attempt saved", "Miss", "Goal", "Post"]))
        ].copy()

        shots = shots.sort_values(["periodId", "minute", "second"]).reset_index(drop=True)
        shots["goal"] = shots["type/displayName"].apply(lambda x: "Goal" if x == "Goal" else "No Goal")
        

        # --- Create vertical pitch (attacking half only) ---
        pitch = VerticalPitch(
            half=True,
            pitch_color="white",
            line_color="black",
            stripe=False,
        )
        fig, ax = pitch.draw(figsize=(10.8, 10.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --- Add logo ---
        logo = Image.open("Afican football analytics logo.png").resize((150, 150))
        ax_logo = fig.add_axes([0.70, 0.8, 0.20, 0.3])
        ax_logo.imshow(logo)
        ax_logo.axis("off")

        # --- Plot arrows for on-target shots ---
        for _, row in shots.iterrows():
            if row["type/displayName"] in ["Attempt saved", "Goal"] and not row.get("isBlocked", False):
                x = (100 - row["y"] )* pitch.dim.pitch_width / 100
                y = (row["x"]+2) * pitch.dim.pitch_length / 100
                goal_x = row["goalMouthY"] * pitch.dim.pitch_width / 100
                goal_y = pitch.dim.pitch_length
                ax.add_patch(FancyArrowPatch(
                    (x, y), (goal_x, goal_y),
                    arrowstyle='->', color='#00ff00', linewidth=4, mutation_scale=12
                ))

        # --- Plot shots with shapes ---
        for goal_type, marker in {"Goal": "s", "No Goal": "o"}.items():
            subset = shots[shots["goal"] == goal_type]
            x_vals = (100 - subset["y"]) * pitch.dim.pitch_width / 100
            y_vals = (subset["x"]+2) * pitch.dim.pitch_length / 100
            ax.scatter(
                x_vals, y_vals,
                marker=marker,
                s=600,
                c="#af1615",
                edgecolors="black",
                linewidths=2,
                label=goal_type,
                zorder=3
            )

        # --- Summary stats ---
        total = len(shots)
        goals = len(shots[shots["goal"] == "Goal"])
        on_target = len(shots[
            ((shots["type/displayName"] == "Attempt saved") & (shots["isBlocked"] == 0)) |
            (shots["type/displayName"] == "Goal")
        ])

        # --- Title and annotation ---
        ax.text(2, pitch.dim.pitch_length + 17, "SHOTS MAP", color="black", size=27,
                ha="left", va="top", weight="bold", family="monospace")
        ax.text(2, pitch.dim.pitch_length + 13, f"{player_name} - {league}",
                color="black", size=19, ha="left", va="top", family="monospace")

        # --- Stats summary block ---
        ax.text(60, 78,
                f"{total} Total Shots \n{on_target} On Target \n{goals} Goals ",
                color="black", size=20, ha="left", va="top",
                family="monospace")

        # --- On-target arrow key (legend marker) ---
        ax.annotate(
            "", 
            xy=(34, 78), xytext=(34, 72),
            arrowprops=dict(arrowstyle="->", color="#00ff00", lw=4),
        )
        ax.text(
            37, 75, "On target", 
            ha="left", va="center", color="black", fontsize=20, family="monospace"
        )

        # --- Direction label ---
        ax.text(25, 45, '----------➤ Attacking Direction', va='center', ha='left',
                fontsize=15, color='black', family='monospace')

        # --- Custom legend ---
        legend_elements = [
            Line2D([0], [0], marker="s", color="w", label="Goal",
                markerfacecolor="#af1615", markeredgecolor="black", markersize=20),
            Line2D([0], [0], marker="o", color="w", label="No Goal",
                markerfacecolor="#af1615", markeredgecolor="black", markersize=20)
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.30, 0.33),
                ncol=1, frameon=False, fontsize=17)

        ax.axis("off")
        return fig



    

    ######################################## Defensive Actions Map ###############################################
    def create_defensive_actions_map(df, player_name, team_name):
        # Get team events and filter defensive actions
        df = df[df['teamName'] == team_name].copy()
        defensive_actions = df[
            (df['playerName'] == player_name) &
            (df['type/displayName'].isin([
                'Tackle', 'Challenge', 'Aerial', 'Ball recovery',
                'Clearance', 'Interception', 'BlockedPass', 'Foul'
            ]))
        ].copy()
        
        # Calculate successful actions
        successful_actions = defensive_actions[defensive_actions['outcomeType/displayName'] == 'Successful']
        success_rate = round((len(successful_actions) / len(defensive_actions)) * 100, 2)
        
        # Create bins for heatmap
        defensive_actions['xbin'] = pd.cut(defensive_actions['x'], bins=np.linspace(0, 100, 11), include_lowest=True)
        defensive_actions['ybin'] = pd.cut(defensive_actions['y'], bins=np.linspace(0, 100, 11), include_lowest=True)
        
        # Calculate average positions
        avg_x = defensive_actions['x'].mean()
        avg_y = defensive_actions['y'].mean()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10.8, 10.8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Draw pitch
        pitch_length, pitch_width = 105, 68
        pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
                    pitch_color='white', line_color='black', line_alpha=0.7, stripe=False)
        pitch.draw(ax=ax)

        # Add logo
        logo = Image.open("Afican football analytics logo.png")
        logo = logo.resize((150, 150))
        ax_logo = fig.add_axes([0.70, 0.7, 0.20, 0.3])
        ax_logo.imshow(logo)
        ax_logo.axis('off')
        
        # Create heatmap
        heatmap_data = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                mask = (
                    (defensive_actions['x'] >= i*20) & (defensive_actions['x'] < (i+1)*20) &
                    (defensive_actions['y'] >= j*20) & (defensive_actions['y'] < (j+1)*20)
                )
                heatmap_data[j, i] = mask.sum()
        
        # Plot heatmap with custom colormap
        colors = ['#ffffff', '#f4b9bc', '#ef9ea1', '#e9777b', '#df3238', '#dc2429']
        custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
        bin_statistic = pitch.bin_statistic(defensive_actions['x'], defensive_actions['y'], statistic='count', bins=(5, 5))
        
        pitch.heatmap(bin_statistic, ax=ax, cmap=custom_cmap, alpha=0.7)
        
        # Add average position line
        ax.axvline(x=avg_x * pitch_length/100, color='#FFCC00', linestyle='-', linewidth=4, alpha=1)

        # Add final third line
        ax.axvline(x=pitch_length*2/3, color='#9f9aa4', linestyle='--', alpha=0.7)


        # Add grid lines
        for x in range(20, 100, 20):
            ax.axvline(x=x * pitch_length/100, color='black', linestyle='-', linewidth=0.7, alpha=0.4)
        for y in range(20, 100, 20):
            ax.axhline(y=y * pitch_width/100, color='black', linestyle='-', linewidth=0.7, alpha=0.4)
        
        # Add stats text with matching style
        ax.text(pitch_length * 0.16, pitch_width + 12, f"{len(defensive_actions)} Total Actions",
                color='#3d85c6', size=18, ha='center', family='monospace', weight='bold')
        ax.text(pitch_length * 0.52, pitch_width + 12, f"{success_rate}% Success Rate",
                color='#3fcf5d', size=18, ha='center', family='monospace', weight='bold')
        
        # Title with matching style
        ax.text(2, pitch_width + 25, "DEFENSIVE ACTIONS", color='black', size=25,
                ha='left', va='top', weight='bold', family='monospace')
        ax.text(2, pitch_width + 19, f"{player_name} - {league}", color='black', size=16,
               ha='left', va='top', family='monospace')
        
        # Add pitch direction arrow
        ax.text(35, pitch_width - 73, '----------➤ Attacking Direction', va='center', ha='left',
                fontsize=15, color='black', family='monospace')
        
        # Add "Average" label above the average defensive line
        ax.text(
            x=avg_x * pitch_length / 100,    # Convert to pitch scale
            y=pitch_width + 2,               # Y > pitch width to place it just above
            s="Average",
            fontsize=16,                     # Similar to R's size=7
            color="#FFCC00",
            ha="center",
            va="bottom",
            family="monospace",         # Or "monospace" if unavailable
            weight="bold"
        )

        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.axis('off')
        
        return fig
    

    ##################################### Tackles Map #######################
    def create_tackles_map(df, player_name, team_name):
        # Filter data
        df = df[df['teamName'] == team_name].copy()
        tackles = df[(df['playerName'] == player_name) & (df['type/displayName'] == 'Tackle')].copy()
        successful_tackles = tackles[tackles['outcomeType/displayName'] == 'Successful']
        unsuccessful_tackles = tackles[tackles['outcomeType/displayName'] != 'Successful']
        tackles_rate = round((len(successful_tackles) / len(tackles)) * 100, 2) if len(tackles) > 0 else 0

        intrec = df[(df['playerName'] == player_name) & 
                    (df['type/displayName'].isin(['Ball recovery', 'Interception']))].copy()

        pitch_length, pitch_width = 105, 68
        

        fig, ax = plt.subplots(figsize=(10.8, 10.8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # --- Draw pitch ---
        pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
                    pitch_color='white', line_color='black', line_alpha=1, stripe=False)
        pitch.draw(ax=ax)

        # --- Plot tackles ---
        ax.scatter(successful_tackles['x'] * pitch_length / 100, successful_tackles['y'] * pitch_width / 100,
                marker='^', color='#25b5af', s=250, alpha=1, linewidth=4)

        ax.scatter(
        unsuccessful_tackles['x'] * pitch_length / 100,
        unsuccessful_tackles['y'] * pitch_width / 100,
        marker='^',  # same shape, triangle up
        facecolors='none',
        edgecolors='#25b5af',
        s=250,
        linewidth=4,
        alpha=1
    )

        #ax.scatter(intrec['x'] * pitch_length / 100, intrec['y'] * pitch_width / 100,
        #        marker='x', color='#FCE700', s=200, alpha=1, linewidth=2.5)

        # --- Add logo ---
        logo = Image.open("Afican football analytics logo.png").resize((150, 150))
        ax_logo = fig.add_axes([0.70, 0.7, 0.20, 0.3])
        ax_logo.imshow(logo)
        ax_logo.axis('off')

        # --- Titles ---
        ax.text(2, pitch_width + 25, "TACKLES  MAP", color='black', size=25,
                ha='left', va='top', weight='bold', family='monospace')

        ax.text(2, pitch_width + 19, f"{player_name} - {league}", color='black', size=16,
                ha='left', va='top', family='monospace')

        

        # --- Stat summaries ---
        ax.text(pitch_length * 0.155, pitch_width + 5, f"{len(tackles)} Total Tackles",
                color='#25b5af', size=18, ha='center', family='monospace', weight='bold')

        ax.text(pitch_length * 0.52, pitch_width + 5, f"{tackles_rate}% Success Rate",
                color='#25b5af', size=18, ha='center', family='monospace', weight='bold')

        #ax.text(pitch_length * 0.85, -5, f"{len(intrec)} Recoveries & Interceptions",
        #        color='#FCE700', size=18, ha='center', family='monospace', weight='bold')

        # --- Arrow direction ---
        ax.text(35, pitch_width - 73, '----------➤ Attacking Direction',
                va='center', ha='left', fontsize=15, color='black', family='monospace')

        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.axis('off')

        return fig


    def create_recoveries_map(df, player_name, team_name):
        # Filter data
        df = df[df['teamName'] == team_name].copy()
        tackles = df[(df['playerName'] == player_name) & (df['type/displayName'] == 'Tackle')].copy()

        intrec = df[(df['playerName'] == player_name) & 
                    (df['type/displayName'].isin(['Ball recovery', 'Interception']))].copy()

        pitch_length, pitch_width = 105, 68
        

        fig, ax = plt.subplots(figsize=(10.8, 10.8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # --- Draw pitch ---
        pitch = Pitch(pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
                    pitch_color='white', line_color='black', line_alpha=1, stripe=False)
        pitch.draw(ax=ax)



        ax.scatter(intrec['x'] * pitch_length / 100, intrec['y'] * pitch_width / 100,
                marker='x', color='#FCE700', s=350, alpha=1, linewidth=5)

        # --- Add logo ---
        logo = Image.open("Afican football analytics logo.png").resize((150, 150))
        ax_logo = fig.add_axes([0.70, 0.7, 0.20, 0.3])
        ax_logo.imshow(logo)
        ax_logo.axis('off')

        # --- Titles ---
        ax.text(2, pitch_width + 25, "RECOVERIES & INTERCEPTIONS MAP", color='black', size=22,
                ha='left', va='top', weight='bold', family='monospace')

        ax.text(2, pitch_width + 19, f"{player_name} - {league}", color='black', size=16,
                ha='left', va='top', family='monospace')

        

        ax.text(pitch_length * 0.303, pitch_width + 5, f"{len(intrec)} Recoveries & Interceptions",
                color='#FCE700', size=20, ha='center', family='monospace', weight='bold')

        # --- Arrow direction ---
        ax.text(35, pitch_width - 73, '----------➤ Attacking Direction',
                va='center', ha='left', fontsize=15, color='black', family='monospace')

        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.axis('off')

        return fig
    


    
    # Button to display the selected map
    if st.button("Show Player Map", key="show_map4"):
        if map_type == "Passes Received":
            fig = create_passes_received_map(df, selected_player, selected_team)
        elif map_type == "Dribbles Map":
            fig = create_dribbles_map(df, selected_player, selected_team)
        elif map_type == "Key & Progressive Passes Map":
            fig = create_key_passes_map(df, selected_player, selected_team)   
        elif map_type == "Shots Map":
            fig = create_shots_map(df, selected_player, selected_team)       
        elif map_type == "Defensive Actions Map":
            fig = create_defensive_actions_map(df, selected_player, selected_team)       
        elif map_type == "Tackles Map":
                        fig = create_tackles_map(df, selected_player, selected_team)      
        elif map_type == "Recoveries & Interceptions Map":
                        fig = create_recoveries_map(df, selected_player, selected_team) 
        st.pyplot(fig)
    
    with st.expander("Metric Glossary"):
        st.write("""
- **Overrun**: Heavy touch in a dribble.
- **Progressive Passes**: A pass that moves the ball closer to the opponent goal by 25% & at least 5 m vertically.
- **Lay-off**: A pass by a striker who has received the ball with back to goal.
- **Dispossessed**: Losing the ball under pressure.
- **GK Distribution**: Successful goalkeeper passes.
- **GK Launches**: Long balls launched forward.
- **Other Goals**: Goals not scored with foot or head.""")          
        