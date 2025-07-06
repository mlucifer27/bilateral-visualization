import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from d3blocks import D3Blocks
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

YEAR_START = 1945
YEAR_END = 2024
COLORMAP = 'RdYlGn'
HEATMAP_FIGSIZE = [1000, 1000]
HEATMAP_ZMIN = 0
HEATMAP_ZMAX = 10
NETWORK_THRESHOLD_DEFAULT = 7

STATE_A_COL = "State A"
STATE_B_COL = "State B"
YEAR_COL = "Year"
OVERALL_COL = "Overall Relationship"
ECONOMIC_COL = "Economic Relationship"
SECURITY_COL = "Security Relationship"
POLITICAL_COL = "Political Relationship"
CULTURAL_COL = "Cultural Relationship"
RELATIONSHIP_COLS = [
    OVERALL_COL, ECONOMIC_COL, SECURITY_COL, POLITICAL_COL, CULTURAL_COL
]
ALL_DATA_COLS = [STATE_A_COL, STATE_B_COL, YEAR_COL] + RELATIONSHIP_COLS

st.set_page_config(page_title="Bilateral Relationship Analysis (1945-2024)", layout="wide")

st.title("Bilateral Relationship Analysis (1945-2024)")
COUNTRIES_PATH = Path("data/countries.txt")
with open(COUNTRIES_PATH, "r", encoding="utf-8") as f:
    countries = [line.strip() for line in f if line.strip()]
country_idx = {country: idx for idx, country in enumerate(countries)}
year = st.slider("Select Year", YEAR_START, YEAR_END, YEAR_END)
colormap = COLORMAP

@st.cache_data
def load_year_data(year, csv_path="data/overall.csv"):
    data = []
    for chunk in pd.read_csv(csv_path, usecols=ALL_DATA_COLS, chunksize=100000):
        filtered = chunk[chunk[YEAR_COL] == year]
        if not filtered.empty:
            data.append(filtered)
    if data:
        df = pd.concat(data, ignore_index=True)
        for col in RELATIONSHIP_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    else:
        return pd.DataFrame(columns=ALL_DATA_COLS)

@st.cache_data
def load_full_data(csv_path="data/overall.csv"):
    df = pd.read_csv(csv_path, usecols=ALL_DATA_COLS)
    for col in RELATIONSHIP_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def show_heatmaps(year, colormap, countries, country_idx):
    st.markdown("""<hr style='margin:40px 0 20px 0; border:1px solid #eee;'>""", unsafe_allow_html=True)
    st.header('Heatmaps')
    st.caption('Visualize the bilateral relationship matrix for the selected year. Hover for details.')
    year_data = load_year_data(year)
    if year_data.empty:
        st.warning('No data available for the selected year.')
        return
    heatmap_matrix = np.full((len(countries), len(countries)), np.nan)
    for _, row in year_data.iterrows():
        a, b = row[STATE_A_COL], row[STATE_B_COL]
        try:
            score = float(row[OVERALL_COL])
        except Exception:
            score = np.nan
        if a in country_idx and b in country_idx:
            i, j = country_idx[a], country_idx[b]
            heatmap_matrix[i, j] = score
            heatmap_matrix[j, i] = score
    np.fill_diagonal(heatmap_matrix, 0)
    heatmap_df = pd.DataFrame(heatmap_matrix, index=countries, columns=countries)
    st.subheader('Plotly Heatmap (Interactive)')
    detail_lookup = {}
    for _, row in year_data.iterrows():
        a, b = row[STATE_A_COL], row[STATE_B_COL]
        try:
            overall = float(row[OVERALL_COL])
            economic = float(row[ECONOMIC_COL])
            security = float(row[SECURITY_COL])
            political = float(row[POLITICAL_COL])
            cultural = float(row[CULTURAL_COL])
        except Exception:
            overall = economic = security = political = cultural = None
        detail = f"<b>{a} - {b}</b><br>Overall: {overall}<br>Economic: {economic}<br>Security: {security}<br>Political: {political}<br>Cultural: {cultural}"
        detail_lookup[(a, b)] = detail
        detail_lookup[(b, a)] = detail
    hover_text = np.empty((len(countries), len(countries)), dtype=object)
    for i, a in enumerate(countries):
        for j, b in enumerate(countries):
            if i == j:
                hover_text[i, j] = f"<b>{a} - {b}</b><br>Self-relation (0)"
            else:
                hover_text[i, j] = detail_lookup.get((a, b), f"<b>{a} - {b}</b><br>No data")
    plotly_z = heatmap_df.values
    plotly_x = countries
    plotly_y = countries
    fig = go.Figure(data=go.Heatmap(
        z=plotly_z,
        x=plotly_x,
        y=plotly_y,
        colorscale=colormap,
        zmin=HEATMAP_ZMIN,
        zmax=HEATMAP_ZMAX,
        colorbar=dict(title='Relationship'),
        text=hover_text,
        hoverinfo='text',
        showscale=True
    ))
    fig.update_layout(
        width=HEATMAP_FIGSIZE[0],
        height=HEATMAP_FIGSIZE[1],
        xaxis=dict(
            tickmode='array',
            tickvals=countries,
            ticktext=countries,
            tickangle=90
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=countries,
            ticktext=countries
        ),
        margin=dict(l=100, r=100, t=50, b=100)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('D3Blocks Heatmap - Clusterable')
    st.caption('Cluster and explore the relationship matrix using d3blocks.')
    d3 = D3Blocks()
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "heatmap.html")
        d3.heatmap(
            heatmap_df,
            title=f"Bilateral Relationship - {year}",
            cmap=colormap,
            figsize=HEATMAP_FIGSIZE,
            showfig=False,
            filepath=html_path
        )
        with open(html_path, "r", encoding="utf-8") as f:
            heatmap_html = f.read()
        st.components.v1.html(heatmap_html, height=HEATMAP_FIGSIZE[1])

def show_topn_widget(year, countries):
    st.markdown("""<hr style='margin:40px 0 20px 0; border:1px solid #eee;'>""", unsafe_allow_html=True)
    st.header('Top Strongest/Weakest Relationships')
    st.caption('See the strongest and weakest bilateral relationships for a selected year.')
    col1, col2 = st.columns(2)
    with col1:
        topn_year = st.slider('Select Year for Top-N', YEAR_START, YEAR_END, year)
    with col2:
        N = st.number_input('N (Top/Bottom)', min_value=1, max_value=50, value=10)
    topn_data = load_year_data(topn_year)
    mask = topn_data[STATE_A_COL] != topn_data[STATE_B_COL]
    topn_data = topn_data[mask]
    topn_data = topn_data.dropna(subset=[OVERALL_COL])
    if topn_data.empty:
        st.info('No bilateral relationship data for the selected year.')
    else:
        strongest = topn_data.sort_values(OVERALL_COL, ascending=False).head(N)
        weakest = topn_data.sort_values(OVERALL_COL, ascending=True).head(N)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f'Top {N} Strongest Relationships ({topn_year})')
            st.dataframe(strongest[[STATE_A_COL, STATE_B_COL] + RELATIONSHIP_COLS]
                .reset_index(drop=True)
                .rename_axis('Rank')
                .set_index(pd.Index(range(1, N+1))))
        with col2:
            st.subheader(f'Top {N} Weakest Relationships ({topn_year})')
            st.dataframe(weakest[[STATE_A_COL, STATE_B_COL] + RELATIONSHIP_COLS]
                .reset_index(drop=True)
                .rename_axis('Rank')
                .set_index(pd.Index(range(1, N+1))))

def show_distribution_widget(year, countries):
    st.markdown("""<hr style='margin:40px 0 20px 0; border:1px solid #eee;'>""", unsafe_allow_html=True)
    st.header('Country Relationship Distribution')
    st.caption('Explore the distribution of relationship scores for a country or all countries in a given year.')
    col1, col2 = st.columns(2)
    with col1:
        dist_year = st.slider('Select Year for Distribution', YEAR_START, YEAR_END, year, key='dist_year')
    with col2:
        country_options = ['All Countries'] + countries
        dist_country = st.selectbox('Select Country', country_options, key='dist_country')
    dist_data = load_year_data(dist_year)
    dist_data = dist_data[dist_data[STATE_A_COL] != dist_data[STATE_B_COL]]
    dist_data[OVERALL_COL] = pd.to_numeric(dist_data[OVERALL_COL], errors='coerce')
    if dist_data.empty:
        st.info('No data for the selected year.')
    else:
        if dist_country == 'All Countries':
            values = dist_data[OVERALL_COL].dropna()
            title = f'All Countries ({dist_year})'
        else:
            mask = (dist_data[STATE_A_COL] == dist_country) | (dist_data[STATE_B_COL] == dist_country)
            values = dist_data[mask][OVERALL_COL].dropna()
            title = f'{dist_country} ({dist_year})'
        if values.empty:
            st.info('No relationship data for the selected country/year.')
        else:
            fig_dist = px.histogram(values, nbins=10, range_x=[HEATMAP_ZMIN, HEATMAP_ZMAX], title=f'Relationship Score Distribution: {title}')
            fig_dist.update_layout(xaxis_title='Overall Relationship', yaxis_title='Count', bargap=0.1)
            st.plotly_chart(fig_dist, use_container_width=True)
            fig_box = px.box(values, points='all', title=f'Relationship Score Boxplot: {title}')
            fig_box.update_layout(yaxis_title='Overall Relationship', xaxis_title='')
            st.plotly_chart(fig_box, use_container_width=True)
            summary_stats = values.describe().rename({
                'count': 'Count',
                'mean': 'Mean',
                'std': 'Std',
                'min': 'Min',
                '25%': '25%',
                '50%': 'Median',
                '75%': '75%',
                'max': 'Max'
            })
            st.write('**Summary Statistics:**')
            st.dataframe(summary_stats.to_frame().T)

def show_network_widget(year, countries, country_idx):
    st.markdown("""<hr style='margin:40px 0 20px 0; border:1px solid #eee;'>""", unsafe_allow_html=True)
    st.header('Country-to-Network Graph')
    st.caption('Visualize the network of countries with strong relationships for a selected year.')
    col1, col2 = st.columns(2)
    with col1:
        net_year = st.slider('Select Year for Network', YEAR_START, YEAR_END, year, key='net_year')
    with col2:
        threshold = st.slider('Relationship Threshold', HEATMAP_ZMIN, HEATMAP_ZMAX, NETWORK_THRESHOLD_DEFAULT, key='net_threshold')
    net_data = load_year_data(net_year)
    net_data = net_data[net_data[STATE_A_COL] != net_data[STATE_B_COL]]
    net_data[OVERALL_COL] = pd.to_numeric(net_data[OVERALL_COL], errors='coerce')
    net_edges = net_data[net_data[OVERALL_COL] >= threshold]
    st.subheader('Plotly Network Graph')
    G = nx.Graph()
    for _, row in net_edges.iterrows():
        G.add_edge(row[STATE_A_COL], row[STATE_B_COL], weight=row[OVERALL_COL])
    pos = nx.spring_layout(G, seed=42, k=0.3)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='top center',
        hoverinfo='text', marker=dict(size=10, color='skyblue', line_width=2))
    fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=f'Country Network (Threshold ≥ {threshold}, {net_year})'))
    st.plotly_chart(fig_net, use_container_width=True)
    st.subheader('D3Blocks Network Graph')
    st.caption('Cluster and explore the country network using d3blocks.')
    if not net_edges.empty:
        net_edges_simple = net_edges[[STATE_A_COL, STATE_B_COL, OVERALL_COL]].copy()
        net_edges_simple.columns = ['source', 'target', 'weight']
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "network.html")
            d3 = D3Blocks()
            d3.d3graph(
                net_edges_simple,
                showfig=False,
                filepath=html_path
            )
            with open(html_path, "r", encoding="utf-8") as f:
                net_html = f.read()
            st.components.v1.html(net_html, height=600)
    else:
        st.info('No relationships above threshold for this year.')

def show_pairwise_widget(countries):
    st.markdown("""<hr style='margin:40px 0 20px 0; border:1px solid #eee;'>""", unsafe_allow_html=True)
    st.header('Pairwise Comparison')
    st.caption('Compare all relationship scores for any two countries, for a specific year or over time.')
    col1, col2, col3 = st.columns(3)
    with col1:
        pair_country_a = st.selectbox('Country A', countries, key='pair_a')
    with col2:
        pair_country_b = st.selectbox('Country B', countries, key='pair_b')
    with col3:
        pair_year_options = ['All Years'] + [str(y) for y in range(YEAR_START, YEAR_END+1)]
        pair_year = st.selectbox('Year', pair_year_options, key='pair_year')
    full_data = load_full_data()
    mask = (
        ((full_data[STATE_A_COL] == pair_country_a) & (full_data[STATE_B_COL] == pair_country_b)) |
        ((full_data[STATE_A_COL] == pair_country_b) & (full_data[STATE_B_COL] == pair_country_a))
    )
    pair_data = full_data[mask]
    if pair_year != 'All Years':
        pair_data = pair_data[pair_data[YEAR_COL] == int(pair_year)]
        if not pair_data.empty:
            st.write(f'Relationship scores for {pair_country_a} and {pair_country_b} in {pair_year}:')
            st.dataframe(pair_data.reset_index(drop=True))
        else:
            st.info('No data for this pair and year.')
    else:
        if not pair_data.empty:
            st.write(f'Relationship scores for {pair_country_a} and {pair_country_b} ({YEAR_START}-{YEAR_END}):')
            pair_data = pair_data.sort_values(YEAR_COL)
            for col in RELATIONSHIP_COLS:
                st.subheader(f'{col} Over Time')
                st.line_chart(pair_data[[YEAR_COL, col]].set_index(YEAR_COL), height=200, use_container_width=True)
        else:
            st.info('No data for this pair.')

show_heatmaps(year, colormap, countries, country_idx)
show_topn_widget(year, countries)
show_distribution_widget(year, countries)
show_network_widget(year, countries, country_idx)
show_pairwise_widget(countries)
