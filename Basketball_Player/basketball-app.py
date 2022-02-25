import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns


st.title('NBA Player Stats Explorer')

st.markdown("""
This app performs simple webscraping of NBA player stats data!
* **Python librairies:** base64, pandas, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/)
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2023))))

# Web Scraping for NBA player stats
@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    raw = raw.astype(str)
    # playerstats = raw.drop(['Rk', 'FG%', '3P%', '2P%', 'eFG%', 'FT%'], axis=1)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = load_data(selected_year)


# Siderbar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team[:10])

# Siderbar - Position selection
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

# Show Dataset and other elements
st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)


# Download NBA Dataset
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)


# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.subheader('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')
    corr = df.corr()
    fig = plt.figure(figsize=(16, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask,center=0,
           cmap='RdBu', linewidths=1,
           annot=True, fmt=".2f", vmin=-1, vmax=1)
    st.pyplot(fig)


# if st.button('Intercorrelation Heatmap'):
#     st.header('Intercorrelation Matrix Heatmap')
#     df_selected_team.to_csv('output.csv', index=False)
#     df = pd.read_csv('output.csv')

#     corr = df.corr()
#     mask = np.zeros_like(corr)
#     mask[np.triu_indices_from(mask)] = True
#     with sns.axes_style('whitegrid'):
#         f, ax = plt.subplots(figsize=(8, 5))
#         ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
#     st.pyplot(f)