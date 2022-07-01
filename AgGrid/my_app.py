from turtle import up
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


# file uploading
file = st.sidebar.file_uploader("Upload a file")

if not file is None:
    @st.cache
    def upload_file():
        df = pd.read_csv(file)
        return df.loc[:100, :]

    df = upload_file()

    st.write('''### Streamlit DataFrame type''')
    st.dataframe(df)

    st.write('''### AgGrid  DataFrame type''')
    grid_df = GridOptionsBuilder.from_dataframe(df)
    grid_df.configure_pagination(enabled=True)
    grid_df.configure_default_column(editable=True, groupable=True)

    select_mode = st.radio('Selection Type :', options=['Single', 'multiple'])
    grid_df.configure_selection(selection_mode=select_mode, use_checkbox=True)
    gridoptions = grid_df.build()
    grid_table = AgGrid(df, gridOptions=gridoptions,
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            height=450, 
                            allow_unsafe_jscode=True, 
                            theme='blue')

    sel_row = grid_table["selected_rows"]
    st.write(sel_row)

