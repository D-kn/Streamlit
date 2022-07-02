import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


# file uploading
file_format = st.sidebar.radio('Select file format : ', options=['csv', 'excel'], key='file_format')
data = st.sidebar.file_uploader(label = '')

if not data is None:
    @st.cache
    def upload_file():
        if file_format == 'csv':
            df = pd.read_csv(data)
        else:
            df = pd.read_excel(data)
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
                            theme='light')

    select_row = grid_table["selected_rows"]
    st.write(select_row)

