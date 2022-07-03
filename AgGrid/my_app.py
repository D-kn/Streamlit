import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

# colors = ['orange', 'blue', 'green', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chocolate', 
#          'coral', 'cornflowerblue', 'darkblue', 'crimson', 'darkcyan','darkgoldenrod', 'cyan','darkblue']

# file uploading
file_format = st.sidebar.radio('Select file format : ', options=['csv', 'excel'], key='file_format')
data = st.sidebar.file_uploader(label = '')

if not data is None:
    @st.cache
    def upload_file():
            if file_format=='csv':
                df = pd.read_csv(data)
            else:
                df = pd.read_excel(data)
            return df
            # return df.loc[:100, :]
        
    df = upload_file()

    st.write('''### Streamlit DataFrame type''')
    with st.expander('Data view'):
        st.dataframe(df)

    _funct = st.sidebar.radio(label='Functions', options=['Display', 'Highlight'])

    st.write('''### AgGrid  DataFrame type''')
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(enabled=True)
    gb.configure_default_column(editable=True, groupable=True)

    if _funct == 'Display':
        select_mode = st.radio('Selection Type', options=['Single', 'multiple'])
        gb.configure_selection(selection_mode=select_mode, use_checkbox=True)
        gridoptions = gb.build()
        grid_table = AgGrid(
                    df, 
                    gridOptions=gridoptions,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    height=450, 
                    allow_unsafe_jscode=True, 
                    theme='light'
                )
        select_row = grid_table["selected_rows"]
        st.write(select_row)

    if _funct == 'Highlight':
        c1, _ = st.columns([2, 1])
        col_choice = c1.selectbox(label='Select column', options=df.columns) 
        cellstyle_jscode = JsCode("""
            function(params){
                if (params.value == 2020){
                    return {
                        'color': 'black', 
                        'backgroundColor': 'orange'
                    }
                    }
                if (params.value == 2021){
                    return {
                        'color': 'black', 
                        'backgroundColor': 'blue'
                    }
                    }
                if (params.value == 2022){
                    return {
                        'color': 'black', 
                        'backgroundColor': 'green'
                    }
                    }
                };
        """)
        gb.configure_columns(col_choice, cellStyle=cellstyle_jscode)
        gridOptions = gb.build()
        grid_table = AgGrid(
                    df, 
                    gridOptions=gridOptions,
                    enable_enterprise_modules=True,
                    height=500,
                    width='100%',
                    theme="light",
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    reload_data=True,
                    allow_unsafe_jscode=True, 
                )
        select_row = grid_table["selected_rows"]
        st.write(select_row)