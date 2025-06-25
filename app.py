import streamlit as st

st.set_page_config(page_title='', layout='wide') 

main_page = st.Page("main_page.py", title="Analysis", icon="📊")
page_2 = st.Page("page_2.py", title="GPU", icon="🖥️")
page_3 = st.Page("page_3.py", title="API", icon="⚙️")
page_4 = st.Page("page_4.py", title="Strategy", icon="🧠")

pg = st.navigation([main_page, page_2, page_3, page_4])

pg.run()
