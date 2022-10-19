import pandas as pd
import streamlit as st

@st.cache(suppress_st_warning=True)
def read_data(path):
    return pd.read_csv(path, error_bad_lines=False, encoding='utf-8')


def head():
    st.markdown("# Book Recommendation App")
    st.caption("By Diego Torres", unsafe_allow_html=True)
