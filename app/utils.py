import numpy as np
import pandas as pd
import scipy.stats
import streamlit as st

@st.cache(suppress_st_warning=True)
def read_data(path):
    return pd.read_csv(path)

def head():
    st.markdown("""
        <ht style='text-align: center; margin bottom: -35px;'>
        Book Recommendation App
        </h1>
    """, unsafe_allow_html=True
        )

    st.caption("""
        <p style='text-align: center'>
        by Diego Torres
        </p>
    """, unsafe_allow_html=True)

    st.write(
        "Click here"
    )

def body(sample):
    st.markdown('----')