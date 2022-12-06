import streamlit as st

import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

def readdata():
    df = pd.read_csv('UdemyCleanedTitle.csv')
    return df



def getcosinemat(df):
    countvect = CountVectorizer()
    cvmat = countvect.fit_transform(df['Clean_title'])
    return cvmat


def cleantitle(df):
    df['Clean_title'] = df['course_title'].apply(nfx.remove_stopwords)

    df['Clean_title'] = df['Clean_title'].apply(nfx.remove_special_characters)

    return df

def cosinesimat(cv_math):
    return cosine_similarity(cv_math)