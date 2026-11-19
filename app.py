#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 14:43:34 2025

@author: layfi
"""


import streamlit as st
import pandas as pd

df = pd.read_csv("titanic.csv")

st.title("Titanic Passenger Insights")

st.subheader("Dataset Dimensions")
st.write("Number of rows:", df.shape[0])
st.write("Number of columns:", df.shape[1])

st.subheader("Column Names")
st.write(df.columns.tolist())

st.subheader("Data Types")
st.write(df.dtypes)

st.subheader("First Five Rows")
st.dataframe(df.head())

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Selected Rows")
st.dataframe(df.iloc[0:5])





# --------------------------------
# TASK 2: HANDLE MISSING VALUES
# --------------------------------

st.header("2. Handle Missing Values")

# Create a copy of original dataset
cleaned_df = df.copy()

# Show missing values before cleaning
st.subheader("Missing Values Before Cleaning")

st.write(cleaned_df[["age", "embarked"]].isnull().sum())

# Fill missing age values with median
age_median = cleaned_df["age"].median()
cleaned_df["age"] = cleaned_df["age"].fillna(age_median)

# Fill missing embarked values with mode
embarked_mode = cleaned_df["embarked"].mode()[0]
cleaned_df["embarked"] = cleaned_df["embarked"].fillna(embarked_mode)

# Show missing values after cleaning
st.subheader("Missing Values After Cleaning")

st.write(cleaned_df[["age", "embarked"]].isnull().sum())

# Show cleaned dataset
st.subheader("Cleaned Dataset")

st.dataframe(cleaned_df)
