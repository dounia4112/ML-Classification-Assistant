
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

def app():
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # store it to use in other pages
        st.session_state['uploaded_data'] = df
        # Dataset Preview
        st.write("### Dataset Preview")
        st.write(df.head())

        # Descriptive Statistics
        st.write("### Descriptive Statistics")
        st.write(df.describe())

        # Correlation Matrix
        st.write("### Correlation Matrix")
        if st.checkbox("Show Correlation Matrix"):
            corr_matrix = df.corr(numeric_only=True)
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="viridis")
            st.plotly_chart(fig_corr)

        # Advanced Visualizations
        st.write("### Advanced Visualizations")
        st.write("Select Columns for Visualization:")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) >= 2:
            x_axis = st.selectbox("X-Axis", num_cols)
            y_axis = st.selectbox("Y-Axis", [col for col in num_cols if col != x_axis])
            fig = px.scatter(df, x=x_axis, y=y_axis, title="Scatter Plot")
            st.plotly_chart(fig)
        else:
            st.warning("Not enough numeric columns for advanced visualizations.")

        st.write("#### Data Distribution")
        num_cols_without_outcome = df.drop(columns='Outcome').select_dtypes(include="number").columns
        column = st.selectbox("Features", num_cols_without_outcome)
        fig1, ax = plt.subplots()
        ax.hist(df[column], bins=20, color='blue', edgecolor='black')
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig1)


# if __name__ == "__main__":
#     app()
