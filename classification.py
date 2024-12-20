
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def app():
    st.title("✅ Classification")
    
    # Check if data is available in session state
    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
        st.write("Data loaded for classification:")
        st.write(df.head())

        # Select target column
        target_column = df.columns[-1]  
    
        # Display the target column prominently
        st.markdown(f"""
        <div style="background-color: #f9f9f9; border-radius: 10px; padding: 10px; text-align: center; margin-bottom: 20px;">
            <h3 style="color: #0078D4;">Target Column: <span style="font-weight: bold;">{target_column}</span></h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Ensure the selected target column is not in the feature set
        features = df.drop(columns=[target_column])
        target = df[target_column]
        
        # Split data into training and testing sets
        test_size = st.slider("Test Size (as %)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)

        # Select Classification Algorithm
        algorithm = st.selectbox("Choose Algorithm", 
                                ["Logistic Regression", "Decision Tree","Random Forest", 'K-Nearest Neighbor'])

        if algorithm == "Logistic Regression":
            # Logistic Regression
            clf = LogisticRegression()
        elif algorithm == "Decision Tree":
            criterion = st.radio("Type of Criterion", ['gini', 'entropy'])
            max_depth = st.slider("Max Depth", 5, 20, 5)
            min_samples_leaf = st.slider("Minimum Samples Leaf", 10, 50, 20)
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        elif algorithm == "Random Forest":
            # Random Forest
            n_estimators = st.slider("Number of Estimators", 10, 50, 10)
            max_depth = st.slider("Max Depth", 2, 20, 10)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            n_neighbors = st.slider("Number of Neighbors", 2, 20, 5)
            metric = st.selectbox("Metric", ["euclidean", "minkowski", 'manhattan'])
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        # Train the model
        if st.button("Train Model"):
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            tab1, tab2= st.tabs(["Metrics", "Confusion Matrix"])
            with tab1:
                st.write("### Performance Metrics")
                report = classification_report(y_test, predictions, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                # st.write("### Classification Metrics")
                st.dataframe(report_df.style.format(precision=2))
            with tab2:
                cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
                st.write("### Confusion Matrix")
                fig3, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
                disp.plot(ax=ax, cmap="Blues", colorbar=False)
                st.pyplot(fig3)
    else:
        st.write("No data found. Please upload a file on the EDA page first.")

