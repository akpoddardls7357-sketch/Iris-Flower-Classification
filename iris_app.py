import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load larger dataset
iris = pd.read_csv("Iris_large.csv")  # Your augmented dataset

# Features and labels
X = iris.drop("Species", axis=1)
y = iris["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "iris_model_large.pkl")

# Streamlit UI
st.title("🌸 Iris Flower Classifier (with Larger Dataset)")
st.write("Predict Iris species by entering values or uploading a dataset.")

# --- Single Prediction with Sliders ---
st.subheader("🔹 Predict Single Flower")
sepal_length = st.slider("Sepal Length (cm)", float(X["SepalLengthCm"].min()), float(X["SepalLengthCm"].max()), 5.0)
sepal_width = st.slider("Sepal Width (cm)", float(X["SepalWidthCm"].min()), float(X["SepalWidthCm"].max()), 3.0)
petal_length = st.slider("Petal Length (cm)", float(X["PetalLengthCm"].min()), float(X["PetalLengthCm"].max()), 1.5)
petal_width = st.slider("Petal Width (cm)", float(X["PetalWidthCm"].min()), float(X["PetalWidthCm"].max()), 0.2)

if st.button("Predict Single Flower"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.success(f"🌿 Predicted Species: **{prediction}**")

# --- Batch Prediction with CSV Upload ---
st.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV with columns: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm", type=["csv"])

if uploaded_file is not None:
    unseen_data = pd.read_csv(uploaded_file)

    # Ensure correct columns
    required_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    if all(col in unseen_data.columns for col in required_columns):
        predictions = model.predict(unseen_data[required_columns])
        unseen_data["Predicted_Species"] = predictions

        st.write("✅ Predictions Completed")
        st.dataframe(unseen_data)

        # Option to download results
        csv = unseen_data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions as CSV", csv, "predictions.csv", "text/csv")

        # --- Interactive Visualization ---
        st.subheader("📊 Interactive Visualization of Predictions")

        feature_options = required_columns
        x_axis = st.selectbox("Choose X-axis feature", feature_options, index=0)
        y_axis = st.selectbox("Choose Y-axis feature", feature_options, index=1)

        fig, ax = plt.subplots()
        for species, color in zip(unseen_data["Predicted_Species"].unique(), ["red", "blue", "green"]):
            subset = unseen_data[unseen_data["Predicted_Species"] == species]
            ax.scatter(subset[x_axis], subset[y_axis], label=species, color=color, alpha=0.6)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.legend()
        st.pyplot(fig)

    else:
        st.error(f"CSV must contain columns: {required_columns}")

# --- Show Accuracy ---
st.subheader("✅ Model Accuracy")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Accuracy on test data: **{acc*100:.2f}%**")

# --- Dataset Visualization ---
st.subheader("📊 Dataset Distribution")
fig, ax = plt.subplots()
iris["Species"].value_counts().plot(kind="bar", ax=ax, color=["#FF9999","#66B2FF","#99FF99"])
ax.set_ylabel("Count")
ax.set_xlabel("Species")
st.pyplot(fig)
