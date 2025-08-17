from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from tensorflow import keras

import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs("model", exist_ok=True)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("__________LOADING TITANIC DATASET__________")
DATA_PATH = "data/titanic.csv"
ttd = pd.read_csv(DATA_PATH)
print(ttd.head())

print("\n__________PREPROCESSING DATA__________")
X = ttd.drop(columns="Survived")
y = ttd["Survived"].values

numfts = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
numfill = SimpleImputer(strategy="median")
X_num = numfill.fit_transform(X[numfts])
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

catfts = ["Sex", "Embarked"]
catfill = SimpleImputer(strategy="most_frequent")
X_cat = catfill.fit_transform(X[catfts])
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_cat)

X_cleaned = np.hstack([X_num_scaled, X_cat_encoded])

print(f"Shape of processed data: {X_cleaned.shape}")

# ---Save Cleaned Data
prepc_ttd = pd.DataFrame(
    X_cleaned,
    columns=numfts + list(encoder.get_feature_names_out(catfts))
)
prepc_ttd["Survived"] = y  # add the target back
prepc_ttd.to_csv("data/cleaned_titanic.csv", index=False)
print("\n➖CLEANED DATA 'cleaned_titanic.csv' SAVED")

# ---Save Preprocessor
joblib.dump({
    "num_imputer": numfill,
    "scaler": scaler,
    "cat_imputer": catfill,
    "encoder": encoder,
    "num_features": numfts,
    "cat_features": catfts
}, "model/titanic_preprocessor.joblib")
print("\n➖PREPROCESSOR MODEL 'titanic_preprocessor.joblib' SAVED")

# ---Train-Test Split
print("\n__________SPLIT TRAIN-TEST DATA__________")
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ---Build Model
print("\n___________BUILDING NEURAL NETWORK MODEL__________")
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  # binary output
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ---Train Model with Early Stopping
print("\n____________TRAINING MODEL__________")
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)
print("___________TRAINING COMPLETE__________")

# ---Evaluate Model
print("\n____________EVALUATING MODEL__________")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

# ---Plot Training History
plt.figure(figsize=(12,4))

# Plot accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy over Epochs")

# Plot loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over Epochs")

plt.show()

# ---Save Model
os.makedirs("model", exist_ok=True)
model.save("model/titanic_model.h5")
print("\n➖MODEL 'titanic_model.h5' SAVED")

# ---Example Prediction
print("\n____________EXAMPLE PREDICTION__________")
sample = X_test[5].reshape(1, -1)
prediction = model.predict(sample)[0][0]
print(f"Raw prediction (probability of survival): {prediction:.4f}")
print(f"Predicted class: {int(prediction >= 0.5)} (1=Survived, 0=Did not survive)")
print(f"True label: {y_test[5]}")