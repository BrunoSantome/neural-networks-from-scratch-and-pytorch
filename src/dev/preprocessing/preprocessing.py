import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_preprocess_data_spacial_objects():
    """Load and preprocess the stellar object classification dataset.
    Returns:
        X_train_scaled (np.ndarray): Scaled training features.
        X_test_scaled (np.ndarray): Scaled testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
    """
    df = pd.read_csv(
        "C:/Users/bruno/OneDrive/Escritorio/Desktop/Repositories/Programming_math_Ai_Assessmment_Msc_Ai/dataset/Stellar_object_classification/SDSS_DR18.csv"
    )
    # They do not provide any useful information for the classification task
    df = df.drop(
        [
            "objid",
            "specobjid",
            "plate",
            "fiberid",
            "camcol",
            "run",
            "rerun",
            "field",
            "mjd",
        ],
        axis=1,
    )

    # df.isna().sum()

    df["class"] = df["class"].replace({"GALAXY": 0, "QSO": 1, "STAR": 2})

    df.head()
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


def load_and_preprocess_fish_classification_binary():
    df = pd.read_csv(
        "C:/Users/bruno/OneDrive/Escritorio/Desktop/Repositories/Programming_math_Ai_Assessmment_Msc_Ai/dataset/fish_classifier/fish_data.csv"
    )
    df.head()
    df["species"].unique()

    df["Species"] = LabelEncoder().fit_transform(df["species"])
    X = df.drop(["species", "Species"], axis=1)
    y = df["Species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test
