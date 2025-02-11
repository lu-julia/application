"""
Prédiction de la survie d'un individu sur le Titanic
"""

import os
import argparse
from dotenv import load_dotenv

import pandas as pd

from src.data.preprocess import split_and_count, split_train_test
from src.models.models import create_pipeline
from src.evaluation.train_evaluate import evaluate_model


# ENVIRONMENT CONFIGURATION ---------------------------

load_dotenv()

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
args = parser.parse_args()

n_trees = args.n_trees
print(f"Valeur de n_trees = {n_trees}")
JETON_API = os.environ["JETON_API"]
MAX_DEPTH = None
MAX_FEATURES = "sqrt"

if JETON_API.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")



# IMPORT ET EXPLORATION DONNEES --------------------------------

TrainingData = pd.read_csv("data.csv")


# Usage example:
ticket_count = split_and_count(TrainingData, "Ticket", "/")
name_count = split_and_count(TrainingData, "Name", ",")


# SPLIT TRAIN/TEST --------------------------------

X_train, X_test, y_train, y_test = split_train_test(TrainingData, test_size=0.1)


# PIPELINE ----------------------------

# Create the pipeline
pipe = create_pipeline(
    n_trees, max_depth=MAX_DEPTH, max_features=MAX_FEATURES
)


# ESTIMATION ET EVALUATION ----------------------

pipe.fit(X_train, y_train)


# Evaluate the model
score, matrix = evaluate_model(pipe, X_test, y_test)
print(f"{score:.1%} de bonnes réponses sur les données de test pour validation")
print(20 * "-")
print("matrice de confusion")
print(matrix)
