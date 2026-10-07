import os
import time
import statistics
import joblib
import psutil

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Load dataset
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

process = psutil.Process(os.getpid())

for name, model in models.items():

    # Warm-up
    model.fit(X_train, y_train)

    # Training time: 5 runs
    training_times = []

    for _ in range(5):
        start = time.perf_counter()

        model.fit(X_train, y_train)

        end = time.perf_counter()
        training_times.append(end - start)

    training_median = statistics.median(training_times)

    # Single sample
    sample = X_test[0].reshape(1, -1)

    # Inference latency: 100 runs
    inference_times = []

    for _ in range(100):
        start = time.perf_counter()

        model.predict(sample)

        end = time.perf_counter()
        inference_times.append((end - start) * 1000)

    inference_median = statistics.median(inference_times)

    # Save model
    filename = name.lower().replace(" ", "_") + ".joblib"
    path = os.path.join("lab01", "results", filename)

    joblib.dump(model, path)

    model_size_kb = os.path.getsize(path) / 1024

    # Memory
    memory_bytes = process.memory_info().rss
    memory_kb = memory_bytes / 1024

    print()
    print(name)
    print(f"Training time median: {training_median:.6f} seconds")
    print(f"Inference latency median: {inference_median:.6f} ms")
    print(f"Model size: {model_size_kb:.2f} KB")
    print(f"Memory: {memory_kb:.2f} KB")