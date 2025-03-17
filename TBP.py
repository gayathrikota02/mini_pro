import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import cdist
from collections import Counter

# Step 1: Load and Preprocess MNIST
def preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize data to range [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape for model input
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)

# Step 2: Build Feature Extractor
def build_feature_extractor():
    input_layer = Input(shape=(28, 28, 1))
    x = Flatten()(input_layer)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(32, activation='relu')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Step 3: Novelty Detection using Extreme Value Theory (EVM) Implementation
def compute_evm_probabilities(features, known_features, threshold=0.5):
    distances = cdist(features, known_features, metric='euclidean')
    min_distances = np.min(distances, axis=1)

    # Convert distances to probabilities (1 - normalized distance)
    probabilities = np.exp(-min_distances / threshold)
    return probabilities

# Step 4: Incremental Learning (Clustering Unknown Classes)
def cluster_and_label(features, true_labels, min_cluster_size=5):
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    unique_labels = np.unique(cluster_labels)
    clusters = {label: features[cluster_labels == label] for label in unique_labels}

    # Filter clusters by size
    filtered_clusters = {label: data for label, data in clusters.items() if len(data) >= min_cluster_size}

    # Assign most frequent true label to each cluster
    cluster_to_true_label = {}
    for label in filtered_clusters:
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_true_labels = true_labels[cluster_indices]
        most_common_label = Counter(cluster_true_labels).most_common(1)[0][0]
        cluster_to_true_label[label] = most_common_label

    predicted_labels = np.array([cluster_to_true_label[label] for label in cluster_labels])
    return predicted_labels, len(filtered_clusters)

# Step 5: Evaluation Metrics
def evaluate_performance(true_labels, predicted_labels):
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    return precision, recall, f1

# Main Pipeline
def main():
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = preprocess_data()

    # Split into known (0-4) and unknown (5-9) classes
    known_classes = [0, 1, 2, 3, 4]
    unknown_classes = [5, 6, 7, 8, 9]

    x_known = x_train[np.isin(y_train, known_classes)]
    y_known = y_train[np.isin(y_train, known_classes)]

    x_unknown = x_test[np.isin(y_test, unknown_classes)]
    y_unknown = y_test[np.isin(y_test, unknown_classes)]

    # Build feature extractor
    feature_extractor = build_feature_extractor()
    feature_extractor.compile(optimizer='adam', loss='mse')

    # Extract features for known data
    known_features = feature_extractor.predict(x_known, verbose=0)

    # Classify unknown data and detect novelty
    unknown_features = feature_extractor.predict(x_unknown, verbose=0)
    novelty_probs = compute_evm_probabilities(unknown_features, known_features)

    # Threshold for novelty detection
    is_novel = novelty_probs < 0.5
    novel_features = unknown_features[is_novel]
    novel_true_labels = y_unknown[is_novel]

    # Incremental learning by clustering
    predicted_labels, num_clusters = cluster_and_label(novel_features, novel_true_labels)

    print(f"Number of new clusters discovered: {num_clusters}")

    # Evaluate
    precision, recall, f1 = evaluate_performance(novel_true_labels, predicted_labels)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

if __name__ == "__main__":
    main()
