from ml_classification import train_models

results = train_models()

print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

print(f"Accuracy: {results['logistic']['accuracy']:.4f}")

print("\nClassification Report")
print(results["logistic"]["report"])

print("\nConfusion Matrix")
print(results["logistic"]["matrix"])

print("\n")

print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)

print(f"Accuracy: {results['random_forest']['accuracy']:.4f}")

print("\nClassification Report")
print(results["random_forest"]["report"])

print("\nConfusion Matrix")
print(results["random_forest"]["matrix"])

print("\nFeature Importance")

print(results["random_forest"]["feature_importance"])
