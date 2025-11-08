import pickle
import joblib

# Load original pipeline
with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Save compressed version
joblib.dump(pipeline, "pipeline_compressed.pkl", compress=3)