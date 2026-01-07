import joblib

def load_model(model_path, encoder_path):
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder
