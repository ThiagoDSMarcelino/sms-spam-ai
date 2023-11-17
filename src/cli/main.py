import joblib

def load_model_and_encoder(model_path, encoder_path):
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

def get_user_input():
    while True:
        email_content = input("Enter an SMS content (type 'stop' to end):\n").lower()

        if email_content == 'stop':
            print('Stopping the program.')
            break

        yield email_content

def predict_sms_category(model, encoder, email_content):
    try:
        prediction = model.predict([email_content])
        category = encoder.inverse_transform(prediction)[0] 
        return category
    except Exception as e:
        print(f"Error predicting category: {e}")
        return None

if __name__ == "__main__":
    model_path = '../data/model.sav'
    encoder_path = '../data/encoder.sav'

    model, encoder = load_model_and_encoder(model_path, encoder_path)

    for sms_content in get_user_input():
        category = predict_sms_category(model, encoder, sms_content)

        if category is not None:
            print(f'This sms is likely to be: {category}')
