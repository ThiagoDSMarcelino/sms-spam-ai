from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/', methods=['POST'])
def check_email():
    if request.method != 'POST':
        return 'This endpoint only supports POST requests.'
    
    json_data = request.get_json()
    if 'data' not in json_data:
        return jsonify({'error': 'No data field provided in the JSON payload'}), 400
    
    data = jsonify({'data': json_data['data']})
    
    model = joblib.load('src\data\model.sav')
    encoder = joblib.load(encode_path)
    
    predict = model.predict(data)
    
    return encoder.inverse_transform(predict)

if __name__ == '__main__':
    app.run(debug=True)