from flask import Flask, request, jsonify
from transformers import pipeline

# Crear una instancia de Flask
app = Flask(__name__)

# Configurar el modelo de resumen usando facebookbart-large-cnn
summarizer = pipeline(summarization, model=facebookbart-large-cnn)

@app.route('resumir', methods=['POST'])
def resumir_texto()
    data = request.json
    texto = data.get('texto', '')
    
    if not texto
        return jsonify({error No se proporcionó texto para resumir}), 400
    
    # Generar el resumen
    resumen = summarizer(texto, max_length=130, min_length=30, do_sample=False)
    
    # Devolver el resumen en formato JSON
    return jsonify({resumen resumen[0]['summary_text']}), 200

if __name__ == '__main__'
    app.run(host='0.0.0.0', port=8080)
