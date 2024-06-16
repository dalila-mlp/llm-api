from flask import Flask, request, jsonify
from llm import lire_fichier, lire_json_models, interroger_modele

app = Flask(__name__)

@app.route('/api/model', methods=['POST'])
def get_model_info():
    data = request.json
    if not data or 'path' not in data or 'json_path' not in data:
        return jsonify({"error": "Paths to the Python file and JSON file are required."}), 400

    path = data['path']
    json_path = data['json_path']
    
    # Lire le contenu des fichiers
    contenu_fichier = lire_fichier(path)
    models_data = lire_json_models(json_path)
    
    if not contenu_fichier or not models_data:
        return jsonify({"error": "Impossible de lire le contenu du fichier ou du fichier JSON."}), 400
    
    model_names = models_data.get("model_names", [])
    model_types = models_data.get("model_types", [])
    
    # Convertir les listes en chaînes de caractères
    model_names_str = ', '.join(model_names)
    model_types_str = ', '.join(model_types)

    # Interroger le modèle
    resultat = interroger_modele(contenu_fichier, model_names_str, model_types_str)
    
    if "error" in resultat:
        return jsonify(resultat), 400
    else:
        return jsonify(resultat)

if __name__ == '__main__':
    app.run(debug=True)
