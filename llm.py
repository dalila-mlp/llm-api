import json
import re
from openai import OpenAI
from collections import OrderedDict

# Initialiser le client OpenAI
client = OpenAI(
    api_key="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    base_url="https://api.llama-api.com"
)

# Fonction pour lire le fichier Python
def lire_fichier(path):
    try:
        with open(path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None

# Fonction pour vérifier le format de la réponse
def verifier_format_response(response):
    pattern = r"^model name : [^;]+ ; model_type: [^;]+ ; best 3 metrics : [^,]+, [^,]+, [^,]+$"
    return re.match(pattern, response, re.IGNORECASE) is not None

# Fonction pour lire le fichier JSON des modèles
def lire_json_models(json_path):
    try:
        with open(json_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier JSON : {e}")
        return None

# Fonction pour interroger le modèle et traiter la réponse
def interroger_modele(contenu_fichier, model_names_str, model_types_str):
    try:
        # Poser la question au modèle
        response = client.chat.completions.create(
            model="llama-13b-chat",
            messages=[
                {"role": "system", "content": "Tu es un expert en machine learning."},
                {"role": "user", "content": f"""
                Voici le contenu d'un fichier Python: {contenu_fichier}
                Voici une liste de nom de modèle parmi lesquels choisir: {model_names_str}
                Voici une liste de types de modèles parmi lesquels choisir: {model_types_str}
                Quel modèle est utilisé dans ce fichier et quels sont les metrics les plus adaptés pour ce code. 
                Je veux une sortie uniquement dans ce format exact rien d'autre:
                'model name : Logistic Regression ; model_type: Classification ; best 3 metrics : Precision, Recall, F1-score'
                """}
            ]
        )
        
        # Extraire la réponse du modèle
        reponse = response.choices[0].message.content.strip()
        print("Réponse du modèle :", reponse)  # Ajout de l'impression de débogage
        
        if verifier_format_response(reponse):
            # Extraire les informations du modèle et des metrics
            match = re.match(r"model name : ([^;]+) ; model_type: ([^;]+) ; best 3 metrics : ([^,]+), ([^,]+), ([^,]+)", reponse, re.IGNORECASE)
            if match:
                model_name = match.group(1).strip()
                model_type = match.group(2).strip()
                metrics = [match.group(3).strip(), match.group(4).strip(), match.group(5).strip()]
                
                # Créer la réponse avec un ordre spécifique des clés
                resultat = OrderedDict([
                    ("model_name", model_name),
                    ("model_type", model_type),
                    ("best 3 metrics", metrics)
                ])
                
                return resultat
            else:
                return {"error": "Le format de la réponse ne correspond pas aux attentes."}
        else:
            return {"error": "Le format de la réponse est incorrect."}
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API ou du traitement de la réponse : {e}")
        return {"error": "Erreur lors de l'appel à l'API ou du traitement de la réponse."}
