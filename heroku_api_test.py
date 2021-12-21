import requests
import json 


params = {
                "age": 34,
                "workclass": "Private",
                "fnlgt": 245487,
                "education": "7th-8th",
                "education-num": 4,
                "marital-status": "Married-civ-spouse",
                "occupation": "Transport-moving",
                "relationship": "Husband",
                "race": "Amer-Indian-Eskimo",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "Mexico"
            }


response = requests.post('https://fast-api-ed.herokuapp.com/predict/', data=json.dumps(params))
print(f"response:  {response.json()}")
print(f"status code : {response.status_code}")