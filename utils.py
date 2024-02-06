import requests  
import json  
import numpy as np
import pandas as pd
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
from io import BytesIO

def generacion_imagenes(input_text):
    
    # Si se hace clic en el botón, mostrar el texto
    url =  "https://stablediffusionapi.com/api/v4/dreambooth"  
    # key cem: KiT6NVdSSo3o3thacEep7QAapYptwy9yZVcbWTQEyTviTQTcNIbX3oSXpTtw
    # key cem2 :inXf3cMyZXC5mzuPBAZtVaNYGFBwF8jA7EjSdb5PqRxtpszHw9tf6HtCIjGe
    payload = json.dumps({  
        "key":  "KiT6NVdSSo3o3thacEep7QAapYptwy9yZVcbWTQEyTviTQTcNIbX3oSXpTtw",  
        "model_id":  "juggernaut-xl-v5",  
        "prompt": input_text,  
        "width":  "512",  
        "height":  "512",  
        "samples":  "1",  
        "num_inference_steps":  "30",  
        "safety_checker":  "no",  
        "enhance_prompt":  "yes",  
        "seed":  None,  
        "guidance_scale":  7.5,  
        "multi_lingual":  "no",  
        "panorama":  "no",  
        "self_attention":  "no",  
        "upscale":  "no",  
        "embeddings":  "embeddings_model_id",  
        "lora":  "lora_model_id",  
        "webhook":  None,  
        "track_id":  None  
        })  
        
    headers =  {  
    'Content-Type':  'application/json'  
    }  
    
    response = requests.request("POST", url, headers=headers, data=payload)
    print("response claves:", response.json().keys())
    resp_json = response.json()['future_links'][0]
    return resp_json
    

def clasificacion_imagenes(uploaded_file):
    if isinstance(uploaded_file, str):
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        # Preprocess the image and prepare it for the model
        print("antes de input")
        response_url = requests.get(uploaded_file)
        img = Image.open(BytesIO(response_url.content))
        print("img: ",img)
        #image = Image.open(img)
        inputs = feature_extractor(images=img, return_tensors="pt")
        # Perform the prediction
        outputs = model(**inputs)
        logits = outputs.logits

        # Retrieve the highest probability class
        predicted_class_idx = logits.argmax(-1).item()
        predicho = model.config.id2label[predicted_class_idx] 
        return predicho
    else:
        # Load the feature extractor and model from Hugging Face
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        image = Image.open(uploaded_file)

        # Preprocess the image and prepare it for the model
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Perform the prediction
        outputs = model(**inputs)
        logits = outputs.logits

        # Retrieve the highest probability class
        predicted_class_idx = logits.argmax(-1).item()
        predicho = model.config.id2label[predicted_class_idx]
        #print("Predicted class:", predicho)
        return predicho
