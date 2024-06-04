from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from service_models import DocumentClassifier, DocumentClassifierConfig, DocumentClassifierConfigLoader, DocumentRequest, PredictedDocumentClass

app = FastAPI()

# Initialize the ConfigLoader with the path to the serialized configuration
document_classification_config = DocumentClassifierConfigLoader(filepath='trellis_document_classification/service_model_bundle.joblib', config_type=DocumentClassifierConfig)

# Initialize the DocumentClassifier with the ConfigLoader
classifier = DocumentClassifier(config_loader=document_classification_config)

classifier.load()

@app.post("/predict/", response_model=PredictedDocumentClass)
async def predict(request: DocumentRequest):
    try:
        # Use the classifier to predict the class of the provided text
        predicted_document_class = await classifier.predict(request.text)
        return PredictedDocumentClass(class_label=predicted_document_class)
    except ValidationError as ve:
        # Handle validation errors specifically
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Validation error: " + str(ve)
        )
    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)