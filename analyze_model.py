import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiseaseNERPredictor:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Extended label list for both disease and chemical entities
        self.label_list = ['O', 'B-DIS', 'I-DIS', 'B-CHEM', 'I-CHEM']
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        
        logger.info(f"Loaded model from {model_path} on {device}")

    def predict(self, text, return_tokens=False):
        """Predict disease and chemical entities in text"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=False
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)[0]
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            labels = [self.id2label[pred.item()] for pred in predictions]

            diseases = self._extract_entities(tokens, labels, "DIS")
            chemicals = self._extract_entities(tokens, labels, "CHEM")

            if return_tokens:
                return (diseases, chemicals), list(zip(tokens, labels))
            return diseases, chemicals
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return ([], []) if not return_tokens else (([], []), [])

    def _extract_entities(self, tokens, labels, entity_prefix):
        """Extract entities from predictions (supports DIS and CHEM)"""
        entities = []
        current_entity = []

        for token, label in zip(tokens, labels):
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue
            
            if label == f"B-{entity_prefix}":
                if current_entity:
                    entity_text = self._combine_tokens(current_entity)
                    if self._is_valid_entity(entity_text, entity_prefix):
                        entities.append(entity_text)
                    current_entity = []
                current_entity.append(token)
            elif label == f"I-{entity_prefix}" and current_entity:
                current_entity.append(token)
            else:
                if current_entity:
                    entity_text = self._combine_tokens(current_entity)
                    if self._is_valid_entity(entity_text, entity_prefix):
                        entities.append(entity_text)
                    current_entity = []

        if current_entity:
            entity_text = self._combine_tokens(current_entity)
            if self._is_valid_entity(entity_text, entity_prefix):
                entities.append(entity_text)

        return entities

    def _combine_tokens(self, tokens):
        text = " ".join(tokens)
        return text.replace(" ##", "").replace("##", "")

    def _is_valid_entity(self, text, entity_prefix):
        """Check if extracted entity is valid (skip false positives for diseases and chemicals)"""
        if len(text) <= 1:
            return False
        
        false_positives = {
            "DIS": ["patient", "treatment", "therapy", "diagnosis", "symptoms", "study"],
            "CHEM": ["treatment", "therapy", "drug", "medication", "injection"]  # List of words to ignore for chemicals
        }

        # Adjust false positive list based on the entity type (disease or chemical)
        if entity_prefix == "DIS" and text.lower() in false_positives["DIS"]:
            return False
        if entity_prefix == "CHEM" and text.lower() in false_positives["CHEM"]:
            return False
        
        # Skip numeric values (e.g., concentrations or dosage amounts) in chemicals
        if text.isnumeric():
            return False
        
        return True

    def predict_batch(self, texts, batch_size=8):
        """Batch prediction for multiple texts"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_offsets_mapping=False
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=2)
                
                for j in range(len(batch)):
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][j])
                    labels = [self.id2label[pred.item()] for pred in predictions[j]]
                    seq_len = (inputs["attention_mask"][j] == 1).sum()
                    tokens = tokens[:seq_len]
                    labels = labels[:seq_len]

                    diseases = self._extract_entities(tokens, labels, "DIS")
                    chemicals = self._extract_entities(tokens, labels, "CHEM")
                    results.append((diseases, chemicals))
                    
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                results.extend([([], []) for _ in batch])
                
        return results

def test_examples(predictor):
    examples = [
        "The patient was diagnosed with breast cancer and diabetes mellitus.",
        "EGFR mutations are associated with lung adenocarcinoma.",
        "Treatment with paclitaxel improved the Hodgkin lymphoma symptoms.",
        "Alzheimer's disease and Parkinson's disease are neurodegenerative disorders.",
        "The study focused on COVID-19 patients with hypertension.",
        "Aspirin is commonly used for cardiovascular disease prevention.",
        "Metformin is the first-line treatment for type 2 diabetes mellitus.",
    ]
    
    print("\nTesting Disease and Chemical NER Predictor:")
    print("=" * 50)
    for text in examples:
        diseases, chemicals = predictor.predict(text)
        print(f"\nText: {text}")
        if diseases:
            print("Detected Diseases:")
            for d in diseases:
                print(f"- {d}")
        if chemicals:
            print("Detected Chemicals:")
            for c in chemicals:
                print(f"- {c}")
        if not diseases and not chemicals:
            print("- No entities detected")

if __name__ == "__main__":
    predictor = DiseaseNERPredictor(
        model_path="/home/ananaya/Desktop/Biobert/LLM_project/new_model/final_model",
        device="cpu"
    )
    
    test_examples(predictor)
    
    print("\nInteractive Mode (type 'quit' to exit)")
    print("=" * 50)
    while True:
        text = input("\nEnter text to analyze: ")
        if text.lower() in ['quit', 'exit']:
            break
        
        diseases, chemicals = predictor.predict(text)
        if diseases:
            print("\nDetected Diseases:")
            for d in diseases:
                print(f"- {d}")
        if chemicals:
            print("\nDetected Chemicals:")
            for c in chemicals:
                print(f"- {c}")
        if not diseases and not chemicals:
            print("- No entities detected")

