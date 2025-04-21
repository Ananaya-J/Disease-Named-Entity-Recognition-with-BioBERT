from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from evaluate import load
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
import torch
import gc
import os
import logging
from collections import defaultdict, Counter
import sys
from datetime import datetime
from torch.nn import CrossEntropyLoss

# Configuration
torch.set_num_threads(12)
device = torch.device("cpu")

# Setup directories
output_dir = "/scratch/ananaya.jain/new_model"
os.makedirs(output_dir, exist_ok=True)

# Subdirectories
log_dir = os.path.join(output_dir, "logs")
results_dir = os.path.join(output_dir, "results")
model_dir = os.path.join(output_dir, "final_model")
tokenized_dir = os.path.join(output_dir, "tokenized_data")

for directory in [log_dir, results_dir, model_dir, tokenized_dir]:
    os.makedirs(directory, exist_ok=True)

# Logging setup
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedNERProcessor:
    def __init__(self):
        logger.info("Initializing EnhancedNERProcessor")
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        
        # Complete entity label set
        self.unified_label_list = [
            "O",
            "B-DIS", "I-DIS",       # Diseases
            "B-CHEM", "I-CHEM",     # Chemicals
            "B-GENE", "I-GENE",     # Genes
            "B-PROT", "I-PROT",     # Proteins
            "B-CELL", "I-CELL",     # Cell lines
            "B-DNA", "I-DNA"        # DNA sequences
        ]
        
        self.label_to_id = {l: i for i, l in enumerate(self.unified_label_list)}
        self.id_to_label = {i: l for l, i in self.label_to_id.items()}

        # Enhanced label mappings with fallbacks
        self.dataset_label_mappings = {
            "ncbi_disease": {
                "O": "O",
                "B-DISO": "B-DIS",
                "I-DISO": "I-DIS",
                "B-Disease": "B-DIS", 
                "I-Disease": "I-DIS"
            },
            "bigbio/bc5cdr": {
                "O": "O",
                "B-Chemical": "B-CHEM",
                "I-Chemical": "I-CHEM",
                "B-Disease": "B-DIS",
                "I-Disease": "I-DIS",
                "B-Gene": "B-GENE",
                "I-Gene": "I-GENE",
                "B-Protein": "B-PROT",
                "I-Protein": "I-PROT"
            },
            "bigbio/bionlp_st_2013_cg": {
                "O": "O",
                "B-Gene_or_gene_product": "B-GENE",
                "I-Gene_or_gene_product": "I-GENE",
                "B-Simple_chemical": "B-CHEM",
                "I-Simple_chemical": "I-CHEM",
                "B-Cancer": "B-DIS",
                "I-Cancer": "I-DIS",
                "B-Amino_acid": "B-CHEM",
                "I-Amino_acid": "I-CHEM",
                "B-Protein": "B-PROT",
                "I-Protein": "I-PROT"
            }
        }

    def strict_tokenize_and_align(self, tokens, label_ids):
        """Robust tokenization with entity validation"""
        try:
            if not tokens or not label_ids:
                return self._empty_example()
            
            tokenized = self.tokenizer(
                tokens,
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True,
                return_tensors="pt"  # Return PyTorch tensors
            )
            
            # Align labels
            word_ids = tokenized.word_ids()
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    aligned_labels.append(
                        label_ids[word_idx] 
                        if word_idx < len(label_ids) 
                        else self.label_to_id["O"]
                    )
                else:
                    aligned_labels.append(-100)
                previous_word_idx = word_idx
            
            # Validate entities
            token_texts = self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
            validated_labels = self._validate_entities(token_texts, aligned_labels)
            
            return {
                'input_ids': tokenized['input_ids'][0].tolist(),
                'attention_mask': tokenized['attention_mask'][0].tolist(),
                'labels': validated_labels
            }
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            return self._empty_example()

    def _validate_entities(self, token_texts, labels):
        """Validate entity continuity and patterns"""
        new_labels = []
        
        for i, (token, label_id) in enumerate(zip(token_texts, labels)):
            if label_id == -100:
                new_labels.append(-100)
                continue
            
            # Check for valid label ID range
            if label_id >= len(self.id_to_label):
                logger.warning(f"Invalid label ID {label_id}, setting to 'O'")
                new_labels.append(self.label_to_id["O"])
                continue
            
            label = self.id_to_label[label_id]
            
            if label.startswith("B-"):
                entity_type = label[2:]
                if not self._is_valid_entity(token, entity_type, token_texts, i):
                    new_labels.append(self.label_to_id["O"])
                    continue
                    
            new_labels.append(label_id)
        
        return new_labels

    def _is_valid_entity(self, token, entity_type, token_texts, start_idx):
        """Entity-specific validation rules"""
        # Chemical validation (numbers, symbols)
        if entity_type == "CHEM":
            return any(c.isdigit() or c in "-/+" for c in token)
        
        # Gene validation (uppercase, prefixes)
        elif entity_type == "GENE":
            return (token.isupper() or 
                   token.startswith(("p", "c", "m")) or
                   "-" in token)
        
        # Protein validation (keywords, prefixes)
        elif entity_type == "PROT":
            return (token.isupper() or 
                   token.lower().startswith(("p", "h")) or
                   "protein" in token.lower())
        
        # DNA validation (sequence patterns)
        elif entity_type == "DNA":
            return any(seq in token for seq in ["DNA", "RNA", "seq"])
        
        return True

    def process_ncbi_disease(self, examples):
        """Process NCBI disease dataset with validation"""
        processed = {'input_ids': [], 'attention_mask': [], 'labels': []}
        
        for i in range(len(examples["tokens"])):
            tokens = examples["tokens"][i]
            tags = examples["ner_tags"][i]
            
            label_ids = [
                self.label_to_id["B-DIS"] if tag == 1 else
                self.label_to_id["I-DIS"] if tag == 2 else
                self.label_to_id["O"]
                for tag in tags
            ]
            
            tokenized = self.strict_tokenize_and_align(tokens, label_ids)
            for key in processed:
                processed[key].append(tokenized[key])
        
        return processed

    def process_bc5cdr(self, examples):
        """Process BC5CDR with enhanced entity recognition"""
        processed = {'input_ids': [], 'attention_mask': [], 'labels': []}
        
        for idx in range(len(examples['id'])):
            # Text processing
            full_text = ' '.join([p['text'][0] for p in examples['passages'][idx]])
            tokens = full_text.split()
            labels = ["O"] * len(tokens)
            
            # Create token offsets
            token_starts = [0]
            for token in tokens[:-1]:
                token_starts.append(token_starts[-1] + len(token) + 1)
            
            # Process entities
            for entity in examples['entities'][idx]:
                entity_type = self._resolve_entity_type(entity)
                
                for offset in entity['offsets']:
                    start, end = offset[0], offset[1]
                    start_idx, end_idx = self._find_token_indices(token_starts, start, end)
                    
                    if start_idx is not None:
                        bio_tag = "B-" + entity_type
                        mapped_label = self.dataset_label_mappings["bigbio/bc5cdr"].get(bio_tag, "O")
                        if mapped_label != "O":
                            labels[start_idx] = mapped_label
                            
                            for i in range(start_idx + 1, min(end_idx, len(tokens))):
                                bio_tag = "I-" + entity_type
                                mapped_label = self.dataset_label_mappings["bigbio/bc5cdr"].get(bio_tag, "O")
                                if mapped_label != "O":
                                    labels[i] = mapped_label
            
            # Convert and tokenize
            label_ids = [self.label_to_id[label] for label in labels]
            tokenized = self.strict_tokenize_and_align(tokens, label_ids)
            
            for key in processed:
                processed[key].append(tokenized[key])
        
        return processed

    def _resolve_entity_type(self, entity):
        """Refine entity type based on text"""
        original_type = entity['type']
        text = entity['text'][0].lower()
        
        if original_type == "Chemical":
            if "protein" in text:
                return "Protein"
            elif any(kw in text for kw in ["gene", "mutant", "variant"]):
                return "Gene"
            elif any(c.isdigit() for c in text):
                return "Chemical"
        return original_type

    def _find_token_indices(self, token_starts, start, end):
        """Find token indices for character offsets"""
        start_idx = None
        end_idx = None
        
        for i, pos in enumerate(token_starts):
            if start_idx is None and pos >= start:
                start_idx = max(0, i - 1)
            if end_idx is None and pos > end:
                end_idx = i
                break
        
        if end_idx is None:
            end_idx = len(token_starts)
            
        return start_idx, end_idx

    def _empty_example(self):
        """Return empty example placeholder"""
        return {
            'input_ids': [self.tokenizer.pad_token_id] * 128,
            'attention_mask': [0] * 128,
            'labels': [-100] * 128
        }

def compute_metrics(p):
    """Enhanced metrics with entity tracking and proper prefix, with error handling"""
    try:
        metric = load("seqeval")
        processor = compute_metrics.processor

        preds, labels = p
        preds = np.argmax(preds, axis=2)

        # Remove ignored index
        true_preds = []
        true_labels = []
        
        for pred, label in zip(preds, labels):
            pred_seq = []
            label_seq = []
            for p, l in zip(pred, label):
                if l != -100:
                    # Ensure valid label IDs
                    if p < len(processor.id_to_label) and l < len(processor.id_to_label):
                        pred_seq.append(processor.id_to_label[p])
                        label_seq.append(processor.id_to_label[l])
                    else:
                        logger.warning(f"Invalid label ID in predictions: pred={p}, label={l}")
                        pred_seq.append("O")
                        label_seq.append("O")
            
            if pred_seq and label_seq:  # Only add non-empty sequences
                true_preds.append(pred_seq)
                true_labels.append(label_seq)

        if not true_preds or not true_labels:
            logger.warning("No valid predictions found for metrics calculation")
            return {f"eval_{metric}_f1": 0.0 for metric in ["overall", "DIS", "CHEM", "GENE", "PROT", "CELL", "DNA"]}

        # Overall metrics
        try:
            overall_results = metric.compute(predictions=true_preds, references=true_labels)
        except Exception as e:
            logger.error(f"Error computing overall metrics: {str(e)}")
            overall_results = {
                "overall_precision": 0.0,
                "overall_recall": 0.0,
                "overall_f1": 0.0,
                "overall_accuracy": 0.0
            }
        
        # Create results dictionary with proper eval_ prefix
        results = {
            "eval_loss": 0.0,  # Placeholder - actual loss is computed elsewhere
            "eval_overall_precision": overall_results.get("overall_precision", 0.0),
            "eval_overall_recall": overall_results.get("overall_recall", 0.0),
            "eval_overall_f1": overall_results.get("overall_f1", 0.0),
            "eval_overall_accuracy": overall_results.get("overall_accuracy", 0.0)
        }
        
        # Per-entity metrics with eval_ prefix
        entities = ["DIS", "CHEM", "GENE", "PROT", "CELL", "DNA"]
        for ent in entities:
            ent_preds = []
            ent_labels = []
            
            for pred_seq, label_seq in zip(true_preds, true_labels):
                pred_ents = [p for p in pred_seq if ent in p]
                label_ents = [l for l in label_seq if ent in l]
                
                if pred_ents or label_ents:
                    ent_preds.append(pred_seq)
                    ent_labels.append(label_seq)
            
            if ent_labels:
                try:
                    ent_results = metric.compute(predictions=ent_preds, references=ent_labels)
                    results.update({
                        f"eval_{ent}_precision": ent_results["overall_precision"],
                        f"eval_{ent}_recall": ent_results["overall_recall"],
                        f"eval_{ent}_f1": ent_results["overall_f1"]
                    })
                except Exception as e:
                    logger.error(f"Error computing metrics for entity {ent}: {str(e)}")
                    results.update({
                        f"eval_{ent}_precision": 0.0, 
                        f"eval_{ent}_recall": 0.0, 
                        f"eval_{ent}_f1": 0.0
                    })
            else:
                # Add metrics for entities not present in evaluation data
                results.update({
                    f"eval_{ent}_precision": 0.0,
                    f"eval_{ent}_recall": 0.0,
                    f"eval_{ent}_f1": 0.0
                })
        
        # Also add non-prefixed versions for backward compatibility
        results.update({
            "overall_precision": overall_results.get("overall_precision", 0.0),
            "overall_recall": overall_results.get("overall_recall", 0.0),
            "overall_f1": overall_results.get("overall_f1", 0.0),
            "overall_accuracy": overall_results.get("overall_accuracy", 0.0)
        })
        
        # Log entity presence
        entity_presence = {ent: results[f"eval_{ent}_f1"] > 0 for ent in entities}
        logger.info(f"Entity presence in evaluation: {entity_presence}")
        
        return results
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}", exc_info=True)
        # Return default metrics with proper eval_ prefix
        default_metrics = {
            "eval_loss": 0.0,
            "eval_overall_precision": 0.0,
            "eval_overall_recall": 0.0,
            "eval_overall_f1": 0.0,
            "eval_overall_accuracy": 0.0
        }
        for ent in ["DIS", "CHEM", "GENE", "PROT", "CELL", "DNA"]:
            default_metrics.update({
                f"eval_{ent}_precision": 0,
                f"eval_{ent}_recall": 0,
                f"eval_{ent}_f1": 0
            })
        return default_metrics

class WeightedTrainer(Trainer):
    """Trainer with class weights and focal loss"""
    def __init__(self, class_weights=None, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.gamma = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Modified to handle newer Transformers versions"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Make sure class weights match number of classes in model
        if self.class_weights is not None:
            num_labels = model.config.num_labels
            
            # Ensure we have weights for all classes
            if len(self.class_weights) != num_labels:
                logger.warning(f"Class weights length ({len(self.class_weights)}) doesn't match num_labels ({num_labels})")
                # Create proper length weight tensor
                weight_tensor = torch.ones(num_labels, device=logits.device)
                for i, w in enumerate(self.class_weights):
                    if i < num_labels:
                        weight_tensor[i] = w
                loss_fct = CrossEntropyLoss(weight=weight_tensor)
            else:
                loss_fct = CrossEntropyLoss(
                    weight=torch.tensor(self.class_weights, device=logits.device)
                )
        else:
            loss_fct = CrossEntropyLoss()
        
        # Focal loss
        ce_loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return (focal_loss, outputs) if return_outputs else focal_loss

def calculate_class_weights(datasets, processor):
    """Calculate inverse frequency class weights for all label IDs"""
    label_counts = Counter()
    
    # Initialize counts for all possible label IDs
    for label_id in range(len(processor.unified_label_list)):
        label_counts[label_id] = 0
    
    # Add minimum count to prevent division by zero
    for label_id in range(len(processor.unified_label_list)):
        label_counts[label_id] += 1  # Add pseudocount
    
    for dataset in datasets:
        for split in ["train", "validation"]:
            if split in dataset:
                for example in dataset[split]:
                    labels = example["labels"]
                    if isinstance(labels, torch.Tensor):
                        labels = labels.tolist()
                    for lid in labels:
                        if lid != -100 and lid < len(processor.unified_label_list):  # Ignore padding token
                            label_counts[lid] += 1
    
    # Calculate weights for all classes
    total = sum(label_counts.values())
    weights = []
    
    for label_id in range(len(processor.unified_label_list)):
        count = label_counts[label_id]
        weight = total / (len(processor.unified_label_list) * count)
        weights.append(weight)
    
    # Print label distribution for debugging
    logger.info("Label distribution:")
    for label_id, label in processor.id_to_label.items():
        logger.info(f"{label}: {label_counts[label_id]} (weight: {weights[label_id]:.4f})")
    
    return weights

def save_tokenized_data(data, path):
    """Save tokenized dataset"""
    try:
        data.save_to_disk(path)
        logger.info(f"Saved tokenized data to {path}")
    except Exception as e:
        logger.error(f"Error saving tokenized data: {str(e)}")

def load_tokenized_data(path):
    """Load saved tokenized data"""
    try:
        data = load_from_disk(path)
        logger.info(f"Loaded tokenized data from {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading tokenized data: {str(e)}")
        return None

def main():
    logger.info("Starting enhanced NER training")
    
    # Initialize processor
    processor = EnhancedNERProcessor()
    compute_metrics.processor = processor
    
    # Try loading pre-tokenized data
    tokenized_paths = {
        "ncbi": os.path.join(tokenized_dir, "ncbi"),
        "bc5cdr": os.path.join(tokenized_dir, "bc5cdr"), 
        "bionlp": os.path.join(tokenized_dir, "bionlp")
    }
    
    if all(os.path.exists(p) for p in tokenized_paths.values()):
        logger.info("Loading pre-tokenized datasets")
        tokenized_ncbi = load_tokenized_data(tokenized_paths["ncbi"])
        tokenized_bc5cdr = load_tokenized_data(tokenized_paths["bc5cdr"])
        tokenized_bionlp = load_tokenized_data(tokenized_paths["bionlp"])
    else:
        # Load and process raw datasets
        logger.info("Processing raw datasets")
        try:
            ncbi = load_dataset("ncbi_disease")
            bc5cdr = load_dataset("bigbio/bc5cdr", "bc5cdr_bigbio_kb")
            bionlp = load_dataset("bigbio/bionlp_st_2013_cg", "bionlp_st_2013_cg_bigbio_kb")
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            sys.exit(1)
        
        # Process datasets
        tokenized_ncbi = ncbi.map(
            processor.process_ncbi_disease,
            batched=True,
            batch_size=32,
            remove_columns=ncbi["train"].column_names
        ).with_format("torch")
        
        tokenized_bc5cdr = bc5cdr.map(
            processor.process_bc5cdr,
            batched=True,
            batch_size=32,
            remove_columns=bc5cdr["train"].column_names
        ).with_format("torch")
        
        tokenized_bionlp = bionlp.map(
            processor.process_bc5cdr,  # Reuse processor
            batched=True, 
            batch_size=32,
            remove_columns=bionlp["train"].column_names
        ).with_format("torch")
        
        # Save tokenized data
        save_tokenized_data(tokenized_ncbi, tokenized_paths["ncbi"])
        save_tokenized_data(tokenized_bc5cdr, tokenized_paths["bc5cdr"])
        save_tokenized_data(tokenized_bionlp, tokenized_paths["bionlp"])
    
    # Combine datasets
    train_sets = [d["train"] for d in [tokenized_ncbi, tokenized_bc5cdr, tokenized_bionlp]]
    val_sets = [
        d["validation"] if "validation" in d else d["test"]
        for d in [tokenized_ncbi, tokenized_bc5cdr, tokenized_bionlp]
    ]
    
    combined_train = concatenate_datasets(train_sets)
    combined_val = concatenate_datasets(val_sets)
    
    logger.info(f"Training samples: {len(combined_train)}")
    logger.info(f"Validation samples: {len(combined_val)}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(
        [tokenized_ncbi, tokenized_bc5cdr, tokenized_bionlp],
        processor
    )
    logger.info(f"Class weights: {class_weights}")
    logger.info(f"Number of classes: {len(processor.unified_label_list)}")
    logger.info(f"Number of weights: {len(class_weights)}")
    
    # Initialize model
    model = AutoModelForTokenClassification.from_pretrained(
        "dmis-lab/biobert-v1.1",
        num_labels=len(processor.unified_label_list),
        id2label=processor.id_to_label,
        label2id=processor.label_to_id,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1
    )
    model.to(device)
    
    # Training configuration with updated metric
    # Choose a metric that's more likely to exist: overall F1 instead of GENE F1
    training_args = TrainingArguments(
        output_dir=results_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir=log_dir,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_overall_f1",  # Changed from eval_GENE_f1
        greater_is_better=True,
        save_total_limit=2,
        use_cpu=True,
        warmup_steps=500,
        report_to="none"
    )
    
    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_train,
        eval_dataset=combined_val,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        gamma=2.0
    )
    
    # Train and evaluate
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Evaluating final model...")
    metrics = trainer.evaluate()
    
    # Save results
    trainer.save_model(model_dir)
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    
    logger.info(f"Training complete. Model saved to {model_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)