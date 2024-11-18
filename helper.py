import os
import re
import warnings
warnings.filterwarnings("ignore")
from dateutil import parser

import torch, gc
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.preprocessing import normalize


date_pattern = r"""
\b(
    # Month Name (Full or Abbreviated) followed by Day, optional comma, and Year
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|
    Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
    \s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4}|\s+\d{4})? |

    # MM/DD/YYYY or MM-DD-YYYY
    \d{1,2}[/-]\d{1,2}[/-]\d{4} |

    # MM/DD or MM-DD
    \d{1,2}[/-]\d{1,2} |

    # ISO Date Format YYYY-MM-DD or YYYY/MM/DD
    \d{4}[/-]\d{1,2}[/-]\d{1,2} |

    # Day-Month-Year Format DD/MM/YYYY or DD-MM-YYYY
    \d{1,2}[/-]\d{1,2}[/-]\d{4}
)\b
"""

class NER:
    def __init__(self, device_map="cpu"):
        self.device_map = device_map
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", trust_remote_code=True, device_map=device_map)
        self.ner = pipeline("ner", model=model, tokenizer=tokenizer)
        self.date_regex = re.compile(date_pattern, re.VERBOSE)

    def run(self, query):
        entities = self.merge_word(self.ner(query))
        query_dates = list(set([self.standardize_date(date) for date in self.date_regex.findall(query)]))
        return entities, query_dates
    
    def merge_word(self, ner_results):
        entities = []
        current_entity = {"entity": None, "text": "", "start": None, "end": None}

        for token in ner_results:
            entity_tag = token["entity"]
            token_text = token["word"]
            token_start = token["start"]
            token_end = token["end"]

            if "PER" in entity_tag: continue # Skip person names because they are preprocessed

            # If it's a beginning of a new entity
            if entity_tag.startswith("B-"):
                # Save the current entity if it exists
                if current_entity["text"]:
                    entities.append(current_entity)

                # Start a new entity
                current_entity = {
                    "entity": entity_tag[2:],  # Remove 'B-' prefix
                    "text": token_text,
                    "start": token_start,
                    "end": token_end,
                }
            elif entity_tag.startswith("I-") and current_entity["entity"] == entity_tag[2:]:
                # Continue the current entity
                current_entity["text"] = current_entity["text"] + f" {token_text}" if not token_text.startswith("##") else current_entity["text"] + token_text[2:]
                current_entity["end"] = token_end
            else:
                # Save the current entity and reset (handles unexpected cases)
                if current_entity["text"]:
                    entities.append(current_entity)
                current_entity = {"entity": None, "text": "", "start": None, "end": None}

        # Add the last entity if any
        if current_entity["text"]:
            entities.append(current_entity)

        return entities
    
    def standardize_date(self, date_str):
        try:
            # Parse the date string
            parsed_date = parser.parse(date_str)
            # Check if the parsed date has a year
            if parsed_date.year:
                return parsed_date.strftime("%m/%d/%Y")
            else:
                return parsed_date.strftime("%m/%d")
        except Exception as e:
            return date_str  # Return the original if parsing fails


class Embedder:
    def __init__(self, model_path, device_map="cpu"):
        self.device_map = device_map
        stella = model_path
        vector_dim = 8192
        vector_linear_directory = f"2_Dense_{vector_dim}"
        self.stella_model = AutoModel.from_pretrained(stella, trust_remote_code=True, device_map=self.device_map, 
                                                      use_memory_efficient_attention=False, unpad_inputs=False).eval() # only for non-cuda devices
        self.stella_tokenizer = AutoTokenizer.from_pretrained(stella, trust_remote_code=True)
        self.stella_linear = torch.nn.Linear(in_features=self.stella_model.config.hidden_size, out_features=vector_dim)
        stella_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(stella, f"{vector_linear_directory}/pytorch_model.bin"), map_location=torch.device(self.device_map)).items()
        }
        self.stella_linear.load_state_dict(stella_linear_dict)
        self.stella_linear.to(self.device_map)

    def embed(self, queries):
        queries_vectors = []
        if type(queries) == str: queries = [queries]
        with torch.no_grad():
            for query in queries:
                input_data = self.stella_tokenizer(query, padding="longest", return_tensors="pt")
                input_data = {k: v.to(self.device_map) for k, v in input_data.items()}
                attention_mask = input_data["attention_mask"]
                last_hidden_state = self.stella_model(**input_data)[0]
                last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                linears = self.stella_linear(vectors)
                queries_vectors.append(normalize(linears.cpu().numpy(force = True)))
                del input_data, attention_mask, last_hidden_state, last_hidden, vectors, linears
                gc.collect(); torch.mps.empty_cache(); torch.cuda.empty_cache()
        queries_vectors = np.concatenate(queries_vectors, axis=0)
        queries_vectors = torch.tensor(queries_vectors, device=self.device_map)
        return queries_vectors

    def similarity(self, queries_vectors, reference_vectors):
        similarities = torch.matmul(queries_vectors, reference_vectors.T).cpu().numpy(force = True)
        return similarities