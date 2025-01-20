import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
def top_n_max_indices(lst, n):
    # Get the indices of the top n elements
    return sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:n]
def preprocess(data_path):
    for prefix in ['train', 'dev', 'test']:
        with open(f'{data_path}/{prefix}.jsonl', 'r') as f_r, open(f'{data_path}/{prefix}.in', 'w') as f_w:
            for line in f_r:
                sample = json.loads(line.strip())
                f_w.write(' '.join(sample['input_tokens'])+'\n')


def postprocess(data_path, example_path):
    for prefix in ['train', 'dev', 'test']:
        print(f"processing {prefix}")
        retrieval_corpus = []
        with open(f'{data_path}/{prefix}.jsonl', 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                retrieval_corpus.append(sample)

        data = []
        with open(f'{data_path}/{prefix}.jsonl', 'r') as raw_data:

            raw_data = raw_data.readlines()

            for line in raw_data:
                sample = json.loads(line.strip())
                data.append(sample)

        # model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load the CodeT5 model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        model = AutoModel.from_pretrained("Salesforce/codet5-base")

        def encode_text(text):
            """Encode text using CodeT5."""

            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

            # print(inputs)
            with torch.no_grad():
                outputs = model.encoder(**inputs)
                # Use the [CLS] token embedding (or equivalent depending on the architecture)
                return outputs.last_hidden_state
        embeddings = []

        for i, other in enumerate(data):
            other_input = other['input']
            # other_input_tokens = ' '.join(other_input_tokens)
            other_embedding = encode_text(other_input)
            embeddings.append(other_embedding)

        with open(f'{data_path}/{prefix}_with_example.jsonl', 'w') as output:
            for i, sample in enumerate(data):
                print(f'process {i}')

                similarity_score = []
                sample_embedding = embeddings[i]
                sample_embedding = sample_embedding[0, 0, :]
                for j, _ in enumerate(data):
                    if i == j:
                        similarity_score.append(-1 * np.inf)
                    else:
                        other_embedding = embeddings[j]
                        other_embedding = other_embedding[0, 0, :]

                        similarity = torch.nn.functional.cosine_similarity(
                            sample_embedding.unsqueeze(0),  # Add batch dimension
                            other_embedding.unsqueeze(0),
                            dim=1)
                        similarity_score.append(similarity)
                top_similarity_idx = top_n_max_indices(similarity_score, len(data))
                similar_samples = []
                target_code = ' '.join(sample['output_tokens'])

                for idx in top_similarity_idx:
                    retrieved_code = data[idx]['output_tokens']
                    retrieved_code = ' '.join(retrieved_code)
                    if retrieved_code == target_code:  # remove target code
                        continue
                    else:
                        similar_samples.append(retrieved_code)
                        if len(similar_samples) == 5:
                            break
                sample['examples'] = similar_samples
                output.write(json.dumps(sample) + '\n')

if __name__ == '__main__':
    postprocess('data/hearthstone', '-')