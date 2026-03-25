import os
import clip
import open_clip
import torch
import json
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import re
import argparse

device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description='Run SimpleNeg for negation understanding in CLIP models')
parser.add_argument('--model_name', type=str, default='ViT-B/32', help='CLIP model name (e.g., ViT-B/32, ViT-L/14)')
parser.add_argument('--openclip_model_name', type=str, default='ViT-B-32', help='OpenCLIP model name (e.g., ViT-B-32). Set to None to skip NegBench/NegCLIP baselines.')
parser.add_argument('--vqa_model_id', type=str, default='Qwen/Qwen3-VL-4B-Instruct', help='VQA model ID for evaluation (e.g., Qwen/Qwen3-VL-4B-Instruct)')
parser.add_argument('--img_database_size', type=int, default=25000, help='Number of images to use from the database')
args = parser.parse_args()

conclip_names_from_model_name = {'ViT-B/32': 'conclip_vit_b32.pt',
                                 'ViT-B/16': 'conclip_vit_b16.pt',
                                 'ViT-L/14': 'conclip_vit_l14.pt'}

def test_example_last_layer(text, coeffecients):
    text = encode_examples_last_layer([text])
    prediction = torch.sigmoid(torch.sum(text * coeffecients)).item()
    if prediction > 0.5:
        print("negative, {:.3f}".format(prediction))
    else:
        print("positive, {:.3f}".format(prediction))
        
@torch.no_grad()
def encode_examples_last_layer(examples, batch_size = 10):
    """ Encodes a list of text examples using the CLIP model and returns the last layer features before normalization layer (that's where we train on) and before the text projector"""
    encoded = []
    for b in range(0, len(examples), batch_size):
        text_input = examples[b : b + batch_size]
        tokenized_text = clip.tokenize(text_input).to(device) 
        x = model.token_embedding(tokenized_text).type(model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + model.positional_embedding.type(model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1)] 
        encoded.append(x)

    encoded = torch.cat(encoded, dim = 0)
    return encoded

def train_binary_classifier(positive_examples, negative_examples):
    pos_np = positive_examples.cpu().numpy()   # shape (N_pos, 512)
    neg_np = negative_examples.cpu().numpy()   # shape (N_neg, 512)

    n = pos_np.shape[0]
    number_of_test_samples = 100
    split_index = n - number_of_test_samples

    pos_train, pos_test = pos_np[:split_index], pos_np[split_index:]
    neg_train, neg_test = neg_np[:split_index], neg_np[split_index:]

    # Stack them to make full train / test sets
    X_train = np.vstack([pos_train, neg_train])   # shape (1800, 512)
    y_train = np.hstack([
        np.zeros(len(pos_train), dtype=int),      # 0 = positive
        np.ones(len(neg_train), dtype=int)        # 1 = negative
    ])

    X_test  = np.vstack([pos_test, neg_test])     # shape ( 200, 512)
    y_test  = np.hstack([
        np.zeros(len(pos_test), dtype=int),       # 0 = positive
        np.ones(len(neg_test), dtype=int)         # 1 = negative
    ])

    # Train a simple logistic‐regression classifier 
    clf = LogisticRegression(
        solver='lbfgs',      # works well for medium‐sized data
        max_iter=1000,
        fit_intercept=False
    )

    clf.fit(X_train, y_train)

    # Evaluate on the test set 
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    directional_vector = torch.from_numpy(clf.coef_).to(device)

    return directional_vector

def train_classifiers_every_layer(positive_stream_activations, negative_stream_activations):
    """
    Train a binary classifier for each layer to distinguish positive vs negative sentences.
    Returns a dictionary of directional vectors (one per layer).
    """
    num_layers = len(positive_stream_activations)
    layer_directional_vectors = {}
    layer_coefficients = {}
    
    for layer_idx in range(num_layers):
        print(f"Training classifier for layer {layer_idx}...")
        
        # Get activations for this layer
        pos_activations = positive_stream_activations[layer_idx]
        neg_activations = negative_stream_activations[layer_idx]
        
        # Train binary classifier for this layer
        directional_vector = train_binary_classifier(pos_activations, neg_activations)
        layer_coefficients[layer_idx] = directional_vector.clone().to(pos_activations.dtype)
        
        # Normalize to unit vector (directional_vector is the unit vector that points in the negative direction)
        directional_vector = directional_vector / (torch.linalg.norm(directional_vector) + 1e-12)
        directional_vector = directional_vector.to(pos_activations.dtype)
        
        layer_directional_vectors[layer_idx] = directional_vector
        
    print("Finished training all layer classifiers!")
    return layer_directional_vectors, layer_coefficients

def get_residual_stream_activations(model, examples):
    """
    Extracts and organizes the residual stream activations of the CLIP Text Encoder
    by layer. Returns a dictionary mapping each layer index to a list of activation
    tensors (one per batch).
    """
    # Number of transformer blocks
    num_layers = len(model.transformer.resblocks)
    # Prepare a dict to store activations per layer
    layer_activations = {layer_idx: [] for layer_idx in range(num_layers)}
    batch_size = 10

    # Hook factory that captures which layer index this hook belongs to
    def make_hook(layer_idx):
        def hook(module, input, output):
            # output shape is (seq_len, batch_size, hidden_size)
            # permute to (batch_size, seq_len, hidden_size)
            x = output.permute(1, 0, 2)
            token_idx = tokenized_text.argmax(dim=-1)  # <eos> token index
            # select for each example in batch the activation at token_idx
            selected = x[torch.arange(x.shape[0]), token_idx]
            # store on CPU (detach) so that GPU tensors don’t accumulate
            layer_activations[layer_idx].append(selected.detach().cpu())
        return hook

    # Register a hook on each transformer block
    hooks = []
    for idx, layer in enumerate(model.transformer.resblocks):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    model.to(device).eval()
    with torch.no_grad():
        for b in range(0, len(examples), batch_size):
            batch_texts = examples[b : b + batch_size]
            tokenized_text = clip.tokenize(batch_texts).to(device)
            _ = model.encode_text(tokenized_text)

    # Remove all hooks once done
    for h in hooks:
        h.remove()

    return layer_activations

def concatenate_layer_activations(layer_activations):
    concatenated = {}
    for layer_idx, batch_list in layer_activations.items():
        concatenated[layer_idx] = torch.cat(batch_list, dim=0)
    return concatenated

# define steering methods here
def steer(text_feats, directional_vector, alpha):
    text_feats_neg = (1 - alpha) * text_feats + alpha * directional_vector * torch.norm(text_feats, dim=-1, keepdim=True)
    return text_feats_neg

def steer_every_layer(model, tokenized_text, layer_directional_vectors, alpha):

    def make_hook_end_token(directional_vector):
        "applies the directional vector only to end tokens"

        def hook(module, inp, out):
            # Original output shape is (seq_len, batch_size, hidden_size) = LND
            output = out.clone()  # Create a copy to modify
            
            # Permute to (batch_size, seq_len, hidden_size) = NLD for easier indexing
            permuted = output.permute(1, 0, 2)

            # Get indices of <eos> tokens
            token_idx = tokenized_text.argmax(dim=-1)  # <eos> token index
            batch_indices = torch.arange(permuted.shape[0])

            # Extract the end tokens
            end_tokens = permuted[batch_indices, token_idx]  # (batch_size, hidden_size)

            # Apply steering to the end tokens
            steered_tokens = steer(end_tokens, directional_vector, alpha=alpha)  # (batch_size, hidden_size)

            # Replace the end tokens in the permuted tensor
            permuted[batch_indices, token_idx] = steered_tokens

            # Permute back to original shape (seq_len, batch_size, hidden_size) = LND
            output = permuted.permute(1, 0, 2)

            return output
        return hook

    hooks = []

    for layer_idx, layer in enumerate(model.transformer.resblocks):
        
        directional_vector = layer_directional_vectors[layer_idx]
        hooks.append(layer.register_forward_hook(make_hook_end_token(directional_vector)))

    with torch.no_grad():
        embedding = model.encode_text(tokenized_text)

    # clean up
    for h in hooks:
        h.remove()

    return embedding


def show_retrieved_images(img_dir, text_feats, img_feats, database, show = True):
    similarity = (100.0 * text_feats @ img_feats.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    retrieved_paths = [database[ind.item()] for ind in indices]
    retrieved_paths = [os.path.join(img_dir, img_name) for img_name in retrieved_paths]

    if show:
        retrieved_imgs = [Image.open(img_name).convert('RGB').resize((224,224)) for img_name in retrieved_paths]

        n_images = len(retrieved_imgs)
        fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))

        # Ensure axes is iterable
        if n_images == 1:
            axes = [axes]

        for ax, img in zip(axes, retrieved_imgs):
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return retrieved_paths

def vqa(client, question, img_path):
    prompt = question + "\nAnswer the question with a yes or no."

    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path,
            },
            {"type": "text", "text": prompt},
        ]}]

    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(client.device)

    generated_ids = client.generate(**inputs, max_new_tokens=10)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    res = output_text[0].lower().replace('.', '').strip()
    return res


def evaluate_retrieved_images(client, paths, question, verification_question):
    results = []
    verification_results = []  # 1/0
    negation_results = []      # 1/0 (0 if verification failed)

    for path in paths:
        verify_res = vqa(client, verification_question, path)  # 'yes'/'no'
        verify_num = 1 if verify_res == 'yes' else 0
        verification_results.append(verify_num)

        if verify_num == 1:  # context or retrieval is correct, now proceed with negation
            res = vqa(client, question, path)  # 'yes'/'no'
            neg_num = 1 if res == 'no' else 0  # 1 means negation is correct
            negation_results.append(neg_num)

            results.append(neg_num)  # final score matches negation correctness when verified
        else:  # context or retrieval is incorrect
            negation_results.append(0)
            results.append(0)

    return {
        "final": results,
        "verification": verification_results,
        "negation": negation_results,
    }

@torch.no_grad()
def encode_images(database, batch_size, img_dir, preprocess, model):
    img_feats = []

    for b in range(0, len(database), batch_size):
        batch_imgs = database[b:b + batch_size]
        batch_imgs = [os.path.join(img_dir, img_name) for img_name in batch_imgs]
        batch_imgs = [Image.open(im).convert('RGB') for im in batch_imgs]
        batch_imgs = [preprocess(im) for im in batch_imgs]
        batch_imgs = torch.stack(batch_imgs, dim = 0).to(device)
        batch_imgs = model.encode_image(batch_imgs)
        img_feats.append(batch_imgs)
        
    img_feats = torch.cat(img_feats, dim = 0)
    img_feats /= img_feats.norm(dim=-1, keepdim=True)
    print("Finished Encoding {} images!".format(len(database)))
    return img_feats

def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = model.float()
    model.load_state_dict(ckpt["model"])
    return model

data = json.load(open('data/train_data.json', 'r'))   
positive_sentences = list(data.keys())
negative_sentences = list(data.values())


model_name = args.model_name
openclip_model_name = args.openclip_model_name if args.openclip_model_name.lower() != 'none' else None
vqa_model_id = args.vqa_model_id

client = Qwen3VLForConditionalGeneration.from_pretrained(vqa_model_id, dtype=torch.bfloat16, 
                                                         attn_implementation="sdpa", device_map="auto")
client.eval()
processor = AutoProcessor.from_pretrained(vqa_model_id)

# load CLIP baseline
model, preprocess = clip.load(model_name, device, download_root = 'clip_models')

# load the ConCLIP baseline
conclip_model, _ = clip.load(model_name, device,  download_root = 'clip_models')  # uses the same preprocess as original clip
conclip_path = "baseline_models/conclip_models/" + conclip_names_from_model_name[model_name]
conclip_model = load_checkpoint(conclip_model, conclip_path)
conclip_model = conclip_model.to(device)
conclip_model.eval()

if openclip_model_name is not None: # Negbench only provides ViT-B-32 models
    # load CLIP_CC12M model
    pretrained_path = 'baseline_models/negbench_models/CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01.pt'
    # negclip.pth, 
    negbench_model, _, negbench_preprocess = open_clip.create_model_and_transforms(openclip_model_name, pretrained=pretrained_path, device = device, weights_only=False)
    negbench_model.eval()
    openclip_tokenizer = open_clip.get_tokenizer(openclip_model_name)

    # Load NegCLIP model
    pretrained_path = 'baseline_models/negbench_models/negclip.pth'
    negclip_model, _, negclip_preprocess = open_clip.create_model_and_transforms(openclip_model_name, pretrained=pretrained_path, device = device, weights_only=False)
    negclip_model.eval()
    # openclip_tokenizer = open_clip.get_tokenizer(openclip_model_name)

save_folder_name = re.sub(r'[^0-9a-z]', '', model_name.lower())
vqa_model_name = vqa_model_id.split('/')[1]
output_path = 'results/' + save_folder_name + '/' + vqa_model_name + '/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

positive_stream_activations = get_residual_stream_activations(model, positive_sentences)  
positive_stream_activations = concatenate_layer_activations(positive_stream_activations)

negative_stream_activations = get_residual_stream_activations(model, negative_sentences)  
negative_stream_activations = concatenate_layer_activations(negative_stream_activations)

# Train classifiers for each layer
layer_directional_vectors, layer_coefficients = train_classifiers_every_layer(positive_stream_activations, negative_stream_activations)

batch_size = 50
img_dir = 'datasets/val2014'
img_database_size = args.img_database_size    
database = sorted(os.listdir(img_dir))[:img_database_size]
eval_benchmark = json.load(open('data/simpleneg.json', 'r'))

img_feats = encode_images(database, batch_size, img_dir, preprocess, model)
img_feats_conclip = encode_images(database, batch_size, img_dir, preprocess, conclip_model)
if openclip_model_name is not None:
    img_feats_negbench = encode_images(database, batch_size, img_dir, negbench_preprocess, negbench_model)
    img_feats_negclip = encode_images(database, batch_size, img_dir, negclip_preprocess, negclip_model)


baseline_results = {}
conclip_results = {}
steering_results = {}


if openclip_model_name is not None:
    negclip_results = {}
    negbench_results = {}

for sample_idx,(text_input, question, verification_question) in enumerate(eval_benchmark):

    tokenized_text = clip.tokenize(text_input).to(device) 
    
    with torch.no_grad():
        # Baseline (no steering)
        text_feats = model.encode_text(tokenized_text) 
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

        # ConCLIP baseline
        conclip_text_feats = conclip_model.encode_text(tokenized_text)
        conclip_text_feats /= conclip_text_feats.norm(dim=-1, keepdim=True)

        # NegBench and NegCLIP baselines
        if openclip_model_name is not None:
            tokenized_text_openclip = openclip_tokenizer(text_input).to(device) 
            negbench_text_feats = negbench_model.encode_text(tokenized_text_openclip)
            negbench_text_feats /= negbench_text_feats.norm(dim=-1, keepdim=True)

            negclip_text_feats = negclip_model.encode_text(tokenized_text_openclip)
            negclip_text_feats /= negclip_text_feats.norm(dim=-1, keepdim=True)

        # Steering (Ours)
        text_feats_neg = steer_every_layer(model, tokenized_text, layer_directional_vectors, alpha=0.13)
        text_feats_neg /= text_feats_neg.norm(dim=-1, keepdim=True)
        
    
    baseline_paths = show_retrieved_images(img_dir, text_feats, img_feats, database, show = False)
    baseline_sample_results = evaluate_retrieved_images(client, baseline_paths, question, verification_question)
    baseline_results[sample_idx] = baseline_sample_results

    conclip_paths = show_retrieved_images(img_dir, conclip_text_feats, img_feats_conclip, database, show = False)
    conclip_sample_results = evaluate_retrieved_images(client, conclip_paths, question, verification_question)
    conclip_results[sample_idx] = conclip_sample_results

    if openclip_model_name is not None:
        negbench_paths = show_retrieved_images(img_dir, negbench_text_feats, img_feats_negbench, database, show = False)
        negbench_sample_results = evaluate_retrieved_images(client, negbench_paths, question, verification_question)
        negbench_results[sample_idx] = negbench_sample_results
        
        negclip_paths = show_retrieved_images(img_dir, negclip_text_feats, img_feats_negclip, database, show = False)
        negclip_sample_results = evaluate_retrieved_images(client, negclip_paths, question, verification_question)
        negclip_results[sample_idx] = negclip_sample_results
    

    steering_paths = show_retrieved_images(img_dir, text_feats_neg, img_feats, database, show = False)
    steering_sample_results = evaluate_retrieved_images(client, steering_paths, question, verification_question)
    steering_results[sample_idx] = steering_sample_results

    with open(output_path + 'baseline_results.json', 'w') as f:
        json.dump(baseline_results, f)

    with open(output_path + 'conclip_results.json', 'w') as f:
        json.dump(conclip_results, f)

    with open(output_path + 'steering_results.json', 'w') as f:
        json.dump(steering_results, f)

    if openclip_model_name is not None:
        with open(output_path + 'negbench_results.json', 'w') as f:
            json.dump(negbench_results, f)

        with open(output_path + 'negclip_results.json', 'w') as f:
            json.dump(negclip_results, f)

    print(f"Finished Evaluating {sample_idx + 1}/{len(eval_benchmark)}")
