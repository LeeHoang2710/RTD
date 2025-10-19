import json
import pickle
from argparse import ArgumentParser
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from clip.clip import load, tokenize
from transformers import CLIPTextModelWithProjection
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from data_utils import collate_fn, PROJECT_ROOT, targetpad_transform
from loader import FashionIQDataset
from encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF
from models import build_text_encoder, Phi, PIC2WORD
from utils import extract_image_features, device, extract_pseudo_tokens_with_phi

torch.multiprocessing.set_sharing_strategy('file_system')


@torch.no_grad()
def fiq_generate_val_predictions(clip_model, relative_val_dataset: Dataset, ref_names_list: List[str],
                                 pseudo_tokens: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
    """
    Generates features predictions for the validation set of Fashion IQ.
    """

    # Create data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=10,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    target_names_list = []

    # Compute features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_captions']

        flattened_captions: list = np.array(relative_captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        input_captions_reversed = [
            f"{flattened_captions[i + 1].strip('.?, ')} and {flattened_captions[i].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        input_captions = [
            f"a photo of $ that {in_cap}" for in_cap in input_captions]
        batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        tokenized_input_captions = tokenize(input_captions, context_length=77).to(device)
        text_features = encode_with_pseudo_tokens_HF(clip_model, tokenized_input_captions, batch_tokens)

        input_captions_reversed = [
            f"a photo of $ that {in_cap}" for in_cap in input_captions_reversed]
        tokenized_input_captions_reversed = tokenize(input_captions_reversed, context_length=77).to(device)
        text_features_reversed = encode_with_pseudo_tokens_HF(clip_model, tokenized_input_captions_reversed,
                                                           batch_tokens)

        predicted_features = F.normalize((F.normalize(text_features) + F.normalize(text_features_reversed)) / 2)
        # predicted_features = F.normalize((text_features + text_features_reversed) / 2)

        predicted_features_list.append(predicted_features)
        target_names_list.extend(target_names)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, target_names_list


@torch.no_grad()
def fiq_compute_val_metrics(relative_val_dataset: Dataset, clip_model, index_features: torch.Tensor,
                            index_names: List[str], ref_names_list: List[str], pseudo_tokens: torch.Tensor) \
        -> Dict[str, float]:
    """
    Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, target_names = fiq_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list,
                                                                    pseudo_tokens)

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    # Compute the distances
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return {'fiq_recall_at10': recall_at10,
            'fiq_recall_at50': recall_at50}


@torch.no_grad()
def fiq_val_retrieval(dataset_path: str, dress_type: str, image_encoder, text_encoder, ref_names_list: List[str],
                      pseudo_tokens: torch.Tensor, preprocess: callable) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
    """
    # Load the model
    #clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    #clip_model = clip_model.float().eval().requires_grad_(False)

    # Extract the index features
    classic_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_val_dataset, image_encoder)

    # Define the relative dataset
    relative_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'relative', preprocess)

    return fiq_compute_val_metrics(relative_val_dataset, text_encoder, index_features, index_names, ref_names_list,
                                   pseudo_tokens)




def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    #### RTD
    parser.add_argument("--eval-type", type=str, choices=['phi', 'pic2word'], required=True,
                        help=
                             "if 'phi' predicts the pseudo tokens using the phi network (LinCIR or SEARLE), "
                             "if 'pic2word' uses the pre-trained pic2word model to predict the pseudo tokens, "
                        )
    ####
    parser.add_argument("--dataset", type=str, required=True, choices=['cirr', 'fashioniq', 'circo'],
                        help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--preprocess-type", default="clip", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument("--phi-checkpoint-name", type=str,
                        help="Phi checkpoint to use, needed when using phi, e.g. 'phi_20.pt'")
    parser.add_argument("--clip_model_name", default="giga", type=str)
    parser.add_argument("--cache_dir", default="./hf_models", type=str)

    parser.add_argument("--l2_normalize", action="store_true", help="Whether or not to use l2 normalization")

    #### RTD
    parser.add_argument("--text-encoder-checkpoint-name", type=str,
                        help="text checkpoint to use, needed when using phi, e.g. 'text_encoder_best.pt'")
    ####
    args = parser.parse_args()

    if args.eval_type == 'phi' and args.phi_checkpoint_name is None:
        raise ValueError("Phi checkpoint name is required when using phi evaluation type")

    # if args.eval_type == 'oti':
    #     experiment_path = PROJECT_ROOT / 'data' / "oti_pseudo_tokens" / args.dataset.lower() / 'val' / args.exp_name
    #     if not experiment_path.exists():
    #         raise ValueError(f"Experiment {args.exp_name} not found")

    #     with open(experiment_path / 'hyperparameters.json') as f:
    #         hyperparameters = json.load(f)

    #     pseudo_tokens = torch.load(experiment_path / 'ema_oti_pseudo_tokens.pt', map_location=device)
    #     with open(experiment_path / 'image_names.pkl', 'rb') as f:
    #         ref_names_list = pickle.load(f)

    #     clip_model_name = hyperparameters['clip_model_name']
    #     clip_model, clip_preprocess = clip.load(clip_model_name, device='cpu', jit=False)

    #     if args.preprocess_type == 'targetpad':
    #         print('Target pad preprocess pipeline is used')
    #         preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
    #     elif args.preprocess_type == 'clip':
    #         print('CLIP preprocess pipeline is used')
    #         preprocess = clip_preprocess
    #     else:
    #         raise ValueError("Preprocess type not supported")


    if args.eval_type in ['phi', 'pic2word']:
        if args.eval_type == 'phi':
            args.mixed_precision = 'fp16'
            image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args)
            #### RTD
            if args.text_encoder_checkpoint_name:
                text_encoder.load_state_dict(
                    torch.load(args.text_encoder_checkpoint_name, map_location=device)[text_encoder.__class__.__name__])
                text_encoder=text_encoder.eval()
            ####
            phi = Phi(input_dim=text_encoder.config.projection_dim,
                      hidden_dim=text_encoder.config.projection_dim * 4,
                      output_dim=text_encoder.config.hidden_size, dropout=0.5).to(device)

            phi.load_state_dict(
                    torch.load(args.phi_checkpoint_name, map_location=device)[phi.__class__.__name__])

            phi = phi.eval()

        elif args.eval_type == 'pic2word':
            args.mixed_precision = 'fp16'
            image_encoder, clip_preprocess, text_encoder, tokenizer = build_text_encoder(args)
            #### RTD
            if args.text_encoder_checkpoint_name:
                text_encoder.load_state_dict(
                    torch.load(args.text_encoder_checkpoint_name, map_location=device)[
                    text_encoder.__class__.__name__])
                text_encoder=text_encoder.eval()
            ####
            
            phi = PIC2WORD(embed_dim=text_encoder.config.projection_dim,
                           output_dim=text_encoder.config.hidden_size,
                           ).to(device)
            sd = torch.load(args.phi_checkpoint_name, map_location=device)['state_dict_img2text']
            sd = {k[len('module.'):]: v for k, v in sd.items()}
            phi.load_state_dict(sd)
            phi = phi.eval()

        else:  # searle or searle-xl
            if args.eval_type == 'searle':
                clip_model_name = 'ViT-B/32'
            else:  # args.eval_type == 'searle-xl':
                clip_model_name = 'ViT-L/14'
            phi, _ = torch.hub.load(repo_or_dir='miccunifi/SEARLE', model='searle', source='github',
                                    backbone=clip_model_name)
            phi = phi.to(device).eval()
            clip_model, clip_preprocess = load(clip_model_name, device=device, jit=False)

        if args.preprocess_type == 'targetpad':
            print('Target pad preprocess pipeline is used')
            preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
        elif args.preprocess_type == 'clip':
            print('CLIP preprocess pipeline is used')
            preprocess = clip_preprocess
        else:
            raise ValueError("Preprocess type not supported")

        if args.dataset.lower() == 'fashioniq':
            relative_val_dataset = FashionIQDataset(args.dataset_path, 'val', ['dress', 'toptee', 'shirt'], 'relative', preprocess, no_duplicates=True)
        else:
            raise ValueError("Dataset not supported")

        #clip_model = clip_model.float().to(device)
        image_encoder = image_encoder.float().to(device)
        text_encoder = text_encoder.float().to(device)
        pseudo_tokens, ref_names_list = extract_pseudo_tokens_with_phi(image_encoder, phi, relative_val_dataset, args)
        pseudo_tokens = pseudo_tokens.to(device)
    else:
        raise ValueError("Eval type not supported")

    print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")
    if args.dataset.lower() == 'fashioniq':
        recalls_at10 = []
        recalls_at50 = []
        for dress_type in ['shirt', 'dress', 'toptee']:
            fiq_metrics = fiq_val_retrieval(args.dataset_path, dress_type, image_encoder, text_encoder, ref_names_list,pseudo_tokens, preprocess)
            recalls_at10.append(fiq_metrics['fiq_recall_at10'])
            recalls_at50.append(fiq_metrics['fiq_recall_at50'])

            for k, v in fiq_metrics.items():
                print(f"{dress_type}_{k} = {v:.2f}")
            print("\n")

        print(f"average_fiq_recall_at10 = {np.mean(recalls_at10):.2f}")
        print(f"average_fiq_recall_at50 = {np.mean(recalls_at50):.2f}")

if __name__ == '__main__':
    main()
