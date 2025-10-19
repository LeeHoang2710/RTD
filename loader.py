'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
import os
import functools
import glob
import random
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal
import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset
import spacy
import numpy as np
import sng_parser
import datasets



#### RTD 
class CaptionDataset_RTD(Dataset):
    def __init__(self, reference_captions, target_captions, conditioning_captions, tokenizer, spacy_nlp):
        self.reference_captions = reference_captions
        self.target_captions = target_captions
        self.conditioning_captions = conditioning_captions

        self.tokenizer = tokenizer
        self.spacy_nlp = spacy_nlp

    def __len__(self):
        return len(self.reference_captions)

    def __getitem__(self, idx):
        # reference caption
        reference_caption = self.reference_captions[idx]
        reference_caption = clean_caption(reference_caption, self.tokenizer)
        reference_caption_dict = self.tokenizer(text=reference_caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        reference_tokens, _ = reference_caption_dict['input_ids'][0], reference_caption_dict['attention_mask'][0]
        
        # target caption
        target_caption = self.target_captions[idx]
        target_caption = clean_caption(target_caption, self.tokenizer)
        target_caption_dict = self.tokenizer(text=target_caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        target_tokens, _ = target_caption_dict['input_ids'][0], target_caption_dict['attention_mask'][0]
        
        # concatenated caption
        concat_caption = 'a photo of [$] that ' +  self.conditioning_captions[idx]        
        concat_token_dict = self.tokenizer(text=concat_caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        concat_tokens, _ = concat_token_dict['input_ids'][0], concat_token_dict['attention_mask'][0]
        concat_tokens = torch.where(concat_tokens == 49408, torch.ones_like(concat_tokens) * 259, concat_tokens)
        # special caption 
        special_caption = 'a photo of [$]'        
        special_token_dict = self.tokenizer(text=special_caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        special_tokens, _ = special_token_dict['input_ids'][0], special_token_dict['attention_mask'][0]
        special_tokens = torch.where(special_tokens == 49408, torch.ones_like(special_tokens) * 259, special_tokens)
        
        return concat_tokens, target_tokens, reference_tokens, special_tokens 

def build_loader_RTD(args, tokenizer, accelerator):

    db_relative= datasets.load_dataset("json", data_files=args.caption_dir)
    reference_captions = db_relative['train']['reference_caption']    
    target_captions = db_relative['train']['target_caption']
    conditioning_captions = db_relative['train']['conditioning_caption']

    # in dataset: concat tokens, target_tokens, reference_tokens, special_tokens
    dataset = CaptionDataset_RTD(reference_captions,target_captions,conditioning_captions, tokenizer, spacy.load('en_core_web_sm'))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)

    return data_loader

####
def extract_keywords(spacy_nlp, caption):
    candidates = []
    nlp_caption = caption

    doc = spacy_nlp(nlp_caption)

    tmp = ''
    for word in doc:
        if word.pos_ == 'ADJ':
            if tmp == '':
                tmp += word.text
            else:
                tmp += ' ' + word.text
        elif word.pos_ == 'NOUN' or word.pos_ == 'PROPN':
            if tmp == '':
                tmp += word.text
            else:
                tmp += ' ' + word.text
        else:
            if tmp != '':
                candidates.append(tmp)
            tmp = ''
    if tmp != '':
        candidates.append(tmp)

    candidates = list(set(candidates))

    return candidates


def extract_keywords_spacy(spacy_nlp, caption):
    sequences = []
    current_sequence = []
    doc = spacy_nlp(caption)
    for token in doc:
        # Check if the token is a noun, proper noun, or adjective
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'DET']:
            current_sequence.append(token.text)
        else:
            # If we encounter a token that's not one of the desired POS and current_sequence is not empty
            if current_sequence:
                sequences.append(" ".join(current_sequence))
                current_sequence = []

    # Adding any remaining sequence after the loop
    if current_sequence:
        sequences.append(" ".join(current_sequence))

    return sequences


def extract_sng(caption):
    graph = sng_parser.parse(caption)
    entities = [x['head'] for i, x in enumerate(graph['entities'])]
    relations = [{'subject': entities[x['subject']], 'object': entities[x['object']], 'relation': x['relation']} for x in graph['relations']]
    return entities, relations


def clean_caption(caption, tokenizer):
    if caption is None:
        caption = ''
    if '<PERSON>' in caption: # to handle with GCC12M
        caption = caption.replace('<PERSON>', 'person')
    caption = caption.lower().replace('$', '').strip()
    tokens = tokenizer.encode(caption, padding='longest', return_tensors='pt')
    if tokens.shape[1] > 77:
        caption = tokenizer.batch_decode(tokens[:,1:76])[0]
    return caption


def preprocess_precomputed_base(sample, spacy_nlp, keywords_list, tokenizer):
    '''
    'image_feature.npy','json'
    '''
    image_feature, image_feature_giga, meta = sample

    caption = clean_caption(meta['source_caption'], tokenizer)

    keywords = ['']
    try:
        keywords = extract_keywords_spacy(spacy_nlp, caption)
    except Exception as e:
        #print(e)
        pass

    # for keywords
    indicator = 1
    replaced_caption = caption
    for keyword in keywords:
        if keyword != '' and keyword in caption:
            replaced_caption = replaced_caption.replace(keyword, '[$]')
        else:
            tmp_keywords = caption.split(' ')
            if len(tmp_keywords) > 0:
                selected_keywords = random.sample(tmp_keywords, k=min(int(len(tmp_keywords) * 1.0), 1))
                for selected_keyword in selected_keywords:
                    replaced_caption = replaced_caption.replace(selected_keyword, '[$]')
            else:
                replaced_caption = f'a photo of [$] that {caption}'
                indicator = 0
            break

    token_dict = tokenizer(text=caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
    tokens, attention_mask = token_dict['input_ids'][0], token_dict['attention_mask'][0]

    replaced_token_dict = tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
    replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]

    replaced_tokens = torch.where(replaced_tokens == 49408,
                                  torch.ones_like(replaced_tokens) * 259,
                                  replaced_tokens)

    if 259 not in replaced_tokens:
        replaced_caption = 'a photo of [$]'
        replaced_token_dict = tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]

        replaced_tokens = torch.where(replaced_tokens == 49408,
                                      torch.ones_like(replaced_tokens) * 259,
                                      replaced_tokens)
        indicator = 0

    new_sample = [tokens, replaced_tokens, indicator]

    return tuple(new_sample)


class CaptionDataset(Dataset):
    def __init__(self, captions, tokenizer, spacy_nlp):
        self.captions = captions
        self.tokenizer = tokenizer
        self.spacy_nlp = spacy_nlp

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]

        caption = clean_caption(caption, self.tokenizer)

        keywords = [""]
        try:
            keywords = extract_keywords_spacy(self.spacy_nlp, caption)
        except Exception as e:
            #print(e)
            pass

        # for keywords
        indicator = 1
        replaced_caption = caption

        if len(keywords) == 0:
            keywords = [""]

        for keyword in keywords:
            if keyword != '' and keyword in caption:
                replaced_caption = replaced_caption.replace(keyword, '[$]')
            else:
                tmp_keywords = caption.split(' ')
                if len(tmp_keywords) > 0:
                    selected_keywords = random.sample(tmp_keywords, k=min(int(len(tmp_keywords) * 1.0), 1))
                    for selected_keyword in selected_keywords:
                        replaced_caption = replaced_caption.replace(selected_keyword, '[$]')
                else:
                    replaced_caption = f'a photo of [$] that {caption}'
                    indicator = 0
                break

        token_dict = self.tokenizer(text=caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        tokens, attention_mask = token_dict['input_ids'][0], token_dict['attention_mask'][0]

        replaced_token_dict = self.tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]

        replaced_tokens = torch.where(replaced_tokens == 49408,
                                      torch.ones_like(replaced_tokens) * 259,
                                      replaced_tokens)

        if 259 not in replaced_tokens:
            replaced_caption = 'a photo of [$]'
            replaced_token_dict = self.tokenizer(text=replaced_caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
            replaced_tokens, replaced_attention_mask = replaced_token_dict['input_ids'][0], replaced_token_dict['attention_mask'][0]

            replaced_tokens = torch.where(replaced_tokens == 49408,
                                          torch.ones_like(replaced_tokens) * 259,
                                          replaced_tokens)
            indicator = 0

        return tokens, replaced_tokens, indicator


def build_loader(args, tokenizer, accelerator):
    data_names = {'dataset1': 'dangne/gcc_caption_only',
                  'dataset2': 'FredZhang7/stable-diffusion-prompts-2.47M',
                  'dataset3': 'Geonmo/midjourney-prompts-only',
                  }

    for k, v in data_names.items():
        if not os.path.exists(os.path.join('./datasets', k)):
            if accelerator.is_main_process:
                print('Downloading captions is required')
                db = datasets.load_dataset(v, cache_dir=os.path.join('./datasets', k))

    captions = []
    for k, v in data_names.items():
        db = datasets.load_dataset(v, cache_dir=os.path.join('./datasets', k))
        captions += db['train']['text']

    dataset = CaptionDataset(captions, tokenizer, spacy.load('en_core_web_sm'))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)

    return data_loader


class FashionIQDataset(Dataset):
    """
    Copy-paste from https://github.com/miccunifi/SEARLE/blob/main/src/datasets.py
    FashionIQ dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield :a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions'] when
             split in ['train', 'val']
            - ['reference_image', 'reference_name', 'relative_captions'] when split == test
    """

    def __init__(self, dataset_path: Union[Path, str], split: Literal['train', 'val', 'test'], dress_types: List[str],
                 mode: Literal['relative', 'classic'], preprocess: callable, no_duplicates: Optional[bool] = False):
        """
        :param dataset_path: path to the FashionIQ dataset
        :param split: dataset split, should be in ['train, 'val', 'test']
        :param dress_types: list of fashionIQ categories, each category should be in ['dress', 'shirt', 'toptee']
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
            - In 'relative' mode the dataset yield dict with keys:
                - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions']
                 when split in ['train', 'val']
                - ['reference_image', 'reference_name', 'relative_captions'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.no_duplicates = no_duplicates

        # Validate the inputs
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(dataset_path / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # Remove duplicates from
        if self.no_duplicates:
            seen = set()
            new_triplets = []
            for triplet in self.triplets:
                if triplet['candidate'] not in seen:
                    seen.add(triplet['candidate'])
                    new_triplets.append(triplet)
            self.triplets = new_triplets

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(dataset_path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        try:
            if self.mode == 'relative':
                relative_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split in ['train', 'val']:
                    reference_image_path = self.dataset_path / 'images' / f"{reference_name}.jpg"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path), return_tensors='pt')['pixel_values'][0]
                    target_name = self.triplets[index]['target']
                    target_image_path = self.dataset_path / 'images' / f"{target_name}.jpg"
                    target_image = self.preprocess(PIL.Image.open(target_image_path), return_tensors='pt')['pixel_values'][0]

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'target_image': target_image,
                        'target_name': target_name,
                        'relative_captions': relative_captions
                    }

                elif self.split == 'test':
                    reference_image_path = self.dataset_path / 'images' / f"{reference_name}.jpg"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path), return_tensors='pt')['pixel_values'][0]

                    return {
                        'reference_image': reference_image,
                        'reference_name': reference_name,
                        'relative_captions': relative_captions
                    }

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = self.dataset_path / 'images' / f"{image_name}.jpg"
                image = self.preprocess(PIL.Image.open(image_path), return_tensors='pt')['pixel_values'][0]

                return {
                    'image': image,
                    'image_name': image_name
                }

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

