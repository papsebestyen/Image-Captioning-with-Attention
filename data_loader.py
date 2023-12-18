import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import json
from pathlib import Path


def get_loader(
    transform,
    data_path: Path,
    mode="train",
    # default batch size
    batch_size=1,
    vocab_threshold=None,
    vocab_file="./vocab.pkl",
    start_word="<start>",
    end_word="<end>",
    unk_word="<unk>",
    vocab_from_file=True,
    num_workers=0,
):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train', 'valid or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary.
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading
    """
    assert mode in [
        "train",
        "dev",
        "test",
    ], "mode must be one of 'train', 'dev' or 'test'."
    if vocab_from_file == False:
        assert (
            mode == "train"
        ), "To generate vocab from captions file, must be in training mode (mode='train')."

    img_folder = data_path / "Flickr8k_Dataset"
    ann_foler = data_path / "Flickr8k_text"
    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == "train":
        if vocab_from_file == True:
            assert os.path.exists(
                vocab_file
            ), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."

    elif mode == "dev":
        assert os.path.exists(
            vocab_file
        ), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."

    elif mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 for testing your model."
        assert os.path.exists(
            vocab_file
        ), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."

    # COCO caption dataset.
    dataset = Flickr8kDataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        vocab_from_file=vocab_from_file,
        img_folder=img_folder,
        ann_folder=ann_foler,
    )

    if mode == "train" or mode == "dev":
        # Randomly sample a caption length and indices of that length
        indices = dataset.get_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices
        # functionality from torch.utils
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=data.sampler.BatchSampler(
                sampler=initial_sampler, batch_size=dataset.batch_size, drop_last=False
            ),
        )
    elif mode == "test":
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=dataset.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    return data_loader


class Flickr8kDataset(data.Dataset):
    def __init__(
        self,
        transform,
        mode,
        batch_size,
        vocab_threshold,
        vocab_file,
        start_word,
        end_word,
        unk_word,
        vocab_from_file,
        img_folder,
        ann_folder,
    ):
        # transform - defined transformation (e.g. Rescale, ToTensor, RandomCrop and etc.)
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.img_folder: Path = img_folder
        self.ann_folder: Path = ann_folder
        self.annotations = dict()
        self.img_ids = list()
        self.ann_ids = list()
        self.img_ann_ids = dict()

        self._load_annotations()

        self.vocab = Vocabulary(
            vocab_threshold=vocab_threshold,
            annotations=self.annotations,
            vocab_file=vocab_file,
            start_word=start_word,
            end_word=end_word,
            unk_word=unk_word,
            vocab_from_file=vocab_from_file,
        )

        all_tokens = [
            nltk.tokenize.word_tokenize(
                str(self.annotations[self.ann_ids[index]]).lower()
            )
            for index in tqdm(np.arange(len(self.ann_ids)))
        ]
        # list of token lengths (number of words for each caption)
        self.caption_lengths = [len(token) for token in all_tokens]

    def _load_annotations(self):
        self.img_ids = (
            (self.ann_folder / f"Flickr_8k.{self.mode}Images.txt")
            .read_text()
            .split("\n")
        )
        for row in (
            (self.ann_folder / "Flickr8k.token.txt").read_text().strip().split("\n")
        ):
            ann_id, caption = row.split("\t")
            img_id, _ = ann_id.split("#")
            if img_id in self.img_ids:
                self.ann_ids.append(ann_id)
                self.annotations[ann_id] = caption
                self.img_ann_ids.setdefault(img_id, list()).append(ann_id)

    def __getitem__(self, index):
        ann_id = self.ann_ids[index]
        caption = self.annotations[ann_id]
        path = ann_id.split("#")[0]

        PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
        orig_image = np.array(PIL_image)
        image = self.transform(PIL_image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()

        caps_all = []
        ids_ann = self.img_ann_ids[path]
        for ann_id in ids_ann:
            capt = self.annotations[ann_id]
            caps_all.append(capt)

        if self.mode == "train":
            return image, caption
        elif self.mode == "dev":
            return image, caption, caps_all
        elif self.mode == "test":
            return orig_image, image

    def get_indices(self):
        # randomly select the caption length from the list of lengths
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where(
            [
                self.caption_lengths[i] == sel_length
                for i in np.arange(len(self.caption_lengths))
            ]
        )[0]
        # select m = batch_size captions from list above
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        # return the caption indices of specified batch
        return indices

    def __len__(self):
        return len(self.ann_ids)
