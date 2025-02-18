import nltk
import pickle
import os.path
from collections import Counter


class Vocabulary(object):
    # similar to get loader function from data_loader.py
    def __init__(
        self,
        vocab_threshold,
        annotations,
        vocab_file="./vocab.pkl",
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        vocab_from_file=False,
    ):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.vocab_from_file = vocab_from_file
        self.get_vocab(annotations)

    def get_vocab(self, annotations):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        # if vocab_file (vocab.pkl) exists and we specified vocab_from_file = True
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            # open vocab.pkl fike in binary format for reading (rb)
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                # vocab.pkl file will have attributes, such as word2idx and idx2word
                # we load vocabulary from file
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            # build_vocab function will build vocabulary from scratch
            # by adding start, end, unknown words + captions
            self.build_vocab(annotations)
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocab(self, annotations):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions(annotations=annotations)

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self, annotations):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        counter = Counter()
        ids = annotations.keys()
        for i, id in enumerate(ids):
            tokens = str(annotations[id])
            counter.update(tokens)

            if i % 5000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
