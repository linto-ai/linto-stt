# coding=utf-8

"""recasepunc file."""

import argparse
import os
import random
import sys
import re
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mosestokenizer import *
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertTokenizer

default_config = argparse.Namespace(
    seed=871253,
    lang='fr',
    # flavor='flaubert/flaubert_base_uncased',
    flavor=None,
    max_length=256,
    batch_size=16,
    updates=24000,
    period=1000,
    lr=1e-5,
    dab_rate=0.1,
    device='cuda',
    debug=False
)

default_flavors = {
    'fr': 'flaubert/flaubert_base_uncased',
    'en': 'bert-base-uncased',
    'zh': 'ckiplab/bert-base-chinese',
    'it': 'dbmdz/bert-base-italian-uncased',
}


class Config(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in default_config.__dict__.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert self.lang in ['fr', 'en', 'zh', 'it']

        if 'lang' in kwargs and ('flavor' not in kwargs or kwargs['flavor'] is None):
            self.flavor = default_flavors[self.lang]

        # print(self.lang, self.flavor)


def init_random(seed):
    # make sure everything is deterministic
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    set_seed(seed)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# NOTE: it is assumed in the implementation that y[:,0] is the punctuation label, and y[:,1] is the case label!

punctuation = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3,
    'EXCLAMATION': 4,
}

punctuation_syms = ['', ',', '.', ' ?', ' !']

case = {
    'LOWER': 0,
    'UPPER': 1,
    'CAPITALIZE': 2,
    'OTHER': 3,
}


class Model(nn.Module):
    def __init__(self, flavor, device):
        super().__init__()
        self.bert = AutoModel.from_pretrained(flavor)
        # need a proper way of determining representation size
        size = self.bert.dim \
            if hasattr(self.bert, 'dim') else self.bert.config.pooler_fc_size \
            if hasattr(self.bert.config, 'pooler_fc_size') else self.bert.config.emb_dim \
            if hasattr(self.bert.config, 'emb_dim') else self.bert.config.hidden_size
        self.punc = nn.Linear(size, 5)
        self.case = nn.Linear(size, 4)
        self.dropout = nn.Dropout(0.3)
        self.to(device)

    def forward(self, x):
        output = self.bert(x)
        representations = self.dropout(F.gelu(output['last_hidden_state']))
        punc = self.punc(representations)
        case = self.case(representations)
        return punc, case

def recase(token, label):
    if label == case['LOWER']:
        return token.lower()
    if label == case['CAPITALIZE']:
        return token.lower().capitalize()
    if label == case['UPPER']:
        return token.upper()
    return token

_PUNCTUATION_MODEL = None

def load_recasepunc_model(config=None):

    global _PUNCTUATION_MODEL
    memoize = (config is None)
    if memoize and _PUNCTUATION_MODEL is not None:
        return _PUNCTUATION_MODEL

    checkpoint_path = os.environ.get('PUNCTUATION_MODEL')
    if not checkpoint_path:
        return None

    if config is None:
        config = default_config

    device = os.environ.get("DEVICE")
    if device is None:
        if torch.cuda.is_available():
            config.device = 'cuda'
        else:
            config.device = 'cpu'

    print(f"Loading recasepunc model from {checkpoint_path} on device={config.device}") # TODO: use logger.info

    loaded = torch.load(checkpoint_path, map_location=config.device)
    if 'config' in loaded:
        config = Config(**loaded['config'])

    if config.flavor is None:
        config.flavor = default_flavors[config.lang]

    init(config)

    model = Model(config.flavor, config.device)
    model.load_state_dict(loaded['model_state_dict'])

    config.model = model

    if memoize:
        _PUNCTUATION_MODEL = config
    return config


def apply_recasepunc(config, line, ignore_disfluencies=False):

    num_threads = os.environ.get("OMP_NUM_THREADS")
    if num_threads:
        torch.set_num_threads(int(num_threads))

    if isinstance(line, list):
        return [apply_recasepunc(config, l, ignore_disfluencies=ignore_disfluencies) for l in line]
    
    if isinstance(line, dict):
        new_dict = line.copy()
        assert "text" in line
        line = line["text"]
        line = apply_recasepunc(config, line, ignore_disfluencies=ignore_disfluencies)
        new_dict["text"] = line
        return new_dict

    assert isinstance(line, str)
    line = line.strip()

    if line.startswith("{") and line.endswith("}"):
        # A dict inside a string
        line = json.loads(line)
        assert isinstance(line, dict)
        return json.dumps(apply_recasepunc(config, line, ignore_disfluencies=ignore_disfluencies), indent=2, ensure_ascii=False)
    
    if not line:
        # Avoid hanging on empty lines
        return ""

    # Remove <unk> tokens (Ugly: LinTO/Kaldi model specific here)
    line = re.sub("<unk> ", "", line)

    if config is None:
        return line

    model = config.model
    set_seed(config.seed)

    # Drop all punctuation that can be generated
    line = ''.join([c for c in line if c not in mapped_punctuation])

    # Relevant only if disfluences annotations
    if ignore_disfluencies:
        # TODO: fix when there are several disfluencies in a row ("euh euh")
        line = collapse_whitespace(line)
        line = re.sub(r"(\w) *' *(\w)", r"\1'\2", line) # glue apostrophes to words
        disfluencies, line = remove_simple_disfluences(line)

    output = ''
    if config.debug:
        print(line)
    tokens = [config.cls_token] + config.tokenizer.tokenize(line) + [config.sep_token]
    if config.debug:
        print(tokens)
    previous_label = punctuation['PERIOD']
    first_time = True
    was_word = False
    for start in range(0, len(tokens), config.max_length):
        instance = tokens[start: start + config.max_length]
        ids = config.tokenizer.convert_tokens_to_ids(instance)
        if len(ids) < config.max_length:
            ids += [config.pad_token_id] * (config.max_length - len(ids))
        x = torch.tensor([ids]).long().to(config.device)
        y_scores1, y_scores2 = model(x)
        y_pred1 = torch.max(y_scores1, 2)[1]
        y_pred2 = torch.max(y_scores2, 2)[1]
        for id, token, punc_label, case_label in zip(ids, instance, y_pred1[0].tolist()[:len(instance)],
                                                        y_pred2[0].tolist()[:len(instance)]):
            if config.debug:
                print(id, token, punc_label, case_label, file=sys.stderr)
            if id in (config.cls_token_id, config.sep_token_id):
                continue
            if previous_label is not None and previous_label > 1:
                if case_label in [case['LOWER'], case['OTHER']]:
                    case_label = case['CAPITALIZE']
            previous_label = punc_label
            # different strategy due to sub-lexical token encoding in Flaubert
            if config.lang == 'fr':
                if token.endswith('</w>'):
                    cased_token = recase(token[:-4], case_label)
                    if was_word:
                        output += ' '
                    output += cased_token + punctuation_syms[punc_label]
                    was_word = True
                else:
                    cased_token = recase(token, case_label)
                    if was_word:
                        output += ' '
                    output += cased_token
                    was_word = False
            else:
                if token.startswith('##'):
                    cased_token = recase(token[2:], case_label)
                    output += cased_token
                else:
                    cased_token = recase(token, case_label)
                    if not first_time:
                        output += ' '
                    first_time = False
                    output += cased_token + punctuation_syms[punc_label]
    if previous_label == 0:
        output += '.'
    # Glue apostrophes back to words
    output = re.sub(r"(\w) *' *(\w)", r"\1'\2", output)

    if ignore_disfluencies:
        output = collapse_whitespace(output)
        output = reconstitute_text(output, disfluencies)
    return output

mapped_punctuation = {
    '.': 'PERIOD',
    '...': 'PERIOD',
    ',': 'COMMA',
    ';': 'COMMA',
    ':': 'COMMA',
    '(': 'COMMA',
    ')': 'COMMA',
    '?': 'QUESTION',
    '!': 'EXCLAMATION',
    '，': 'COMMA',
    '！': 'EXCLAMATION',
    '？': 'QUESTION',
    '；': 'COMMA',
    '：': 'COMMA',
    '（': 'COMMA',
    '(': 'COMMA',
    '）': 'COMMA',
    '［': 'COMMA',
    '］': 'COMMA',
    '【': 'COMMA',
    '】': 'COMMA',
    '└': 'COMMA',
    #'└ ': 'COMMA',
    '_': 'O',
    '。': 'PERIOD',
    '、': 'COMMA',  # enumeration comma
    '、': 'COMMA',
    '…': 'PERIOD',
    '—': 'COMMA',
    '「': 'COMMA',
    '」': 'COMMA',
    '．': 'PERIOD',
    '《': 'O',
    '》': 'O',
    '，': 'COMMA',
    '“': 'O',
    '”': 'O',
    '"': 'O',
    #'-': 'O', # hyphen is a word piece
    '〉': 'COMMA',
    '〈': 'COMMA',
    '↑': 'O',
    '〔': 'COMMA',
    '〕': 'COMMA',
}

def collapse_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


# modification of the wordpiece tokenizer to keep case information even if vocab is lower cased
# forked from https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py

class WordpieceTokenizer:
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100, keep_case=True):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.keep_case = keep_case

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in text.strip().split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    # optionaly lowercase substring before checking for inclusion in vocab
                    if (self.keep_case and substr.lower() in self.vocab) or (substr in self.vocab):
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


# modification of XLM bpe tokenizer for keeping case information when vocab is lowercase
# forked from https://github.com/huggingface/transformers/blob/cd56f3fe7eae4a53a9880e3f5e8f91877a78271c/src/transformers/models/xlm/tokenization_xlm.py
def bpe(self, token):
    def to_lower(pair):
        # print('  ',pair)
        return (pair[0].lower(), pair[1].lower())

    from transformers.models.xlm.tokenization_xlm import get_pairs

    word = tuple(token[:-1]) + (token[-1] + "</w>",)
    if token in self.cache:
        return self.cache[token]
    pairs = get_pairs(word)

    if not pairs:
        return token + "</w>"

    while True:
        bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(to_lower(pair), float("inf")))
        # print(bigram)
        if to_lower(bigram) not in self.bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new_word.extend(word[i:])
                break
            else:
                new_word.extend(word[i:j])
                i = j

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        pairs = get_pairs(word)
    word = " ".join(word)
    if word == "\n  </w>":
        word = "\n</w>"
    self.cache[token] = word
    return word


def init(config):
    init_random(config.seed)

    if config.lang == 'fr':
        config.tokenizer = tokenizer = AutoTokenizer.from_pretrained(config.flavor, do_lower_case=False)

        from transformers.models.xlm.tokenization_xlm import XLMTokenizer
        assert isinstance(tokenizer, XLMTokenizer)

        # monkey patch XLM tokenizer
        import types
        tokenizer.bpe = types.MethodType(bpe, tokenizer)
    else:
        # warning: needs to be BertTokenizer for monkey patching to work
        config.tokenizer = tokenizer = BertTokenizer.from_pretrained(config.flavor, do_lower_case=False)

        # warning: monkey patch tokenizer to keep case information
        # from recasing_tokenizer import WordpieceTokenizer
        config.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token)

    if config.lang == 'fr':
        config.pad_token_id = tokenizer.pad_token_id
        config.cls_token_id = tokenizer.bos_token_id
        config.cls_token = tokenizer.bos_token
        config.sep_token_id = tokenizer.sep_token_id
        config.sep_token = tokenizer.sep_token
    else:
        config.pad_token_id = tokenizer.pad_token_id
        config.cls_token_id = tokenizer.cls_token_id
        config.cls_token = tokenizer.cls_token
        config.sep_token_id = tokenizer.sep_token_id
        config.sep_token = tokenizer.sep_token

    if not torch.cuda.is_available() and config.device == 'cuda':
        print('WARNING: reverting to cpu as cuda is not available', file=sys.stderr)
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

def remove_simple_disfluences(text, language=None):
    if language is None:
        # Get language from environment
        language = os.environ.get("LANGUAGE","")
    language = language.lower()[:2]
    disfluencies = DISFLUENCIES.get(language, [])
    all_hits = []
    for disfluency in disfluencies:
        all_hits += re.finditer(r" *\b"+disfluency+r"\b *", text)
    all_hits = sorted(all_hits, key=lambda x: x.start())
    to_be_inserted = [(hit.start(), hit.group()) for hit in all_hits]
    new_text = text
    for hit in all_hits[::-1]:
        new_text = new_text[:hit.start()] + " " + new_text[hit.end():]
    return to_be_inserted, new_text

punctuation_regex = r"["+re.escape("".join(mapped_punctuation.keys()))+r"]"

def reconstitute_text(text, to_be_inserted):
    if len(to_be_inserted) == 0:
        return text
    pos_punc = [s.start() for s in re.finditer(punctuation_regex, text)]
    for start, token in to_be_inserted:
        start += len([p for p in pos_punc if p < start])
        text = text[:start] + token.rstrip(" ") + text[start:]
        print(text)
    return text


DISFLUENCIES = {
    "fr": [
        "euh",
        "heu",
    ]
}