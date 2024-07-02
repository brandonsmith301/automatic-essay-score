from sklearn.model_selection import StratifiedKFold
import re, codecs 
# from spellchecker import SpellChecker
from text_unidecode import unidecode
from typing import Tuple
from tqdm import tqdm
 

# SpaCy tokenizer
# import spacy
# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start:error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start:error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def get_additional_special_tokens():
    return {
        '\n': '[BR]',
        'Generic_School': '[GENERIC_SCHOOL]',
        'Generic_school': '[GENERIC_SCHOOL]',
        'SCHOOL_NAME': '[SCHOOL_NAME]',
        'STUDENT_NAME': '[STUDENT_NAME]',
        'Generic_Name': '[GENERIC_NAME]',
        'Genric_Name': '[GENERIC_NAME]',
        'Generic_City': '[GENERIC_CITY]',
        'LOCATION_NAME': '[LOCATION_NAME]',
        'HOTEL_NAME': '[HOTEL_NAME]',
        'LANGUAGE_NAME': '[LANGUAGE_NAME]',
        'PROPER_NAME': '[PROPER_NAME]',
        'OTHER_NAME': '[OTHER_NAME]',
        'PROEPR_NAME': '[PROPER_NAME]',
        'RESTAURANT_NAME': '[RESTAURANT_NAME]',
        'STORE_NAME': '[STORE_NAME]',
        'TEACHER_NAME': '[TEACHER_NAME]',
    }


def replace_special_tokens(text):
    special_tokens_replacement = get_additional_special_tokens()
    for key, value in special_tokens_replacement.items():
        text = text.replace(key, value)
    return text


def pad_punctuation(text):
    text = re.sub('([.,!?()-])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    return text


# def detect_and_replace_spelling_errors(text):
#     spell = SpellChecker()
    
#     doc = nlp(text)
#     words = [token.text for token in doc if token.is_alpha]  
    
#     misspelled = spell.unknown(words)
    
#     corrected_text = text
#     for word in misspelled:
#         corrected_text = corrected_text.replace(word, '[SPELL_ERROR]')

#     return corrected_text


def make_folds(df, target_cols, n_splits, random_state):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for n, (train_index, val_index) in enumerate(kfold.split(df, df[target_cols])):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    return df


def get_max_len_from_df(df, tokenizer, n_special_tokens=3):
    lengths = []
    tk0 = tqdm(df['full_text'].fillna("").values, total=len(df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    max_length = max(lengths) + n_special_tokens
    return max_length

def preprocess_text(text):
    text = resolve_encodings_and_normalize(text)
    text = replace_special_tokens(text)
    return text