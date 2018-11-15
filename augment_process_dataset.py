import os
import sys
import argparse
import re
#import logging

from fuzzywuzzy import process, fuzz
#from nltk.tokenize.treebank import TreebankWordTokenizer
from util import www2fb, processed_text
#tokenizer = TreebankWordTokenizer()
#logger = logging.getLogger()
#logger.disabled = True

def get_indices(src_list, pattern_list):
    indices = None
    for i in range(len(src_list)):
        match = 1
        for j in range(len(pattern_list)):
            if src_list[i+j] != pattern_list[j]:
                match = 0
                break
        if match:
            indices = range(i, i + len(pattern_list))
            break
    return indices

def get_ngram(tokens):
    ngram = []
    for i in range(1, len(tokens)+1):
        for s in range(len(tokens)-i+1):
            ngram.append((" ".join(tokens[s: s+i]), s, i+s))
    return ngram

def reverseLinking(sent, text_candidate):
    tokens = sent.split()
    label = ["O"] * len(tokens)
    text_attention_indices = None
    exact_match = False
    if text_candidate is None or len(text_candidate) == 0:
        return '<UNK>', ' '.join(label), exact_match
    # sorted by length
    for text in sorted(text_candidate, key=lambda x:len(x), reverse=True):
        pattern = r'(^|\s)(%s)($|\s)' % (re.escape(text))
        if re.search(pattern, sent):
            text_attention_indices = get_indices(tokens, text.split())
            break
    if text_attention_indices != None:
        exact_match = True
        for i in text_attention_indices:
            label[i] = 'I'
    else:
        try:
            v, score = process.extractOne(sent, text_candidate, scorer=fuzz.partial_ratio)
        except:
            print("Extraction Error with FuzzyWuzzy : {} || {}".format(sent, text_candidate))
            return '<UNK>', ' '.join(label), exact_match
        v = v.split()
        n_gram_candidate = get_ngram(tokens)
        n_gram_candidate = sorted(n_gram_candidate, key=lambda x: fuzz.ratio(x[0], v), reverse=True)
        top = n_gram_candidate[0]
        for i in range(top[1], top[2]):
            label[i] = 'I'
    entity_text = []
    for l, t in zip(label, tokens):
        if l == 'I':
            entity_text.append(t)
    entity_text = " ".join(entity_text)
    label = " ".join(label)
    return entity_text, label, exact_match

def augment_dataset(datadir, outdir):
    #  Get the name dictionary
    names_map = {}
    with open(os.path.join(outdir, 'names.trimmed.txt'), 'r') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("line: {}".format(i))

            items = line.strip().split("\t")
            if len(items) != 2:
                print("ERROR: line - {}".format(line))
                continue
            entity = items[0]
            literal = items[1].strip()
            if names_map.get(entity) is None:
                names_map[entity] = [(literal)]
            else:
                names_map[entity].append(literal)
    print("creating new datasets...")
    entiset = set()
    predset = set()
    wordset = []
    for f_tuple in [("annotated_fb_data_train", "train"), ("annotated_fb_data_valid", "valid"),
                    ("annotated_fb_data_test", "test")]:
        f = f_tuple[0]
        fname = f_tuple[1]
        fpath = os.path.join(datadir, f + ".txt")
        fpath_numbered = os.path.join(outdir, fname + ".txt")
        total_exact = 0
        outfile = open(fpath_numbered, 'w')
        print("reading from {}".format(fpath))

        with open(fpath, 'r') as f:
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                if len(items) != 4:
                    print("ERROR: line - {}".format(line))
                    sys.exit(0)
                lineid = i + 1
                subject = www2fb(items[0])
                predicate = www2fb(items[1])
                object = www2fb(items[2])
                question = processed_text(items[3])
                entiset.add(subject)
                entiset.add(object)
                predset.add(predicate)

                if names_map.get(subject) is None:
                    cand_entity_names = None
                else:
                    cand_entity_names = names_map[subject]

                entity_name, label, exact_match = reverseLinking(question, cand_entity_names)
                if exact_match:
                    total_exact += 1
                for token in question.split():
                    wordset.append(token)
                outfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(lineid, subject, entity_name, predicate, object, question, label))
        outfile.close()
        print("Exact Match Entity : {} out of {} : {}".format(total_exact, lineid, total_exact / lineid))
        print("wrote to {}".format(fpath_numbered))
    print('Total entities {}'.format(len(entiset)))
    print('Total predicates {}'.format(len(predset)))
    print('Total words {}'.format(len(set(wordset)) - 1))  # -1 for '<UNK>'
    # outfile = open(os.path.join(outdir, 'synthetic.txt'), 'w')
    # total_exact = 0
    # lineid = 0
    # whereset = {'location', 'place', 'geographic', 'region', 'places'}
    # whoset = {'composer', 'people', 'artist', 'author', 'publisher', 'directed', 'developer', 'director', 'lyricist',
    #           'edited', 'parents', 'instrumentalists', 'produced', 'manufacturer', 'written', 'designers', 'producer'}
    # for line in open(os.path.join(outdir, 'transE_valid.txt'), 'r'):
    #     items = line.strip().split("\t")
    #     subject = items[0]
    #     if names_map.get(subject) is not None:
    #         lineid += 1
    #         shortest = 10000
    #         for name in names_map[subject]:
    #             if len(name.split()) < shortest:
    #                 cand_entity_names = name
    #         tokens = items[2].replace('.', ' ').replace('_', ' ').split()
    #         seen = set()
    #         clean_token = [token for token in tokens if not (token in seen or seen.add(token))]
    #         flag = True
    #         for token in clean_token:
    #             if token in whereset:
    #                 question = 'where is the ' + ' '.join(clean_token) + ' of ' + cand_entity_names
    #                 flag = False
    #                 break
    #             elif token in whoset:
    #                 question = 'who is the ' + ' '.join(clean_token) + ' of ' + cand_entity_names
    #                 flag = False
    #                 break
    #         if flag:
    #             question = 'what is the ' + ' '.join(clean_token) + ' of ' + cand_entity_names
    #         cand_entity_names = [cand_entity_names]
    #         entity_name, label, exact_match = reverseLinking(question, cand_entity_names)
    #         if exact_match:
    #             total_exact += 1
    #             outfile.write(
    #                 '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(lineid, subject, entity_name, items[2], items[1], question,
    #                                                       label))
    # outfile.close()
    # print("Exact Match Entity : {} out of {} : {}".format(total_exact, lineid, total_exact / lineid))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment dataset with line ids, shorted names, entity names')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store', required=True,
                        help='path to the dataset directory - contains train, valid, test files')
    parser.add_argument('-o', '--output', type=str, default='preprocess', help='output directory for new dataset')

    args = parser.parse_args()
    print("Dataset: {}".format(args.dataset))
    print("Index - Names: /{}/names.trimmed.txt".format(args.output))
    print("Output: {}".format(args.output))

    augment_dataset(args.dataset, args.output)
