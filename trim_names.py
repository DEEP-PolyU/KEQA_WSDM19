import argparse
import os
import random
from util import www2fb, processed_text, clean_uri

# output 'cleanedFB.txt', 'names.trimmed.txt', 'transE_*.txt', 'entity2id.txt', 'relation2id.txt'

def get_fb_mids_set(cleanfile, fbsubset):
    print('get all mids in the Freebase subset...')
    lines_seen = set()  # holds lines already seen
    outfile = open(cleanfile, "w")
    mids = []
    for i, line in enumerate(open(fbsubset, "r")):
        if i % 1000000 == 0:
            print("line: {}".format(i))
        items = line.strip().split("\t")
        if len(items) != 3:
            print("ERROR: line - {}".format(line))
        entity1 = www2fb(items[0])
        line = "{}\t{}\t{}\n".format(entity1, www2fb(items[2]), www2fb(items[1]))
        if line not in lines_seen:  # not a duplicate
            mids.append(entity1)  # mids.extend(entity2.split())
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
    return set(mids)

def findsetgrams(dataset):
    grams = []  # all possible grams for head entities
    ground = []  # Ground truth, for evluation only
    whhowset = [{'what', 'how', 'where', 'who', 'which', 'whom'},
                {'in which', 'what is', "what 's", 'what are', 'what was', 'what were', 'where is', 'where are',
                 'where was', 'where were', 'who is', 'who was', 'who are', 'how is', 'what did'},
                {'what kind of', 'what kinds of', 'what type of', 'what types of', 'what sort of'}]
    for fname in ["annotated_fb_data_valid", "annotated_fb_data_test"]:
        for i, line in enumerate(open(os.path.join(dataset, fname + ".txt"), 'r')):
            items = line.strip().split("\t")
            if len(items) != 4:
                print("ERROR: line - {}".format(line))
                break
            ground.append(www2fb(items[0]))
            question = processed_text(items[3]).split()
            if len(question) > 2:
                for j in range(3, 0, -1):
                    if ' '.join(question[0:j]) in whhowset[j - 1]:
                        del question[0:j]
                        continue
            maxlen = len(question)
            for token in question:
                grams.append(token)
            for j in range(2, maxlen + 1):
                for token in [question[idx:idx + j] for idx in range(maxlen - j + 1)]:
                    grams.append(' '.join(token))
    return set(grams), set(ground)

def get_all_entity_mids(fbpath, entiset):
    print('based on selected entities filter Freebase subset')
    mids = []
    #mids_dic = {}
    relat = []
    trainfile = open(os.path.join(args.output, 'transE_train.txt'), 'w')
    validfile = open(os.path.join(args.output, 'transE_valid.txt'), 'w')
    testfile = open(os.path.join(args.output, 'transE_test.txt'), 'w')
    with open(fbpath, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))
            items = line.strip().split("\t")
            entity1 = items[0]
            if entity1 in entiset:  # or entity2 in entiset:  # or predicate in predset:
                predicate = items[2]
                relat.append(predicate)
                mids.append(entity1)
                #if mids_dic.get(entity1) is None:
                #    mids_dic[entity1] = [(predicate)]
                #else:
                #    mids_dic[entity1].append(predicate)
                #for entity2 in items[1].split():
                entity2 = items[1].split()[0]  #  could be a list of entities
                mids.append(entity2)
                trainfile.write("{}\t{}\t{}\n".format(entity1, entity2, predicate))
                j = random.randrange(10)
                if not j:
                    validfile.write("{}\t{}\t{}\n".format(entity1, entity2, predicate))
                if j == 1:
                    testfile.write("{}\t{}\t{}\n".format(entity1, entity2, predicate))
    trainfile.close()
    validfile.close()
    testfile.close()
    with open(os.path.join(args.output, 'entity2id.txt'), 'w',encoding='UTF-8',errors='ignore') as outfile:
        for i, entity in enumerate(set(mids)):
            outfile.write("{}\t{}\n".format(entity, i))
            #if mids_dic.get(entity) is None:
            #    outfile.write("{}\t{}\n".format(entity, i))
            #else:
            #    tokens = []
            #    for context in mids_dic[entity]:
            #        tokens.append(context.replace('.', ' ').replace('_', ' '))
            #        seen = set()
            #    outfile.write("{}\t{}\t{}\n".format(entity, i, ' '.join([token for token in tokens if not (token in seen or seen.add(token))])))
    print('Number of entities in transE_*: {}'.format(i + 1))
    outfile.close()
    with open(os.path.join(args.output, 'relation2id.txt'), 'w',encoding='UTF-8',errors='ignore') as outfile:
        for i, predicate in enumerate(set(relat)):
            outfile.write("{}\t{}\n".format(predicate, i))
    print('Number of predicates in transE_*: {}'.format(i + 1))
    outfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the questions to match the Freebase')
    parser.add_argument('-n', '--names', dest='names', action='store', required=True,
                        help='path to the names file (from CFO)')
    parser.add_argument('-f', '--fbsubset', dest='fbsubset', action='store', required=True,
                        help='path to freebase subset file')
    parser.add_argument('-d', '--dataset', type=str, default='data', help='directory contains annotated_fb_data_*')
    parser.add_argument('-o', '--output', type=str, default='preprocess/', help='output directory for new dataset')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    cleanfile = os.path.join(args.output, 'cleanedFB.txt')
    fb_mids = get_fb_mids_set(cleanfile, args.fbsubset)
    gramset, groundset = findsetgrams(args.dataset)

    print('select head entities based on questions:')
    entiset = set()  # selected head entities
    with open(os.path.join(args.dataset, "annotated_fb_data_train.txt"), 'r',encoding='UTF-8',errors='ignore') as f:
        for i, line in enumerate(f):
            items = line.strip().split("\t")
            if len(items) != 4:
                print("ERROR: line - {}".format(line))
                break
            entiset.add(www2fb(items[0]))  # entiset.add(www2fb(items[2]))
    outfile = open(os.path.join(args.output, 'names.trimmed.txt'), 'w',encoding='UTF-8',errors='ignore')  # output file path for trimmed names file
    with open(args.names, 'r',encoding='UTF-8',errors='ignore') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("line: {}".format(i))
            items = line.strip().split("\t")
            if len(items) != 4:
                print("ERROR: line - {}".format(line))
            entity = www2fb(clean_uri(items[0]))
            if entity in fb_mids:
                name = processed_text(clean_uri(items[2]))
                if name.strip() != "":
                    if entity in entiset:
                        outfile.write("{}\t{}\n".format(entity, name))
                    elif name in gramset:
                        entiset.add(entity)
                        outfile.write("{}\t{}\n".format(entity, name))
                        #name_gram = [name]
                        #tokens = name.split()
                        #maxlen = len(tokens)
                        #if maxlen > 2:
                        #    j = maxlen - 1
                        #    for token in [tokens[idx:idx + j] for idx in range(maxlen - j + 1)]:
                        #        name_gram.append(' '.join(token))
                        #for token in name_gram:
    outfile.close()
    print('{} out of {} entities are selected for head'.format(len(entiset), i + 1))
    i = 0
    for entity in groundset:
        if entity in entiset:
            i += 1
    print('recall of head entity selection: {}'.format(float(i) / len(groundset)))
    get_all_entity_mids(cleanfile, entiset)
