import torch
import torch.nn as nn
import time
import os
import numpy as np
import random

from torchtext import data
from argparse import ArgumentParser
from embedding import EmbedVector
from evaluation import get_names_for_entities
from sklearn.metrics.pairwise import euclidean_distances

parser = ArgumentParser(description="Training")
parser.add_argument('--qa_mode', type=str, required=True, help='options are LSTM, CNN, GRU')
parser.add_argument('--embed_dim', type=int, default=250)
parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
parser.add_argument('--gpu', type=int, default=0)  # Use -1 for CPU
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--seed', type=int, default=3435)
parser.add_argument('--dev_every', type=int, default=10000)
parser.add_argument('--log_every', type=int, default=2000)
parser.add_argument('--output_channel', type=int, default=300)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--best_prefix', type=str, default='entity')
parser.add_argument('--num_layer', type=int, default=2)
parser.add_argument('--rnn_fc_dropout', type=float, default=0.3)
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--rnn_dropout', type=float, default=0.3)
parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
parser.add_argument('--vector_cache', type=str, default="data/sq_glove300d.pt")
parser.add_argument('--weight_decay',type=float, default=0)
parser.add_argument('--cnn_dropout', type=float, default=0.5, help='dropout before fully connected layer in CNN')
parser.add_argument('--fix_embed', action='store_false', dest='train_embed')
parser.add_argument('--output', type=str, default='preprocess')
args = parser.parse_args()

################## Prepare training and validation datasets ##################
# Dictionary and embedding for words
if os.path.isfile(args.vector_cache):
    stoi, vectors, words_dim = torch.load(args.vector_cache)
else:
    print("Error: Need word embedding pt file")
    exit(1)

mid_dic = {}  # Dictionary for MID
for line in open(os.path.join(args.output, 'entity2id.txt'), 'r'):
    items = line.strip().split("\t")
    mid_dic[items[0]] = int(items[1])
outfile = open(os.path.join(args.output, 'entity_train.txt'), 'w')
for line in open(os.path.join(args.output, 'train.txt'), 'r'):
    items = line.strip().split("\t")
    if items[1] in mid_dic:
        outfile.write("{}\t{}\n".format(items[5], mid_dic[items[1]]))
outfile.close()
        # context = []
        # for token in list(compress(items[5].split(), [obj == 'O' for obj in items[6].split()])):
        #    if token not in stop_words and stoi.get(token) is not None:
        #        context.append(token)
        # if context:

entities_emb = np.fromfile(os.path.join(args.output, 'entities_emb.bin'), dtype=np.float32).reshape((len(mid_dic), args.embed_dim))
mid_emb_list = []
mids_list = []
index_names = get_names_for_entities(os.path.join(args.output, 'names.trimmed.txt'))
outfile = open(os.path.join(args.output, 'entity_valid.txt'), 'w')
for line in open(os.path.join(args.output, 'valid.txt'), 'r'):
    items = line.strip().split("\t")
    if items[1] in mid_dic and items[2] in index_names:
        mids = [mid for mid in index_names.get(items[2]) if mid in mid_dic]
        if len(mids) > 1:
            mids_list.append(mids)
            outfile.write("{}\t{}\n".format(items[5], mid_dic[items[1]]))
            mid_emb = []
            for mid in mids:
                mid_emb.append(entities_emb[mid_dic[mid]])
            mid_emb_list.append(np.asarray(mid_emb))
        #if flag:
        #    outtrain.write("{}\t{}\n".format(items[5], mid_dic[items[1]]))
outfile.close()
del index_names
entities_emb = torch.from_numpy(entities_emb)

#with open(os.path.join(args.output, entity2id.txt'), 'r') as f:
#    for line in f:
#        items = line.strip().split("\t")
#        if len(items) == 3:
#            context = []
#            for token in items[2].split():
#                if token not in stop_words and stoi.get(token) is not None:
#                    context.append(token)
#            if context:
#                entity_train.write("{}\t{}\n".format(' '.join(context), items[0]))

################## Set random seed for reproducibility ##################
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")

################## Load the datasets ##################
TEXT = data.Field(lower=True)
ED = data.Field(sequential=False, use_vocab=False)
train, dev = data.TabularDataset.splits(path=args.output, train='entity_train.txt', validation='entity_valid.txt', format='tsv', fields=[('text', TEXT), ('mid', ED)])
field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', None)]
test = data.TabularDataset(path=os.path.join(args.output, 'test.txt'), format='tsv', fields=field)
TEXT.build_vocab(train, dev, test)  # training data includes validation data


match_embedding = 0
TEXT.vocab.vectors = torch.Tensor(len(TEXT.vocab), words_dim)
for i, token in enumerate(TEXT.vocab.itos):
    wv_index = stoi.get(token, None)
    if wv_index is not None:
        TEXT.vocab.vectors[i] = vectors[wv_index]
        match_embedding += 1
    else:
        TEXT.vocab.vectors[i] = torch.FloatTensor(words_dim).uniform_(-0.25, 0.25)
print("Word embedding match number {} out of {}".format(match_embedding, len(TEXT.vocab)))
del stoi, vectors


################## batch ##################
train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True, sort_within_batch=False)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False, sort_within_batch=False)
config = args
config.words_num = len(TEXT.vocab)
config.label = args.embed_dim
config.words_dim = words_dim
model = EmbedVector(config)
model.embed.weight.data.copy_(TEXT.vocab.vectors)

if args.cuda:
    modle = model.to(torch.device("cuda:{}".format(args.gpu)))
    print("Shift model to GPU")
    entities_emb = entities_emb.cuda()

print(config)
print("VOCAB num", len(TEXT.vocab))
print("Train instance", len(train))
print("Dev instance", len(dev))
print(model)

parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.MSELoss()

early_stop = False
best_model, iterations, iters_not_improved = 0, 0, 0
num_dev_in_epoch = (len(train) // args.batch_size // args.dev_every) + 1
patience = args.patience * num_dev_in_epoch  # for early stopping
epoch = 0
start = time.time()
print('  Time Epoch Iteration Progress    (%Epoch)   Loss')
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f}'.split(','))

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, best_model))
        break
    epoch += 1
    train_iter.init_epoch()
    for batch_idx, batch in enumerate(train_iter):
        # Batch size : (Sentence Length, Batch_size)
        iterations += 1
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(batch), entities_emb[batch.mid, :])
        loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()
        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:
            model.eval()
            dev_iter.init_epoch()
            baseidx, n_dev_correct = 0, 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                batch_size = dev_batch.text.size()[1]
                answer = model(dev_batch).cpu().data.numpy()
                label = dev_batch.mid.data
                for devi in range(batch_size):
                    if label[devi].item() == mid_dic[mids_list[baseidx + devi][
                        euclidean_distances(answer[devi].reshape(1, -1), mid_emb_list[baseidx + devi]).argmin(axis=1)[
                            0]]]:
                        n_dev_correct += 1
                baseidx = baseidx + batch_size
            curr_accu = n_dev_correct / len(mids_list)
            print('Current Accuracy: {}'.format(curr_accu))
            # update model
            if curr_accu > best_model:
                best_model = curr_accu
                iters_not_improved = 0
                # save model, delete previous 'best_snapshot' files
                torch.save(model, os.path.join(args.output, args.best_prefix + '_best_model.pt'))
            else:
                iters_not_improved += 1
                if iters_not_improved > patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 1:
            # print progress message
            print(log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.item(), ' ' * 8, ' ' * 12))