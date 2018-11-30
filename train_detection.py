import torch
import torch.nn as nn
import time
import os
import numpy as np
from torchtext import data
import random
from argparse import ArgumentParser
from evaluation import evaluation
from entity_detection import EntityDetection

np.set_printoptions(threshold=np.nan)
parser = ArgumentParser(description="Joint Prediction")
parser.add_argument('--entity_detection_mode', type=str, required=True, help='options are LSTM, GRU')
parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
parser.add_argument('--gpu', type=int, default=0)  # Use -1 for CPU
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=.0003)
parser.add_argument('--seed', type=int, default=3435)
parser.add_argument('--dev_every', type=int, default=10000)
parser.add_argument('--log_every', type=int, default=2000)
parser.add_argument('--patience', type=int, default=12)
parser.add_argument('--dete_prefix', type=str, default='dete')
parser.add_argument('--words_dim', type=int, default=300)
parser.add_argument('--num_layer', type=int, default=2)
parser.add_argument('--rnn_fc_dropout', type=float, default=0.3)
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--rnn_dropout', type=float, default=0.3)
parser.add_argument('--clip_gradient', type=float, default=0.6, help='gradient clipping')
parser.add_argument('--vector_cache', type=str, default="data/sq_glove300d.pt")
parser.add_argument('--weight_decay',type=float, default=0)
parser.add_argument('--fix_embed', action='store_false', dest='train_embed')
# added for testing
parser.add_argument('--output', type=str, default='preprocess/')
args = parser.parse_args()

outfile = open(os.path.join(args.output, 'dete_train.txt'), 'w')
for line in open(os.path.join(args.output, 'train.txt'), 'r'):
    items = line.strip().split("\t")
    tokens = items[6].split()
    if any(token != tokens[0] for token in tokens):
        outfile.write("{}\t{}\n".format(items[5], items[6]))
outfile.close()

# Set random seed for reproducibility
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

# Set up the data for training
TEXT = data.Field(lower=True)
ED = data.Field()
train = data.TabularDataset(path=os.path.join(args.output, 'dete_train.txt'), format='tsv', fields=[('text', TEXT), ('ed', ED)])
field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', ED)]
dev, test = data.TabularDataset.splits(path=args.output, validation='valid.txt', test='test.txt', format='tsv', fields=field)
TEXT.build_vocab(train, dev, test)
ED.build_vocab(train, dev)

match_embedding = 0
if os.path.isfile(args.vector_cache):
    stoi, vectors, dim = torch.load(args.vector_cache)
    TEXT.vocab.vectors = torch.Tensor(len(TEXT.vocab), dim)
    for i, token in enumerate(TEXT.vocab.itos):
        wv_index = stoi.get(token, None)
        if wv_index is not None:
            TEXT.vocab.vectors[i] = vectors[wv_index]
            match_embedding += 1
        else:
            TEXT.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
else:
    print("Error: Need word embedding pt file")
    exit(1)

print("Embedding match number {} out of {}".format(match_embedding, len(TEXT.vocab)))

train_iter = data.Iterator(train, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=True,
                           repeat=False, sort=False, shuffle=True, sort_within_batch=False)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=False,
                         repeat=False, sort=False, shuffle=False, sort_within_batch=False)

config = args
config.words_num = len(TEXT.vocab)
config.label = len(ED.vocab)
model = EntityDetection(config)
model.embed.weight.data.copy_(TEXT.vocab.vectors)
if args.cuda:
    modle = model.to(torch.device("cuda:{}".format(args.gpu)))
    print("Shift model to GPU")

print(config)
print("VOCAB num",len(TEXT.vocab))
print("Train instance", len(train))
print("Dev instance", len(dev))
print("Entity Type", len(ED.vocab))
print(model)

parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.NLLLoss()

early_stop = False
best_dev_R = 0
iterations = 0
iters_not_improved = 0
num_dev_in_epoch = (len(train) // args.batch_size // args.dev_every) + 1
patience = args.patience * num_dev_in_epoch # for early stopping
epoch = 0
start = time.time()
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:10.6f}%'.split(','))
print('  Time Epoch Iteration Progress    (%Epoch)   Loss         Accuracy')

index2tag = np.array(ED.vocab.itos)  # ['<unk>' '<pad>' 'O' 'I']

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Recall: {}".format(epoch, best_dev_R))
        break
    epoch += 1
    train_iter.init_epoch()
    n_correct, n_total = 0, 0

    for batch_idx, batch in enumerate(train_iter):
        # Batch size : (Sentence Length, Batch_size)
        iterations += 1
        model.train()
        optimizer.zero_grad()
        scores = model(batch)
        # Entity Detection
        n_correct += torch.sum(
            torch.sum((torch.max(scores, 1)[1].view(batch.ed.size()).data == batch.ed.data), dim=0) == batch.ed.size()[
                0]).item()
        loss = criterion(scores, batch.ed.view(-1, 1)[:, 0])
        n_total += batch.batch_size
        loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:
            model.eval()
            dev_iter.init_epoch()
            gold_list = []
            pred_list = []

            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)
                #n_dev_correct += (
                #            (torch.max(answer, 1)[1].view(dev_batch.ed.size()).data == dev_batch.ed.data).sum(dim=0) ==
                #            dev_batch.ed.size()[0]).sum()
                index_tag = np.transpose(torch.max(answer, 1)[1].view(dev_batch.ed.size()).cpu().data.numpy())
                gold_list.append(np.transpose(dev_batch.ed.cpu().data.numpy()))
                pred_list.append(index_tag)

            P, R, F = evaluation(gold_list, pred_list, index2tag, type=False)
            print("{} Recall: {:10.6f}% Precision: {:10.6f}% F1 Score: {:10.6f}%".format("Dev", 100. * R, 100. * P,
                                                                                         100. * F))

            # update model
            if R > best_dev_R:
                best_dev_R = R
                iters_not_improved = 0
                snapshot_path = os.path.join(args.output, args.dete_prefix + '_best_model.pt')
                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)  # .state_dict()
            else:
                iters_not_improved += 1
                if iters_not_improved > patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 1:
            print(log_template.format(time.time() - start, epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss.item(), ' ' * 3,
                                      100. * n_correct / n_total))

# Early Stopping. Epoch: 119, Best Dev Recall: 0.9513194245975041