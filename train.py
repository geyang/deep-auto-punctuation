import visdom

vis = visdom.Visdom()
vis.env = "deep-punc-experiment"

emb = None
opt = None
smx = None
lss = None


def plot_progress(embeded, output, softmax, losses):
    global emb, opt, smx, lss
    emb = vis.heatmap(embeded, win=emb or None, opts=dict(title="Input Embedding"))
    opt = vis.heatmap(output, win=opt or None, opts=dict(title="GRU Output"))
    smx = vis.heatmap(softmax, win=smx or None, opts=dict(title="Softmax Activation"))
    lss = vis.line(Y=losses, win=lss or None, opts=dict(title="Loss"))


input_chars = list(" \nabcdefghijklmnopqrstuvwxyz01234567890")
output_chars = ["<nop>", "<cap>"] + list(".,;:?!\"'$")

import utils, data, metric, model
from tqdm import tqdm
import numpy as np
from IPython.display import HTML, clear_output

input_chars = list(" \nabcdefghijklmnopqrstuvwxyz01234567890")
output_chars = ["<nop>", "<cap>"] + list(".,;:?!\"'$")

# torch.set_num_threads(8)
batch_size = 64

char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True)
output_char2vec = utils.Char2Vec(chars=output_chars)
input_size = char2vec.size
output_size = output_char2vec.size

print("input_size is: " + str(input_size) + "; ouput_size is: " + str(output_size))
hidden_size = input_size
layers = 1

rnn = model.GruRNN(input_size, hidden_size, output_size, batch_size=batch_size, layers=layers, bi=True)
egdt = model.Engadget(rnn, char2vec, output_char2vec)
# egdt.load('./data/Gru_Engadget_1_layer_bi_batch_290232.tar')

learning_rate = 0.5e-2
egdt.setup_training(learning_rate)

seq_length = 500

for epoch_num in range(24):

    for batch_ind, (max_len, sources) in enumerate(tqdm(data.batch_gen(data.train_gen(), batch_size))):

        # prepare the input and output chunks
        input_srcs = []
        punc_targs = []
        for chunk in sources:
            input_source, punctuation_target = data.extract_punc(chunk, egdt.char2vec.chars, egdt.output_char2vec.chars)
            input_srcs.append(input_source)
            punc_targs.append(punctuation_target)

            # at the begining of the file, reset hidden to zero
        egdt.init_hidden_(random=False)
        seq_len = data.fuzzy_chunk_len(max_len, seq_length)
        for input_, target_ in zip(zip(*[data.chunk_gen(seq_len, src) for src in input_srcs]),
                                   zip(*[data.chunk_gen(seq_len, tar, ["<nop>"]) for tar in punc_targs])):

            try:
                egdt.forward(input_, target_)
                egdt.descent()
            except KeyError:
                raise KeyError

        if batch_ind % 25 == 24:
            print('Epoch {:d} Batch {}'.format(epoch_num + 1, batch_ind + 1))
            print("=================================")
            punctuation_output = egdt.output_chars()
            plot_progress(egdt.embeded[0, :400].data.numpy().T,
                          egdt.output[0, :400].data.numpy().T,
                          egdt.softmax[0, :400].data.numpy().T,
                          np.array(egdt.losses))

            metric.print_pc(utils.flatten(punctuation_output), utils.flatten(target_))
            print('\n')

        if batch_ind % 100 == 99:
            validate_target = data.apply_punc(input_[0], target_[0])
            result = data.apply_punc(input_[0],
                                     punctuation_output[0])
            print(validate_target)
            print(result)

    # print('Dev Set Performance {:d}'.format(epoch_num))
    egdt.save('./data/engadget_train_epoch-{}_batch-{}.tar'.format(epoch_num + 1, batch_ind + 1))
