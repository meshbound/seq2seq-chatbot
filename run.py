#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
import pickle
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.models.seq2seq_with_attention import Seq2seqLuongAttention
import os
import discord


def load_data(PATH=''):
    # read data control dictionaries
    try:
        with open(PATH + 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = None
    # read numpy arrays
    # with open(PATH + 'metadata.pkl', 'rb') as f:
    # metadata = pickle.load(f)

    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


def split_dataset(x, y, ratio=[0.7, 0.15, 0.15]):
    # number of examples
    data_len = len(x)
    lens = [int(data_len * item) for item in ratio]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0] + lens[1]], y[lens[0]:lens[0] + lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX, trainY), (testX, testY), (validX, validY)


def initial_setup():
    metadata, idx_q, idx_a = load_data(PATH='PATH_TO_DATA_FOLDER')
    (trainX, trainY), (testX, testY), (validX, validY) = split_dataset(idx_q, idx_a)
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    validX = tl.prepro.remove_pad_sequences(validX.tolist())
    validY = tl.prepro.remove_pad_sequences(validY.tolist())
    return metadata, trainX, trainY, testX, testY, validX, validY


if __name__ == "__main__":

    # data preprocessing
    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup()

    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    batch_size = 32
    n_step = src_len // batch_size
    print(trainX)
    src_vocab_size = len(metadata['idx2w'])  # 8002 (0~8001)
    emb_dim = 1024

    word2idx = metadata['w2idx']  # dict  word 2 index
    idx2word = metadata['idx2w']  # list index 2 word

    unk_id = word2idx['unk']  # 1
    pad_id = word2idx['_']  # 0

    start_id = src_vocab_size  # 8002
    end_id = src_vocab_size + 1  # 8003

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']

    src_vocab_size = tgt_vocab_size = src_vocab_size + 2

    vocabulary_size = src_vocab_size


    def inference(seed, top_n):
        model_.eval()
        seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")]
        sentence_id = model_(inputs=[[seed_id]], seq_length=20, start_token=start_id, top_n=top_n)
        sentence = []
        for w_id in sentence_id[0]:
            w = idx2word[w_id]
            if w == 'end_id':
                break
            sentence = sentence + [w]
        return sentence


    decoder_seq_length = 20

    model_ = Seq2seq(
        decoder_seq_length=decoder_seq_length,
        cell_enc=tf.keras.layers.GRUCell,
        cell_dec=tf.keras.layers.GRUCell,
        n_layer=3,
        n_units=256,
        embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
    )

    # Uncomment below statements if you have already saved the model

    load_weights = tl.files.load_npz(name='model.npz')
    tl.files.assign_weights(load_weights, model_)

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    # model_.train()

    # seeds = ["hello", "test"]

    client = discord.Client()


    @client.event
    async def on_ready():
        print('We have logged in as {0.user}'.format(client))


    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        if message.content.startswith('$'):
            new_message = generate_response(message.content[1:len(message.content)])
            # print(message.content + " : " + new_message)

            await message.channel.send(new_message)


    def generate_response(seed):
        print("Query >", seed)
        sentence = inference(seed,1)
        print(" >", ' '.join(sentence))
        return ' '.join(sentence)


    client.run("Discord_bot_tolken")
