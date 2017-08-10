from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

def prepare_input():
    filenames = ['data/got1.txt', 'data/got2.txt', 'data/got3.txt', 'data/got4.txt', 'data/got5.txt']
    with open('data/input.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def main():

    prepare_input()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--input_encoding', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--rnn_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--seq_length', type=int, default=25)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--grad_clip', type=float, default=5.)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--decay_rate', type=float, default=0.97)
    parser.add_argument('--gpu_mem', type=float, default=0.666)
    args = parser.parse_args()
    train(args)

def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.input_encoding)
    args.vocab_size = data_loader.vocab_size

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    model = Model(args)

    merged = tf.summary.merge_all()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        for e in range(model.epoch_pointer.eval(), args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            speed = 0
            assign_op = model.epoch_pointer.assign(e)
            sess.run(assign_op)

            for b in range(data_loader.pointer, data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state,
                        model.batch_time: speed}
                summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                speed = time.time() - start
                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, speed))
                if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
