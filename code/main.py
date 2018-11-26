# encoding=utf8
import os
import codecs
import pickle
import datetime
import itertools
from collections import OrderedDict
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from prepro import change_format
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

flags = tf.app.flags
flags.DEFINE_boolean("clean", False, "clean train folder")
flags.DEFINE_string("mode", "train", "train/test/evaluate")
# model参数配置
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_integer("lstm_dim", 256, "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")

# training参数配置
flags.DEFINE_float("clip", 5, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 128, "batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", True, "Wither lower case")
flags.DEFINE_float("dev_percentage", .1, "percentage of dev data")

flags.DEFINE_integer("max_epoch", 1000, "maximum training epochs")
flags.DEFINE_integer("steps_check", 20, "steps per checkpoint")
flags.DEFINE_string("ckpt_path", "../data/ckpt_best", "Path to save model")
flags.DEFINE_string("summary_path", "summary", "Path to store summaries")
flags.DEFINE_string("log_file", "../data/log/train.log", "File for log")
flags.DEFINE_string("map_file", "maps.pkl", "file for maps")
flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")
flags.DEFINE_string("config_file", "config_file", "File for config")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", "../data/result", "Path for results")
flags.DEFINE_string("filepath", "../data/ruijin_round1_train2_20181022", "train file path")
flags.DEFINE_string("test_filepath", "../data/ruijin_round1_test_a_20181022", "test file path")
flags.DEFINE_string("test_b_filepath", "../data/ruijin_round1_test_b_20181112", "test file path")
flags.DEFINE_string("ner_result", "../submit", "Path for results")

flags.DEFINE_string("model_type", "bilstm", "Model type selection")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# 参数导入
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1, f1


def train_new():
    train_sent = load_sentences(FLAGS.filepath)

    update_tag_scheme(train_sent, FLAGS.tag_schema)

    if not os.path.isfile(FLAGS.map_file):
        _c, char_to_id, id_to_char = char_mapping(train_sent, FLAGS.lower)
        print("random embedding")

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sent)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # 数据准备，划分验证集和训练集
    np.random.seed(10)
    train_sent_ = np.array(train_sent)
    shuffle_indices = np.random.permutation(np.arange(len(train_sent)))

    sent_shuffled = train_sent_[shuffle_indices]
    dev_sample_index = -1 * int(FLAGS.dev_percentage * float(len(train_sent)))
    train_sent_new, dev_sent = sent_shuffled[:dev_sample_index], sent_shuffled[dev_sample_index:]

    train_data = prepare_dataset(
        train_sent_new, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sent, char_to_id, tag_to_id, FLAGS.lower
    )

    print("%i / %i sentences in train." % (len(train_data), len(dev_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)

    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = FLAGS.log_file
    logger = get_logger(log_path)
    print_config(config, logger)

    # 根据需求，设置动态使用GPU资源
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        plt.grid(True)
        plt.ion()

        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(FLAGS.max_epoch):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)

                if step % 20 == 0:
                    ax.scatter(step, np.mean(loss), c='b', marker='.')
                    plt.pause(0.001)

                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(iteration, step % steps_per_epoch, steps_per_epoch,
                                                           np.mean(loss)))
                    loss = []
            best, f1 = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            ax2.scatter(i + 1, f1, c='b', marker='.')
            plt.pause(0.001)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger, "best")


def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        f = codecs.open(os.path.join(FLAGS.test_filepath, "127_9.txt"), "r", "utf-8")
        s = f.read()
        line = []
        sent = ''
        for i in range(len(s)):
            if s[i] != '。':
                sent += s[i]
            else:
                sent += s[i]
                line.append(sent)
                sent = ''

        line = input("请输入测试句子:")
        for info in line:
            print(info)
            result = model.evaluate_line(sess, input_from_line(info, char_to_id), id_to_tag)
            for info1 in result['entities']:
                print(info1)


def test():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)

        files = os.listdir(FLAGS.test_b_filepath)

        temp_dir = FLAGS.ner_result + '/submit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print(temp_dir)
        os.makedirs(temp_dir)

        for f1 in files:
            f_name = f1.split(".")[0]
            with codecs.open(os.path.join(FLAGS.test_b_filepath, f1), "r", "utf-8") as f2:
                s = f2.read()
                line = []
                sent = ''
                leng = 0  # 当前处理句子的位置长度
                for i in range(len(s)):
                    if s[i] != '。':
                        sent += s[i]
                    else:
                        sent += s[i]
                        line.append(sent)
                        sent = ''

                f3 = codecs.open(os.path.join(temp_dir, f_name + ".ann"), "w", "utf-8")
                print(f3.name)
                i = 0
                for info in line:
                    result = model.evaluate_line(sess, input_from_line(info, char_to_id), id_to_tag)
                    tag = result['entities']
                    # print(tag[0])

                    for char in tag:
                        sent = "T" + str(i + 1) + "\t" + char['type'] + " "
                        if char['word'].find("\n") == 0 or char['word'].find(" ") == 0:
                            char_start = char['word'][0]
                            start = char['start'] + 1 + leng
                            word_new = char['word'].replace(char_start, "")
                            if char['word'].endswith("\n") or char['word'].endswith(" "):
                                char_end = char['word'][-1]
                                end = char['end'] - 1 + leng
                                sent = sent + str(start) + " " + str(end) + "\t" + word_new.replace(char_end, "")
                            elif 0 < char['word'].find("\n") < len(char['word']):
                                j = char['word'].find("\n")
                                sent = sent + str(start) + " " + str(char['start'] + leng + j) + ";" + str(
                                    char['start'] + leng + j + 1) + " " + str(
                                    char['end'] + leng) + "\t" + word_new.replace("\n", " ")
                            else:
                                sent = sent + str(start) + " " + str(char['end'] + leng) + "\t" + word_new
                        else:
                            start = char['start'] + leng
                            if char['word'].endswith("\n") or char['word'].endswith(" "):
                                char_end = char['word'][-1]
                                end = char['end'] - 1 + leng
                                sent = sent + str(start) + " " + str(end) + "\t" + char['word'].replace(char_end, "")
                            elif 0 < char['word'].find("\n") < len(char['word']):
                                j = char['word'].find("\n")
                                sent = sent + str(start) + " " + str(char['start'] + leng + j) + ";" + str(
                                    char['start'] + leng + j + 1) + " " + str(char['end'] + leng) + "\t" + char[
                                           'word'].replace("\n", " ")
                            else:
                                sent = sent + str(start) + " " + str(char['end'] + leng) + "\t" + char['word']

                        f3.write(sent + '\n')
                        i += 1
                    leng += len(info)
                f3.close()


def main(_):
    if FLAGS.mode == "train":
        if FLAGS.clean:
            clean(FLAGS)
        train_new()
    elif FLAGS.mode == "test":
        test()
    elif FLAGS.mode == "evaluate":
        evaluate_line()
    else:
        print("Unknown mode!")
        exit(0)


if __name__ == "__main__":
    tf.app.run(main)
