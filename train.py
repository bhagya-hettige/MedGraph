import numpy as np
import tensorflow as tf
import argparse
import time

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

from model import MedGraph
from utils import DataLoader, sparse_feeder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='mimic')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--vc_batch_size', type=int, default=128)
    parser.add_argument('--vv_batch_size', type=int, default=128)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=10.0)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--time_dis', action='store_true')

    # For uncertainty modelling
    parser.add_argument('--gauss', action='store_true')
    parser.add_argument('--distance', default='kl', help='kl or w2')
    args = parser.parse_args()
    train(args)


def train(args):
    np.random.RandomState(46)

    # Load the data into MedGraph data structure
    graph_file = 'data/%s.npz' % args.dataset
    data_loader = DataLoader(graph_file)

    # Set user-defined settings in the data loader
    data_loader.embedding_dim = args.embedding_dim
    data_loader.vc_batch_size = args.vc_batch_size
    data_loader.K = args.K
    data_loader.alpha = args.alpha
    data_loader.beta = args.beta
    data_loader.gamma = args.gamma
    data_loader.learning_rate = args.learning_rate
    data_loader.is_gauss = args.gauss
    data_loader.distance = args.distance
    data_loader.is_time_dis = args.time_dis

    model = MedGraph(data_loader)

    # Number of training iterations in each epoch
    global_step = 0
    num_iter = len(data_loader.vv_train) // args.vv_batch_size
    print('Number of iterations per epoch: {}'.format(num_iter))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(args.num_epochs):
            start_time = time.time()
            tot_loss = 0
            # data = data_loader.sequential_randomize_vv_sequences(data_loader.vv_train_seq)
            data = data_loader.randomize_vv_sequences(data_loader.vv_train_seq)

            for iteration in range(num_iter):
                global_step += 1
                start = iteration * args.vv_batch_size
                end = (iteration + 1) * args.vv_batch_size if iteration < num_iter else data[0].shape[0]

                # Fetch vv sequences for the current batch
                (batch_vv_inputs, batch_time_train_in, batch_time_train_out, batch_out_mask,
                 batch_vv_outputs) = data_loader.fetch_vv_batch(data, start, end)

                # Fetch vc edges for the current batch
                (vc_u_i, vc_u_j, vc_label) = data_loader.fetch_vc_batch(batch_size=args.vc_batch_size, K=args.K)

                # Run optimization operation (backprop)
                feed_dict_batch = {
                    model.X_visits: sparse_feeder(data_loader.X_visits_train),
                    model.vv_inputs: batch_vv_inputs,
                    model.vv_outputs: batch_vv_outputs,
                    model.vv_in_time: batch_time_train_in,
                    model.vv_out_time: batch_time_train_out,
                    model.vv_out_mask: batch_out_mask,
                    model.vc_u_i: vc_u_i,
                    model.vc_u_j: vc_u_j,
                    model.vc_label: vc_label}
                loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict_batch)
                tot_loss += loss

            print("Epoch {:3d}:\t Training loss: {:.4f}\t Time taken: {:.4f}sec".format(epoch + 1,
                                                                                        tot_loss / num_iter,
                                                                                        time.time() - start_time))

            # Run validation and test after every epoch

            # Predict for validation set
            feed_dict_valid = {model.X_visits: sparse_feeder(data_loader.X_visits_val),
                               model.vv_inputs: data_loader.vv_valid_seq[0],
                               model.vv_in_time: data_loader.vv_valid_seq[1],
                               model.vv_out_time: data_loader.vv_valid_seq[2],
                               model.vv_out_mask: data_loader.vv_valid_seq[3]}
            y_pred_valid = sess.run(model.y, feed_dict=feed_dict_valid)
            # Calculate validation set evaluation metrics
            if data_loader.n_classes == 2:
                val_auc = roc_auc_score(y_true=data_loader.vv_valid_seq[4], y_score=y_pred_valid)
                val_ap = average_precision_score(y_true=data_loader.vv_valid_seq[4], y_score=y_pred_valid)
            else:
                val_auc = roc_auc_score(y_true=data_loader.vv_valid_seq[4], y_score=y_pred_valid, average='micro')
                # val_rec10 = get_recall_k(y_pred_valid, data_loader.vv_valid_seq[4], 10)
                # val_rec20 = get_recall_k(y_pred_valid, data_loader.vv_valid_seq[4], 20)
                val_rec30 = get_recall_k(y_pred_valid, data_loader.vv_valid_seq[4], 30)

            # Predict for test set
            feed_dict_test = {model.X_visits: sparse_feeder(data_loader.X_visits_test),
                              model.vv_inputs: data_loader.vv_test_seq[0],
                              model.vv_in_time: data_loader.vv_test_seq[1],
                              model.vv_out_time: data_loader.vv_test_seq[2],
                              model.vv_out_mask: data_loader.vv_test_seq[3]}
            y_pred_test = sess.run(model.y, feed_dict=feed_dict_test)
            # Calculate test set evaluation metrics
            if data_loader.n_classes == 2:
                test_auc = roc_auc_score(y_true=data_loader.vv_test_seq[4], y_score=y_pred_test)
                test_ap = average_precision_score(y_true=data_loader.vv_test_seq[4], y_score=y_pred_test)
                print(
                    "Validation AUC: {:.4f}\t Validation AP: {:.4f}\t Test AUC: {:.4f}\t Test AP: {:.4f}\t".format(
                        val_auc, val_ap, test_auc, test_ap))
            else:
                test_auc = roc_auc_score(y_true=data_loader.vv_test_seq[4], y_score=y_pred_test, average='micro')
                # test_rec10 = get_recall_k(y_pred_test, data_loader.vv_test_seq[4], 10)
                # test_rec20 = get_recall_k(y_pred_test, data_loader.vv_test_seq[4], 20)
                test_rec30 = get_recall_k(y_pred_test, data_loader.vv_test_seq[4], 30)
                print(
                    "Validation AUC: {:.4f}\t Validation Recall@30: {:.4f}\t Test AUC: {:.4f}\t Test Recall@30: {:.4f}\t".format(
                        val_auc, val_rec30, test_auc, test_rec30))

            # Save visit and code embeddings for test data (we use the same mapping dictionary)
            if args.gauss:
                mu, sigma = sess.run([model.embedding, model.sigma], feed_dict=feed_dict_test)
                np.save('emb/%s_embedding.npy' % args.dataset,
                        {'mu': data_loader.embedding_mapping(mu),
                         'sigma': data_loader.embedding_mapping(sigma)})
            else:
                mu = sess.run(model.embedding, feed_dict=feed_dict_test)
                np.save('emb/%s_embedding.npy' % args.dataset, data_loader.embedding_mapping(mu))

            print(
                '----------------------------------------------------------------------------------------------------------------------------------')


def get_recall_k(predicted, actual, k):
    total_recall = 0
    act_len = 0
    for i in range(len(actual)):
        act_set = set(np.where(actual[i] > 0)[0])
        if len(act_set) > 0:
            pred_set = set(np.argsort(predicted[i])[:-(k + 1):-1])
            total_recall += len(act_set & pred_set) / float(len(act_set)) if len(act_set) > 0 else 0
            act_len += 1
    total_recall /= act_len
    return total_recall


def get_auc(predicted, actual):
    batch_size = np.shape(predicted)[0]
    column = np.shape(predicted)[1]
    all = 0
    yes = 0
    avg = 0
    for ep, er in zip(predicted, actual):
        # print(er, column)
        temp_all = np.sum(er) * (column - np.sum(er))
        if temp_all > 0:
            all = all + temp_all
            temp_yes = 0
            for er_i in range(column):
                if er[er_i] == 1:
                    for er_j in range(column):
                        if er[er_j] == 0:
                            if ep[er_i] > ep[er_j]:
                                temp_yes = temp_yes + 1
            yes = yes + temp_yes
            avg = avg + float(temp_yes) / float(temp_all)
    return avg / batch_size


if __name__ == '__main__':
    main()
