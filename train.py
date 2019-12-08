import numpy as np
import tensorflow as tf
import argparse

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

from model import MedGraph
from utils import DataLoader, sparse_feeder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='mimic')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--vc_batch_size', type=int, default=128)
    parser.add_argument('--vv_batch_size', type=int, default=32)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--is_gauss', default=True)
    parser.add_argument('--distance', default='kl', help='kl or w2')
    parser.add_argument('--is_time_dis', default=True)
    args = parser.parse_args()
    args.is_gauss = True if args.is_gauss == 'True' else False
    args.is_time_dis = True if args.is_time_dis == 'True' else False
    train(args)


def train(args):
    np.random.RandomState(46)

    # Load the data into MedGraph data structure
    graph_file = 'data/%s.npz' % args.dataset
    data_loader = DataLoader(graph_file)

    display_freq = 10  # Frequency of displaying the training results

    # Set user-defined settings in the data loader
    data_loader.embedding_dim = args.embedding_dim
    data_loader.vc_batch_size = args.vc_batch_size
    data_loader.K = args.K
    data_loader.learning_rate = args.learning_rate
    data_loader.is_gauss = args.is_gauss
    data_loader.distance = args.distance
    data_loader.is_time_dis = args.is_time_dis

    model = MedGraph(data_loader)

    # Number of training iterations in each epoch
    global_step = 0
    num_iter = len(data_loader.vv_train) // args.vv_batch_size
    print('Number of iterations per epoch: {}'.format(num_iter))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(args.num_epochs):
            print('Training epoch: {}'.format(epoch + 1))

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
                sess.run(model.train_op, feed_dict=feed_dict_batch)

                if iteration % display_freq == 0:
                    # Calculate and display the batch loss and accuracy
                    loss_batch = sess.run(model.loss, feed_dict=feed_dict_batch)
                    print("Iter {0:3d}:\t Loss={1:.4f}".format(iteration, loss_batch))

            # Run validation and test after every epoch

            # Predict for validation set
            feed_dict_valid = {model.X_visits: sparse_feeder(data_loader.X_visits_val),
                               model.vv_inputs: data_loader.vv_valid_seq[0],
                               model.vv_in_time: data_loader.vv_valid_seq[2],
                               model.vv_out_time: data_loader.vv_valid_seq[3],
                               model.vv_out_mask: data_loader.vv_valid_seq[4]}
            y_pred_valid = sess.run(model.y, feed_dict=feed_dict_valid)
            # Calculate validation set evaluation metrics
            val_auc = roc_auc_score(y_true=data_loader.vv_valid_seq[1], y_score=y_pred_valid)
            val_ap = average_precision_score(y_true=data_loader.vv_valid_seq[1], y_score=y_pred_valid)

            # Predict for test set
            feed_dict_test = {model.X_visits: sparse_feeder(data_loader.X_visits_test),
                              model.vv_inputs: data_loader.vv_test_seq[0],
                              model.vv_in_time: data_loader.vv_test_seq[2],
                              model.vv_out_time: data_loader.vv_test_seq[3],
                              model.vv_out_mask: data_loader.vv_test_seq[4]}
            y_pred_test = sess.run(model.y, feed_dict=feed_dict_test)
            # Calculate test set evaluation metrics
            test_auc = roc_auc_score(y_true=data_loader.vv_test_seq[1], y_score=y_pred_test)
            test_ap = average_precision_score(y_true=data_loader.vv_test_seq[1], y_score=y_pred_test)
            test_precision = precision_score(data_loader.vv_test_seq[1][:, 1], np.argmax(y_pred_test, 1))
            test_recall = recall_score(data_loader.vv_test_seq[1][:, 1], np.argmax(y_pred_test, 1))

            # Save visit and code embeddings for test data (we use the same mapping dictionary)
            if args.is_gauss:
                mu, sigma = sess.run([model.embedding, model.sigma], feed_dict=feed_dict_test)
                np.save('emb/%s_embedding.npy' % args.dataset,
                        {'mu': data_loader.embedding_mapping(mu),
                         'sigma': data_loader.embedding_mapping(sigma)})
            else:
                mu = sess.run(model.embedding, feed_dict=feed_dict_test)
                np.save('emb/%s_embedding.npy' % args.dataset, data_loader.embedding_mapping(mu))

            print('---------------------------------------------------------')
            print(
                "Epoch: {0}, validation auc: {1:.01%}, validation ap: {2:.01%}, test auc: {3:.4f}, test ap: {4:.4f}, test precision: {5:.4f}, test recall: {6:.4f}".format(
                    epoch + 1, val_auc, val_ap, test_auc, test_ap, test_precision, test_recall))
            print('---------------------------------------------------------')


if __name__ == '__main__':
    main()
