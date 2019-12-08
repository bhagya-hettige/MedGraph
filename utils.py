import networkx as nx
import numpy as np
import scipy.sparse as sp


class DataLoader:
    def __init__(self, graph_file):
        with np.load(graph_file, allow_pickle=True) as loader:
            loader = dict(loader)

            # vc graph
            self.A = sp.csr_matrix((loader['adj_data_vc'], loader['adj_indices_vc'],
                                    loader['adj_indptr_vc']), shape=loader['adj_shape_vc'])

            # vc node attributes
            self.X_visits_train = sp.csr_matrix((loader['visit_attr_data'], loader['visit_attr_indices'],
                                                 loader['visit_attr_indptr']), shape=loader['visit_attr_shape'])
            self.X_codes = sp.csr_matrix((loader['code_attr_data'], loader['code_attr_indices'],
                                          loader['code_attr_indptr']), shape=loader['code_attr_shape'])
            self.X_visits_val = sp.csr_matrix((loader['attr_data_val'], loader['attr_indices_val'],
                                               loader['attr_indptr_val']), shape=loader['attr_shape_val'])
            self.X_visits_test = sp.csr_matrix((loader['attr_data_test'], loader['attr_indices_test'],
                                                loader['attr_indptr_test']), shape=loader['attr_shape_test'])

            # vv sequences
            self.vv_train = loader['vv_train']
            self.vv_valid = loader['vv_valid']
            self.vv_test = loader['vv_test']

            # Get mapping dictionaries for nodes
            self.ent2vtx_train = loader['ent2vtx_train'].item()
            self.ent2vtx_valid = loader['ent2vtx_valid'].item()
            self.ent2vtx_test = loader['ent2vtx_test'].item()

            # Process vc graph
            self.__process_vc_graph()

            # Process vv graph
            self.__process_vv_graph()

    def __process_vc_graph(self):
        self.vc_graph = nx.from_scipy_sparse_matrix(self.A)

        self.edges = list(self.vc_graph.edges(data=False))
        self.nodes = list(self.vc_graph.nodes(data=True))

        self.edge_distribution = np.ones(self.vc_graph.number_of_edges())
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.vc_graph.degree(node) for node in self.vc_graph.nodes()], dtype=np.float32), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

    def __process_vv_graph(self):
        max_seq_train = max([len(x) for x in self.vv_train])
        max_seq_valid = max([len(x) for x in self.vv_valid])
        max_seq_test = max([len(x) for x in self.vv_test])
        self.n_classes = np.max(self.vv_train[:, 3]) + 1

        vv_inputs = np.empty((len(self.vv_train), max_seq_train))
        vv_outputs = []
        vv_in_time = np.empty((len(self.vv_train), max_seq_train))
        vv_out_time = np.empty((len(self.vv_train), max_seq_train))
        output_mask = np.empty((len(self.vv_train), max_seq_train))
        for i, x in enumerate(self.vv_train):
            x = np.array([np.array(y) for y in x])
            vv_inputs[i] = np.pad(x[:, 0], (0, max_seq_train - len(x)), 'constant', constant_values=0)
            vv_outputs.append(np.eye(self.n_classes)[x[:, 3].astype(np.int)])
            vv_in_time[i] = np.pad(x[:, 1], (0, max_seq_train - len(x)), 'constant', constant_values=-1)
            vv_out_time[i] = np.pad(x[:, 2], (0, max_seq_train - len(x)), 'constant', constant_values=0)
            output_mask[i] = np.pad([idx + 1 for idx in range(len(x))], (0, max_seq_train - len(x)), 'constant',
                                    constant_values=0)
        self.vv_train_seq = [vv_inputs, vv_in_time, vv_out_time, output_mask.reshape([-1, max_seq_train, 1]),
                             np.asarray(vv_outputs)]

        vv_inputs_valid = np.empty((len(self.vv_valid), max_seq_valid))
        vv_outputs_valid = []
        vv_in_time_valid = np.empty((len(self.vv_valid), max_seq_valid))
        vv_out_time_valid = np.empty((len(self.vv_valid), max_seq_valid))
        output_mask_valid = np.empty((len(self.vv_valid), max_seq_valid))
        for i, x in enumerate(self.vv_valid):
            x = np.array([np.array(y) for y in x])
            vv_inputs_valid[i] = np.pad(x[:, 0], (0, max_seq_valid - len(x)), 'constant', constant_values=0)
            vv_outputs_valid.extend(np.eye(self.n_classes)[x[:, 3].astype(np.int)])
            vv_in_time_valid[i] = np.pad(x[:, 1], (0, max_seq_valid - len(x)), 'constant', constant_values=-1)
            vv_out_time_valid[i] = np.pad(x[:, 2], (0, max_seq_valid - len(x)), 'constant', constant_values=0)
            output_mask_valid[i] = np.pad([idx + 1 for idx in range(len(x))], (0, max_seq_valid - len(x)), 'constant',
                                          constant_values=0)
        self.vv_valid_seq = [vv_inputs_valid, vv_in_time_valid, vv_out_time_valid,
                             output_mask_valid.reshape([-1, max_seq_valid, 1]), np.asarray(vv_outputs_valid)]

        vv_inputs_test = np.empty((len(self.vv_test), max_seq_test))
        vv_outputs_test = []
        vv_in_time_test = np.empty((len(self.vv_test), max_seq_test))
        vv_out_time_test = np.empty((len(self.vv_test), max_seq_test))
        output_mask_test = np.empty((len(self.vv_test), max_seq_test))
        for i, x in enumerate(self.vv_test):
            x = np.array([np.array(y) for y in x])
            vv_inputs_test[i] = np.pad(x[:, 0], (0, max_seq_test - len(x)), 'constant', constant_values=0)
            vv_outputs_test.extend(np.eye(self.n_classes)[x[:, 3].astype(np.int)])
            vv_in_time_test[i] = np.pad(x[:, 1], (0, max_seq_test - len(x)), 'constant', constant_values=-1)
            vv_out_time_test[i] = np.pad(x[:, 2], (0, max_seq_test - len(x)), 'constant', constant_values=0)
            output_mask_test[i] = np.pad([idx + 1 for idx in range(len(x))], (0, max_seq_test - len(x)), 'constant',
                                         constant_values=0)
        self.vv_test_seq = [vv_inputs_test, vv_in_time_test, vv_out_time_test,
                            output_mask_test.reshape([-1, max_seq_test, 1]), np.asarray(vv_outputs_test)]

    def fetch_vc_batch(self, batch_size=16, K=5):
        edge_batch_index = self.edge_sampling.sampling(batch_size)
        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.vc_graph.__class__ == nx.Graph:
                if np.random.rand() > 0.5:
                    edge = (edge[1], edge[0])
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            for i in range(K):
                while True:
                    negative_node = self.node_sampling.sampling()
                    if not self.vc_graph.has_edge(negative_node, edge[1]):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
        return (u_i, u_j, label)

    def fetch_vv_batch(self, data, start, end):
        batch_idxes = range(start, end)
        batch_vv_inputs = data[0][batch_idxes, :]
        batch_time_train_in = data[1][batch_idxes, :]
        batch_time_train_out = data[2][batch_idxes, :]
        batch_out_mask = data[3][batch_idxes, :, :]
        batch_vv_outputs = []
        for idx in batch_idxes:
            batch_vv_outputs.extend(data[4][idx])
        batch_vv_outputs = np.asarray(batch_vv_outputs)
        return (batch_vv_inputs, batch_time_train_in, batch_time_train_out, batch_out_mask, batch_vv_outputs)

    def randomize_vv_sequences(self, data):
        permutation = np.random.permutation(len(data[0]))
        shuffled_data = [
            data[0][permutation, :],
            data[1][permutation, :],
            data[2][permutation, :],
            data[3][permutation, :, :],
            data[4][permutation]
        ]
        return shuffled_data

    def sequential_randomize_vv_sequences(self, data, batch_size=16):
        lengths = np.array([np.count_nonzero(x) for x in data[0]])
        sorted_idx = [x for _, x in sorted(zip(lengths, range(len(lengths))))]
        data_batches = [sorted_idx[i * batch_size: (i + 1) * batch_size]
                        for i in range((len(sorted_idx) + batch_size - 1) // batch_size)]
        data_batches = list(np.random.permutation(data_batches[:-1])) + [data_batches[-1]]
        permutation = [np.random.permutation(x) for x in data_batches]
        permutation = [i for x in permutation for i in x]
        shuffled_data = [
            data[0][permutation, :],
            data[1][permutation, :],
            data[2][permutation, :],
            data[3][permutation, :, :],
            data[4][permutation]
        ]
        return shuffled_data

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node, _ in self.nodes}


def sparse_feeder(M):
    M = sp.coo_matrix(M, dtype=np.float32)
    return np.vstack((M.row, M.col)).T, M.data, M.shape


# Reference from: LINE graph embedding (https://dl.acm.org/citation.cfm?id=2741093)
class AliasSampling:
    # Reference: https://en.wikipedia.org/wiki/Alias_method
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res
