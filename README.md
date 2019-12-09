# MedGraph: Structural and Temporal Representation Learning of Electronic Medical Records

## Running MedGraph

MedGraph can be trained for either a predictive healthcare task to output future medical risks or an unsupervised model to obtain visit and code embeddings.

### System requirements

* [Python 3.6](https://www.python.org)
* [Tensorflow 1.14](https://www.tensorflow.org)
* [NetworkX 2.3](https://networkx.github.io)

### Input data format for MedGraph

MedGraph expects a numpy compressed file (`.npz`) with the following elements in `data` directory:

* Mapping dictionaries for node identifier and index in the VC and VV graphs of training, validation and test sets: `ent2vtx_train`, `ent2vtx_valid` & `ent2vtx_test`
* Adjacency matrix of VC structural relations in the training set: `A_vc`
* Visit attribute vector matrices of training, validation and test sets: `X_visits_train`, `X_visits_val` & `X_visits_test`
* Code attribute vector matrix: `X_codes`
* VV sequences of training, validation and test sets: `vv_train`, `vv_valid` & `vv_test`, in which we have temporal sequences of visits where each visit event is expressed by "\[visit index, input time, output time, auxiliary task label\]". For example, for each patient we can represent the visit sequence as a list of tuples: `[[v1, t1, t2, y1], [v2, t2, t3, y2], [v3, t3, t4, y3], ...]`. 

Have a look at the `utils.py` file for more details.

### Running MedGraph script

```
python train.py dataset --embedding_dim=128 --vc_batch_size=128 --vv_batch_size=32 --K=10 --num_epochs=10 --learning_rate=0.001 --is_gauss=True --distance=w2 --is_time_dis=True
```
* `dataset`: name of the EMR dataset
* `embedding_dim`: visit and code embedding dimension
* `vc_batch_size`: batch size of VC bipartite edges
* `vv_batch_size`: batch size of VV event sequences
* `K`: number of negative VC edges for negative sampling
* `num_epochs`: number of training epochs
* `learning_rate`: learning rate of the Adam optimizer
* `is_gauss`: if `True` MedGraph learns Gaussian embeddings for visits and codes, or if `False` MedGraph produces point vector embeddings
* `distance`: if we represent visits and codes as Gaussians, we can define either `w2` (2-nd Wasserstein distance) or `kl` (symmetric KL divergence) as the distance measure
* `is_time_dis`: if `True` MedGraph makes predictions at each time step of the visit sequence, or if `False` MedGraph makes predictions at the last time step of the visit sequence

### MedGraph embeddings

Visit and code embeddings for test EMR are saved in `emb` directory as a dictionary in a numpy file (`.npy`). We can use the `ent2vtx_test` mapping dictionary to find the corresponding embedding representations of the visits and codes.

* If we learn Gaussian embeddings, `mu` gives the mean vector dictionary and `sigma` gives the diagonal co-variance vector dictionary
* If we learn point vector embeddings, the embedding dictionary gives the corresponding vector embeddings of the visits and codes

## 2-D visualisation of the code embeddings learned from MedGraph

First, Medgraph produces 128-dimensional code embeddings for ICD-10-CM codes.
Then, we use t-SNE to project these embeddings into 2 dimensions, for visualisation.
Colour of a code indicates its associated CCS class.
When you hover on the plot, you can see the ICD code, its definition and the relevant CCS class.
