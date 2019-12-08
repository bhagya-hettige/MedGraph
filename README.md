# MedGraph: Structural and Temporal Representation Learning of Electronic Medical Records

## Input data format for MedGraph

MedGraph expects a numpy compressed file (`.npz`) with the following elements:

- Mapping dictionaries for node identifier and index in the VC and VV graphs of training, validation and test sets: `ent2vtx_train`, `ent2vtx_valid` & `ent2vtx_test`
- Adjacency matrix of VC structural relations in the training set: `A_vc`
- Visit attribute vector matrices of training, validation and test sets: `X_visits_train`, `X_visits_val` & `X_visits_test`
- Code attribute vector matrix: `X_codes`
- VV sequences of training, validation and test sets: `vv_train`, `vv_valid` & `vv_test`, in which we have temporal sequences of visits where each visit event is expressed by "\[visit index, input time, output time, auxiliary task label\]". For example, for each patient we can represent the visit sequence as a list of tuples: `[[v1, t1, t2, y1], [v2, t2, t3, y2], [v3, t3, t4, y3], ...]`. 

Have a look at the `utils.py` file for more details.

## Running MedGraph

MedGraph can be trained for either a predictive healthcare task to output future medical risks or an unsupervised model to obtain visit and code embeddings.

### System requirements
* [Python 3.6](https://www.python.org)
* [Tensorflow 1.14](https://www.tensorflow.org)
* [NetworkX 2.3](https://networkx.github.io)

### Running MedGraph script
```
python train.py dataset --embedding_dim=128 --vc_batch_size=128 --vv_batch_size=32 --K=10 --num_epochs=10 --learning_rate=0.001 --is_gauss=True --distance=w2 --is_time_dis=True
```

## 2-D visualisation of the code embeddings learned from MedGraph

First, Medgraph produces 128-dimensional code embeddings for ICD-10-CM codes.
Then, we use t-SNE to project these embeddings into 2 dimensions, for visualisation.
Colour of a code indicates its associated CCS class.
When you hover on the plot, you can see the ICD code, its definition and the relevant CCS class.
