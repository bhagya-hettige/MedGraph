# MedGraph: Structural and Temporal Representation Learning of Electronic Medical Records

Input data format for MedGraph
-

MedGraph expects a numpy compressed file (`.npz`) with the following elements:

- Mapping dictionaries for node identifier and index in the VC and VV graphs of training, validation and test sets: `ent2vtx_train`, `ent2vtx_valid` & `ent2vtx_test`
- Adjacency matrix of VC structural relations in the training set: `A_vc`
- Visit attribute vector matrices of training, validation and test sets: `X_visits_train`, `X_visits_val` & `X_visits_test`
- Code attribute vector matrix: `X_codes_train`
- VV sequences of training, validation and test sets: `vv_train`, `vv_valid` & `vv_test`, in which we have temporal sequences of visits where each visit event is expressed by (visit index, input time, output time, auxiliary task label). For example, for each patient we can represent the visit sequence as a list of tuples: `[(v1, t1, t2, y1), (v2, t2, t3, y2), (v3, t3, t4, y3), ...]`. Please have a look at the `utils.py` file for more details.

2-D visualisation of the code embeddings learned from MedGraph
-

First, Medgraph produces 128-dimensional code embeddings for ICD-10-CM codes.
Then, we use t-SNE to project these embeddings into 2 dimensions, for visualisation.
Colour of a code indicates its associated CCS class.
When you hover on the plot, you can see the ICD code, its definition and the relevant CCS class.
