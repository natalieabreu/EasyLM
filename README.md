# Optimal Second Order Optimization

## EasyLM
* The codebase builds off of the [EasyLM](https://github.com/young-geng/EasyLM) repo
* The following changes were made from the original repo (at time of writing):
  * Initialization of llama model adjusted (divides by sqrt(hidden_size))
  * Eval script modified to reset evaluation iterator each time eval is called
  * Adds support for using pre-tokenized data

## Optimal Second Order Methods
* llama_train_taylor.py contains the code for using a Taylor expansion each parameter update
  * Taylor expansion uses the [neural-tangents](https://neural-tangents.readthedocs.io/en/latest/index.html) library
* Includes support for data parallelism, which can be specified with the first argument of the mesh dimensions
