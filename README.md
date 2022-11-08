## Discrete Architectural Perturbations for Neural Architecture Search

Requirements
------------

Install dependencies using the `requirements.txt` file

`pip install -r requirements.txt`

Search Evolutionary
-------------------

Search on the DARTS search space with the evolutionary strat, i.e., expand an edge followed by perturb-val.

**Step 1** - Install FFCV from [here](https://github.com/libffcv/ffcv#install-with-anaconda)

**Step 2** - Create FFCV-compatible CIFAR-10 dataset using `write_datasets.py` script [here](https://github.com/libffcv/ffcv/tree/main/examples/cifar) and save it somewhere. The location containing the FFCV compatible dataset should look like this

```
cifar_ffcv
|-- cifar_test.beton
`-- cifar_train.beton
```

**Step 3** - Searching using a population `npop` of 4 discrete models

```
torchrun --nnodes=1 --nproc_per_node=4 search_evolutionary.py --seed 1 
                                                             --save ./ 
                                                             --note 'evolutionary' 
                                                             --npop 4 
                                                             --distributed 
                                                             --cluster local 
                                                             --train_path <path to cifar_ffcv directory from step 2>
```

The genotype would be printed on the terminal and is also logged in the log directory

The results in the paper were produced using `npop=4`, `batch_size=256`, and 4 GPUs. 

Evaluation
----------
For CIFAR and ImageNet, evaluation of the searched cells follows the same procedure as DARTS-like works (e.g., [DARTS+PT](https://github.com/ruocwang/darts-pt)).

For **NASBench-301**, please follow the installation instructions at [their repo](https://github.com/automl/nasbench301)

Place the generated cell genotype in `nasbench301/example.py` file in the `genotype_config` variable and run the `example.py` script. We utilize same architecture for both normal and reduction cells. 

The best searched cell using supernet-based search achieves 94.58% and the best searched cell using evolutionary approach achieves 94.71% accuracy. Architectures of best cells are as below

Searched Cells
--------------

Supernet-based search:

![alt text][cell-supernet]

[cell-supernet]: https://github.com/afzalxo/PertNAS/blob/main/result_genotypes/bestcell-supernet.png "Searched Cell using Supernet-based Method"

Evolutionary search:

![alt text][cell-evolutionary]

[cell-evolutionary]: https://github.com/afzalxo/PertNAS/blob/main/result_genotypes/bestcell-evol.png "Searched Cell using Evolutionary search method"

