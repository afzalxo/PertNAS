## Discrete Architectural Perturbations for Neural Architecture Search

### Requirements

Install dependencies using the `requirements.txt` file

`pip install -r requirements.txt`

This repository is build upon works by [FFCV](https://ffcv.io/), [DARTS](https://github.com/quark0/darts), [Progressive DARTS](https://github.com/chenxin061/pdarts), and [DARTS+PT](https://github.com/ruocwang/darts-pt).

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

Use `--distributed` flag even if using a single GPU.

The searched genotype would be printed on the terminal and is also logged in the log directory

The results in the paper were produced using `npop=4`, `batch_size=256`, and 4 GPUs.  

During search we only use one type of cell, meaning reduction cells have the same architecture as normal cells but with stride-2 ops to reduce spatial resolution. This is because eval networks according to the DARTS-like works consist of 20 total cells with only 2 reduction cells, meaning searching for reduction cells separately does not impact accuracy by much (Although the search cost is reduced significantly given we only have to search one kind of cell rather than two). 

Furthermore, search network consists only of 5 cells since we found the accuracy of searched models using only 5 cells to search similar to that achieved when using 8 cells as utilized by DARTS-like works. 

Athough these changes might raise concerns on fairness of comparison of results against previous works, we argue that these trivial optimizations to improve the search cost should be scrutinized and adopted by future works, given they help reduce the compute costs of search with little to no impact on accuracy. Rather than debating fairness of comparison between existing works that could be improved using these trivial modifications for search, we should be focused on making NAS open for researchers and practitioners with low compute budgets. 

Evaluation
----------
For CIFAR and ImageNet, evaluation of the searched cells follows the same procedure as DARTS-like works (e.g., [DARTS+PT](https://github.com/ruocwang/darts-pt)).

#### CIFAR10
Generated best accuracy of 97.51% on CIFAR10 using evolutionary approach. Eval log available in `eval-evolutionary` directory. We use searched cells from **two top runs**, best cell as normal and second best as reduction cell for this evaluation. 

#### ImageNet
Evaluation follows similar procedure as DARTS-like works, except for the slightly longer training of 300 epochs rather than 250 epochs utilized by most existing works. This has little impact on accuracy though. Top-1 accuracy of 75.53% is achieved as can be seen in logs in `eval-imagenet` directory. The saved model file `model_best.pth.tar` is also available in the logs directory. Evaluation cells are the same as that used for CIFAR10 (i.e., best cells from **two top search runs** on CIFAR10). 

#### NASBench-301
For **NASBench-301**, please follow the installation instructions at [their repo](https://github.com/automl/nasbench301)

Place the generated cell genotype in `nasbench301/example.py` file in the `genotype_config` variable and run the `example.py` script. We utilize same architecture for both normal and reduction cells. 

The best searched cell using supernet-based search achieves 94.58% and the best searched cell using evolutionary approach achieves 94.71% accuracy on NASBench-301. Architectures of best cells are as below


Searched Cells
--------------

Supernet-based search:

<img src="https://github.com/afzalxo/PertNAS/blob/main/result_genotypes/bestcell-supernet.png" width=25%>

Evolutionary search:

<img src="https://github.com/afzalxo/PertNAS/blob/main/result_genotypes/bestcell-evol.png" width=50%>

