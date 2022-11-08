## Source Code for 'Discrete Architectural Perturbations for Neural Architecture Search'

Search Evolutionary
-------------------

Search on the DARTS search space with the evolutionary strat, i.e., expand an edge followed by perturb-val.

Step 1 - Install FFCV from [here](https://github.com/libffcv/ffcv#install-with-anaconda)

Step 2 - Create FFCV-compatible CIFAR-10 dataset using `write_datasets.py` script [here](https://github.com/libffcv/ffcv/tree/main/examples/cifar) and save it somewhere. The location containing the FFCV compatible dataset should look like this

```
cifar_ffcv
|-- cifar_test.beton
`-- cifar_train.beton
```

Step 3 - Searching using a population `npop` of 4 discrete models
`torchrun search_evolutionary.py --seed 1 --save ./ --note 'evolutionary' --npop 4 --distributed --cluster local --train_path <path to cifar_ffcv directory from step 2>`

The genotype would be generated on the terminal and is also logged in the log directory
