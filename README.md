# Neural-HATS

This repository is the implementation of the paper: “[Neural-HATS: Neural Hybrid Approach for Time Series Causal Discovery](https://par.nsf.gov/biblio/10579073)" 

## Overview

- ./CI_tests contains the scripts to perform the conditional independence tests to generate a CI matrix using Attention-based Encoder-Decoder architecture and KCI tests
- The generated CI matrix is saved in the ./CI_tests/outputs directory
- To run each baseline, provide the path to the data, their ground truth causal graph, and the generated CI matrix
- The hybrid baseline methods are provided in individual directories, each of them requiring their own environments and settings to run

## Usage
- Clone the repository from GitHub and navigate into the project directory. Replace my-project with your preferred folder name:
    > - git clone https://github.com/SaimaAbsar/NTiCD.git
    > - cd my-project

- To generate CI matrix: 
    > - $cd CI_tests 
    > - (Update the script name and the corresponding paths in the run_CI_tests.sh file) 
    > - $bash run_CI_tests.sh 

- To run the baselines:
    > GVAR: 
            > - $ cd baselines/GVAR/
            > - update the path to the input data, input causal graph (for accurracy), and CI matrix in run_grid_search_modified.py and training.py 
            > - $bash run_grid_search_newdata.sh or run_grid_search_fMRI 

    > NTiCD: 
            > - $ cd baselines/NTiCD/
            > - (Update paths in the run_NTiCD.sh file) 
            > - $bash run_NTiCD.sh

    > DYNOTEARS: 
            > - $ cd baselines/DYNOTEARS/
            > - (Update the paths in the run.sh file) 
            > - $bash run.sh

    > NTS-NOTEARS: 
            > - $ cd baselines/NTS_NOTEARS/notears
            > - (Update the paths in the run.sh file) 
            > - $bash run.sh


## Citation

If you use this work in your research, please cite:

```bibtex
@article{osti_10579073,
  place       = {Country unknown/Code not available},
  title       = {Neural-HATS: Neural Hybrid Approach for Time Series Causal Discovery},
  url         = {https://par.nsf.gov/biblio/10579073},
  abstractNote= {},
  journal     = {},
  publisher   = {AAAI'25 Workshop on Artificial Intelligence for Time Series Analysis (AI4TS)},
  author      = {Absar, Saima and Huang, Wen and Wu, Yongkai and Zhang, Lu},
}
