# Robust Spectral Clustering (RSC)
<img src="https://www.in.tum.de/fileadmin/w00bws/daml/rsc/rsc.png" width="500">

Implementation of the method proposed in the paper: "[Robust Spectral Clustering for Noisy Data: Modeling Sparse Corruptions Improves Latent Embeddings](https://dl.acm.org/citation.cfm?id=3098156)", Aleksandar Bojchevski, Yves Matkovic, and Stephan Günnemann, SIGKDD 2017.

## Installation
```bash
python setup.py install
```

## Requirements
* numpy/scipy
* sklearn


## Demo
See example.ipynb for a comparison with vanilla Spectral Clustering on the moons dataset.

## Cite
Please cite our paper if you use this code in your own work.
```
@inproceedings{bojchevski2017robust,
  title={Robust Spectral Clustering for Noisy Data: Modeling Sparse Corruptions Improves Latent Embeddings},
  author={Bojchevski, Aleksandar and Matkovic, Yves and G{\"u}nnemann, Stephan},
  booktitle={Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={737--746},
  year={2017},
  organization={ACM}
}
```
