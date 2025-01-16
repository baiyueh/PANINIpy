This folder contains the code implementing the partial rankings algorithm developed by Alec and me. The contents of the folder are as follows:

- **partial_rankings.py**: Python module containing the code implementing the greedy agglomerative algorithm to infer partial rankings.
- **preprocessing.py**: Python module containing some utility functions to extract the required parameters from the match list. It is independent of the partial rankings code and does not affect its performance.
- **wolf.txt**: Match list containing dominance interactions between a group of wolves in captivity[^1].
- **example.ipynb**: Jupyter notebook containing an example case applying the partial rankings algorithm to the wolf.txt match list.
- **README.md**: This file

[^1]: J. A. van Hooff and J. Wensing, 11. dominance and its behavioral measures in a captive wolf pack. Man Wolf Adv. Issues Probl. Captive Wolf Res 4, 219 (1987).