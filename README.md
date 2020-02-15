# plagiarism
Plagiarism project

## About

This repository is a classification task of plagiarism with Wikipedia as the source of articles. The algorithm we use here is a Support Vector Machine with RBF kernel. We perform hyperparameter tuning in AWS and deploy it in the notebook. The features that we use are: the containment of 1-gram and 3-gram, and the length of the longest common sequence. 

The training is performed over the a modified version of a dataset by Paul Clough and Mark Stevenson, U. of Sheffield. More information cna be found at [their webpage](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html). 

> Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press.


## For Review

- Data preparation - feature engineering: [pdf](2_Plagiarism_Feature_Engineering.pdf) or [html](2_Plagiarism_Feature_Engineering.html).
- Training, hyperparameter tuning and deployement:  [pdf](3_Training_a_Model.pdf) or [html](3_Training_a_Model.html).



## Getting Started

You would need to clone this repository in a jupyter environment in SageMaker. Then follow the jupyter notebooks in order.


## Requirements
In principle if it is run in AWS, the requirements are included in the folder train

