# Overview

This repository contains files used in the pipeline described in our Bachelor's Thesis in Statistics and Data Science: \*insert link\*

Disclaimer: The KC-extraction process and the computing of embeddings were done in Python, while the clustering analysis was performed in R.

The included files are not sufficient for reproducing the entire pipeline, specifically the KC-extraction process. However, it is possible to reproduce the embeddings and the clustering analysis. To do this, you must:

a. Convert the KCs in **KCs.txt** into embeddings with the code in **st-embeddings.py**. Note: The script expects json-format as input.

b. Implement the results to the R-file and run the code.



# File descriptions

* **KCs.txt** - Text file with all extracted knowledge components ("skill"; includes non-unique components) and corresponding question ids ("qid")
* **kc-extraction.py** - Main script for KC-extraction
* **llm.py** - Configuration script for KC-extraction
* **r-code.qmd** - R-code used for the clustering analysis
* **st-embeddings.py** - Code used to convert KCs into embeddings



If you have any questions about the code or the data, please contact either rikard.hsieh@gmail.com or svante.hellgren1@gmail.com

