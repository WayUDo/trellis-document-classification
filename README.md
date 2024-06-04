# trellis-document-classification
classify the docs!

This was a fun exercise!

I took the input docs and did the following: 

1. analyzed the docs in analysis/doc_exploration.ipynb (dont forget to check that out)
2. settled on a model after trying a few (BERT embeddings + logistic regression + validated threshold)
3. created the service using fastapi (trellis_document_classification/service.py)
4. created service tests 

The entire project was managed with poetry!