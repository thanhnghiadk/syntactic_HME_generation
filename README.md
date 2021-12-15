# syntactic_data_generation_HME
This project aim to generate syntactically valid handwritten mathematical expression patterns. The patterns is generated from the CROHME 2014 training set. We decompose the handwritten mathematical expression(HME) patterns in the source dataset to get their structures and then use the structures for generating new syntactically valid patterns.

We publish the generation code as well as the mathematical grammar. The code for building a parser from the grammar is not provided due to private reason. 

The code for generating HMEs is run after the data decomposition step to get the strucutures of all handwritten mathematical expression in the dataset. The file containing the strucutres of HMEs is also provied.

## Citation
If you find the code useful in your research, please consider citing:

    @article{TRUONG202283,
        title = {Syntactic data generation for handwritten mathematical expression recognition},
        journal = {Pattern Recognition Letters},
        volume = {153},
        pages = {83-91},
        year = {2022},
        issn = {0167-8655},
        doi = {https://doi.org/10.1016/j.patrec.2021.12.002},
        url = {https://www.sciencedirect.com/science/article/pii/S0167865521004293},
        author = {Thanh-Nghia Truong and Cuong Tuan Nguyen and Masaki Nakagawa},
        keywords = {Syntactic data generation, Context free grammar, Parser, Mathematical expression, Handwriting recognition}
    }


We are updating the source code to make all the processes of data generation available.
