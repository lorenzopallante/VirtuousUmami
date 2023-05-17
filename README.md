# VirtuousUmami

The **VirtuousUmami** tool predict the umami/non-umami taste of query molecules based on their molecular structures. 
>Pallante, L., Korfiati, A., Androutsos, L., Stojceski, F., Bompotas, A., Giannikos, I., Raftopoulos, C., Malavolta, M., Grasso, G., Mavroudi, S., Kalogeras, A., Martos, V., Amoroso, D., Piga, D., Theofilatos, K., & Deriu, M. A. (2022). Toward a general and interpretable umami taste predictor using a multi-objective machine learning approach. Scientific Reports, 12(1), 21735. https://doi.org/10.1038/s41598-022-25935-3

The VirtuousUmami is also implemented into a webserver interface at http://195.251.58.251:19009/#/virtuous-umami

This tool was developed within the Virtuous Project (https://virtuoush2020.com/)

[![Virtuous button][Virtuous_image]][Virtuous link]

[Virtuous_image]: https://virtuoush2020.com/wp-content/uploads/2021/02/V_logo_h.png
[Virtuous link]: https://virtuoush2020.com/


----------------
### Repo Structure
The repository is organized in the following folders:

- VirtuousUmami/
>Collecting python codes and sources files to run the umami prediction

- data/
> Collecting the training and the test sets of the model, the prioritized list of molecular descriptors and the external DBs with their relative umami predictions

- samples/
> Including examples files to test the code


### Authors
1. [Lorenzo Pallante](https://github.com/lorenzopallante)
2. [Aigli Korfiati](https://github.com/aiglikorfiati)
3. [Lampros Androutsos](https://github.com/lamprosandroutsos)

----------------
## Prerequisites
----------------

1. Create conda environment:

        conda create -n myenv python=3.8
        conda activate myenv

2. Install required packages:

        conda install -c conda-forge rdkit chembl_structure_pipeline
        conda install -c mordred-descriptor mordred
        pip install tqdm knnimpute joblib Cython scikit-learn==0.22.2 xmltodict

3. Clone the `VirtuousUmami` repository from GitHub

        git clone https://github.com/lorenzopallante/VirtuousUmami

Enjoy!        

---------------------------
## How to use VirtuousUmami
---------------------------

The main code is `VirtuousUmami.py` within the VirtuousUmami folder.

To learn how to run, just type:

    python VirtuousUmami.py --help

And this will print the help message of the program:

    usage: VirtuousUmami.py [-h] [-c COMPOUND] [-f FILE] [-d DIRECTORY] [-v]

    VirtuousUmami: ML-based tool to predict the umami taste

    optional arguments:
      -h, --help            show this help message and exit

      -c COMPOUND, --compound COMPOUND
                        query compound (allowed file types are SMILES, FASTA, Inchi, PDB, Sequence, Smarts, pubchem name)

      -f FILE, --file FILE  text file containing the query molecules

      -d DIRECTORY, --directory DIRECTORY name of the output directory

      -v, --verbose         Set verbose mode

To test the code you can submit an example txt file in the "samples" fodler (test.txt)      

The code will create a log file and an output folder containing:

    1. "best_descriptors.csv": a csv file collecting the 12 best molecular descriptors for each processed smiles on which the prediction relies
    2. "descriptors.csv": a csv file collecting all the calculated molecular descriptors for each processed smiles
    3. "result_labels": a txt file containing the predicted taste classes (umami/non-umami) for each processed smiles
    4. "predictions.csv": a csv summarising the results of the prediction


------------------
## Acknowledgement
------------------

The present work has been developed as part of the VIRTUOUS project, funded by the European Union’s Horizon 2020 research and innovation program under the Marie Sklodowska-Curie-RISE Grant Agreement No 872181 (https://www.virtuoush2020.com/).
