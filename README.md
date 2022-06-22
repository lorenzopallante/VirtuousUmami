# VirtuousUmami

[![pypi button][pypi_image]][pypi_link]

[pypi_image]: https://img.shields.io/pypi/v/m3ba.svg
[pypi_link]: https://pypi.org/project/m3ba/

The **VirtuousUmami** tool predict the umami/non-umami taste of query molecules based on their molecular structures.

This tool was developed within the Virtuous Project (https://virtuoush2020.com/)

[![Virtuous button][Virtuous_image]][Virtuous link]

[Virtuous_image]: https://virtuoush2020.com/wp-content/uploads/2021/02/V_logo_h.png
[Virtuous link]: https://virtuoush2020.com/

The VirtuousUmami is also implemented into a webserver interface at http://195.251.58.251:19009/#/virtuous-umami

### Repo Structure
The repository is organized in the following folders:

- VirtuousUmami/
>Collecting python codes to run the umami prediction

- data/
> Collecting the training and the test sets of the model, the prioritized list of molecular descriptors and the external DBs with their relative umami predictions

- src/
> Containing files needed to effectively run the code

- samples/
> Including examples files to test the code


### Authors
1. [Lorenzo Pallante](https://github.com/lorenzopallante)
2. [Aigli Korfiati](https://github.com/aiglikorfiati)
3. [Lampros Androutsos](https://github.com/lamprosandroutsos)

----------------
## Prerequisites
----------------

1. Clone the `VirtuousUmami` repository from GitHub

        git clone https://github.com/lorenzopallante/VirtuousUmami
        cd VirtuousUmami


2. create conda environment from yml in the src/ folder

        cd src/
        conda env create -f VIRTUOUS.yml
        conda activate VIRTUOUS

----------------
## Installation
----------------

    pip install VirtuousUmami


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
