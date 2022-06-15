# Virtuous Umami

The Virtuous Umami tool predict the umami/non-umami taste of query molecules based on their molecular structures.

This tool is mainly based on:

    1. VirtuousUmami-main.py: a main script which calls the following functionalities
    2. Virtuous.py: library of preprocessing functionalities
    3. testing_umami.py: prediction code

---------------------
## 0. Prerequisites
---------------------

If you use conda, you can directly create a new environment using the VIRTUOUS.yml file in the src/ folder.

Otherwise, you'll need to run the following commands:

>conda install -c conda-forge rdkit

>pip install tqdm

>conda install -c conda-forge chembl_structure_pipeline

>conda install -c mordred-descriptor mordred

>pip install knnimpute

>pip install joblib

>pip install Cython

>pip install scikit-learn


---------------------
## 1. VirtuousUmami main
---------------------

Probably the only code you need.

To learn how to run, just type:

    python VirtuousUmami-master.py --help

And this will print the help message of the program:

    usage: VirtuousUmami-master.py [-h] [-c COMPOUND] [-f FILE] [-d DIRECTORY] [-v]

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


**_In case the user would like to run the Virtuous.py and testing_umami.py scripts as stand-alone packages, it is suggested to read the following instructions_**


-------------------
## 2. Virtuous Library
-------------------

The Virtuous Library contains several functionality to process molecules using RDKit and calculate molecular descriptors.

 ===== Virtuous Library requires python3 and the following standard python libraries =====

    1. sys
    2. os
    3. numpy
    4. pandas

 ==== Additional libraries required ======

    1. chembl_structure_pipeline (https://github.com/chembl/ChEMBL_Structure_Pipeline)
        conda install -c conda-forge chembl_structure_pipeline

    2. Mordred
        conda install -c mordred-descriptor mordred

    3. rdkit
        conda install -c conda-forge rdkit

    4. networkx (version 2.1.0 required by mordred)
        pip install networkx==2.1.0


==== Installation ===

Just add the folder containing the Virtuous.py library in your PYTHONPATH variable


----------------
## 3. Testing Umami
----------------

 ===== testing_umami.py was developed with Python v3.8.10 and requires the following python libraries =====

	1.	pandas: 0.25.3
	2.	numpy: 1.17.4
	3.	joblib: 0.14.0
	4.	knnimpute: 0.1.0

 ==== How to run ====

This script takes as input csv file(s) with Mordred molecular descriptors calculated from the Virtuous.py library.

We provide two indicative testing datasets in the samples folder:

	1.	test_sample.csv:  With only one sample
	2.	test_samples.csv: With nine samples

When we want to predict a new compound, we have to save it in a file like test_sample.csv.
For batch computations, we follow the test_samples.csv file.

The script only needs the name of the dataset as an input. So, we can run it as follows:

python3 testing_umami.py test_sample.csv
