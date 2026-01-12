# RiboNN: A deep learning model to predict translation efficiency from mRNA sequence

For more information, please see our [RiboNN paper](https://www.nature.com/articles/s41587-025-02712-x).

- System requirements:

  This code has been tested on a system with 4 CPUs, 16 Gb RAM, and 1 NViDIA 10A GPU, with Ubuntu 20.04 as the OS (with CUDA Toolkit 11.3 installed). The required softwares are listed in environment.yml.

- To install project requirements:
  ```bash
  sudo apt install make

  # install mamba (https://github.com/conda-forge/miniforge) into "miniforge3/" in the home directory. 
  curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge3-$(uname)-$(uname -m).sh -b 
  ~/miniforge3/bin/mamba shell init 
  source ~/.bashrc

  # clone the repo
  git clone https://github.com/Sanofi-Public/RiboNN.git && cd RiboNN
  
  # install the RiboNN environment
  make install

  # activate the riboNN environment
  mamba activate RiboNN

  ```
  Note: Depending on your network speed, it may take a few minutes to install the required packages.


- To train the RiboNN model from scratch:
   1. Put the training data in a tab-separated text file in the "data" folder, which already contain an example training data file with **fake** TEs. The tab-separated text file should have columns named "tx_id" (unique transcript IDs), "utr5_sequence", "cds_sequence" (including start and stop codons), and "utr3_sequence". Alternatively, the file may have columns named "tx_id", "tx_sequence" (full transcript seuquences containing 5'UTR, CDS, and 3'UTR), "utr5_size" (lengths of the 5'UTRs), and "cds_size" (lengths of the CDSs). The published human and mouse models were trained on data in the Supplementary Tables published in the RiboNN paper.
   2. Edit the path to the training data ("tx_info_path") and other hyperparameters defined in the config/conf.yml file. 
   3. Edit the code below line 18 of src/main.py to control how the model will be trained.
   4. Run `make train` at the terminal to start the training process.
  
- To do transfer learning (using pretrained human multi-task models automatically downloaded from https://zenodo.org/records/17258709):
   1. Put the training data in a tab-separated text file in the "data" folder, which already contain an example training data file. The tab-separated text file should have columns named "tx_id" (unique transcript IDs), "utr5_sequence", "cds_sequence" (including start and stop codons), and "utr3_sequence". Alternatively, the file may have columns named "tx_id", "tx_sequence" (full transcript seuquences containing 5'UTR, CDS, and 3'UTR), "utr5_size" (lengths of the 5'UTRs), and "cds_size" (lengths of the CDSs). 
   2. Edit the path to the training data ("tx_info_path") and other hyperparameters defined in the config/conf.yml file. 
   3. Edit the code below line 118 of src/main.py to control how the model will be trained.
   4. Run `make transfer_learning` at the terminal to start the training process.
  
- To make predictions using pretrained multi-task models automatically downloaded from https://zenodo.org/records/17258709:
  1. Create a tab-separated text file with columns named "tx_id" (unique transcript IDs), "utr5_sequence", "cds_sequence" (including start and stop codons), and "utr3_sequence". Alternatively, the file may have columns named "tx_id", "tx_sequence" (full transcript seuquences containing 5'UTR, CDS, and 3'UTR), "utr5_size" (lengths of the 5'UTRs), and "cds_size" (lengths of the CDSs). 
  2. Save the text file as "prediction_input.txt" in the "data" folder. An example input file can be found in the "data" folder.
  3. (Optional) Edit the code below line 163 of src/main.py to control how the model will be used for prediction.
  4. To use human models for prediction, run `make predict_human` at the terminal. To use mouse models for prediction, run `make predict_mouse`.
  5. The predictions will be automatically written to a tab-separated file named "prediction_output.txt" in the "results/human" or "results/mouse" folder. Pre-existing files with the same name will be overwritten.
  
  **Note:** Input transcripts with 5'UTRs longer than 1,381 nt or combined CDS and 3'UTR sizes larger than 11,937 nt will be excluded in the output.   
