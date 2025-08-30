# PredicMem25 - `testbench` Directory

This folder contains the testbench and validation utilities for the **PredicMem25** project. It provides the main entry point for running experiments, reading traces, and validating the prediction algorithms.

## How `main.cpp` Works

The `main.cpp` file is the main executable for running validation experiments on the memory prediction modules. Its workflow is as follows:

1. **Command-Line Parsing**  
   Parses arguments to select which validation to run. Supported options are:
   - `--validateInputBuffer` / `-vIB`: Validates the input buffer logic.
   - `--validateDictionary` / `-vD`: Validates the dictionary logic.
   - `--validateSVM` / `-vSVM`: Validates the SVM classifier.
   - `--validateGASP` / `-vG`: Validates the GASP prefetcher.
   - `--validateSGASP` / `-vS`: Validates the SGASP prefetcher.
   - `--validateBSGASP` / `-vB`: Validates the BSGASP burst prefetcher.

2. **Experiment Setup**  
   For each selected validation, loads the corresponding trace header and sets up a series of experiments using the `Experimentation` class.

3. **Experiment Execution**  
   Iterates through each experiment and its operations, feeding inputs to the prediction modules (input buffer, dictionary, SVM, prefetchers) and collecting outputs.

4. **Validation**  
   Compares the outputs against expected results or computes metrics (hit rate, match rate, precision) to determine if the experiment passes.

5. **Result Reporting**  
   Prints whether all selected validations passed.

---

## Types of Experiments and Validations

The file `experimentation.hpp` implements several experiment types, including:

- **Input Buffer Validation**  
  Checks the correctness and hit rate of the input buffer logic.

- **Dictionary Validation**  
  Validates the dictionary module for address delta and confidence prediction.

- **SVM Validation**  
  Tests the Support Vector Machine classifier for address class prediction.

- **GASP Validation**  
  Validates the GASP prefetcher for memory address prediction.

- **SGASP Validation**  
  Validates the SGASP prefetcher for region-based address prediction.

- **BSGASP Validation**  
  Validates the BSGASP prefetcher for burst-aware memory prediction, including burst length accuracy.

Each experiment type can be run in strict or "soft" mode, allowing for exact or threshold-based validation of results.

---

This structure allows users to easily run different types of validation and experiments on the memory prediction algorithms, ensuring reproducibility and flexibility.