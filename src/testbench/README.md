# PredicMem25 - `testbench` directory

This folder contains the testbench and validation utilities for the **PredicMem25** project. It provides the main entry point for running experiments, reading traces, and validating the described hardware.

## How `main.cpp` Works

The `main.cpp` file is the main executable for running validation experiments on the memory prediction modules. Its workflow is as follows:

1. **Command-Line parsing**  
   Parses arguments to select which validation to run. Supported options are:
   - `--validateInputBuffer` / `-vIB`: Validates the input buffer logic.
   - `--validateDictionary` / `-vD`: Validates the dictionary logic.
   - `--validateSVM` / `-vSVM`: Validates the SVM classifier.
   - `--validateGASP` / `-vG`: Validates the GASP prefetcher.
   - `--validateSGASP` / `-vS`: Validates the SGASP prefetcher.
   - `--validateBSGASP` / `-vB`: Validates the BSGASP burst prefetcher.

2. **Experiment setup**  
   For each selected validation, loads the corresponding trace header and sets up a series of experiments using the `Experimentation` class.

3. **Experiment execution**  
   Iterates through each experiment and its operations, feeding inputs to the prediction modules (input buffer, dictionary, SVM, or GASP-based prefetchers) and collecting outputs.

4. **Validation**  
   Compares the outputs against expected results or computes metrics (hit rate, matching rate, precision) to determine if the experiment passes.

5. **Result reporting**  
   Prints whether all selected validations passed.

---

## Types of experiments and validations

The file `experimentation.hpp` implements several experiment types, including:

- **Input buffer validation**  
  Checks the correctness and hit rate of the input buffer. The hit rate is computed as the number of accesses that hit in the input buffer divided by by the number of all input buffer accesses.

- **Dictionary validation**  
  Checks the correctness and hit rate of the dictionary. The hit rate is computed as the number of accesses that hit in the dictionary divided by the number of all dictionary accesses.

- **SVM validation**  
  Checks the correctness and precision of the SVM. The precision is computed as the number of correct SVM predictions divided by the number of all SVM predictions.

- **GASP validation**  
  Validates the GASP prefetcher via its matching rate. The matching rate is computed as the number of all matched resulting prefetches (those cases where the prefetched address is equal to the prefetched address of the baseline GASP given by the trace file) divided by the number of all prefetches of the baseline.

- **SGASP validation**  
  Validates the SGASP prefetcher via its matching rate. The matching rate is computed as the number of all matched resulting prefetches (those cases where the prefetched address is equal to the prefetched address of the baseline SGASP given by the trace file) divided by the number of all prefetches of the baseline.

- **BSGASP validation**  
  Work-in-progress. Validates the BSGASP prefetcher via the matching rate of both the prefetched addresses and the predicted burst lengths.

Each experiment type can be run in strict or "soft" mode, allowing for exact or threshold-based validation of results.

---

This structure allows users to easily run different types of validation and experiments on the developed hardware, ensuring reproducibility and flexibility.