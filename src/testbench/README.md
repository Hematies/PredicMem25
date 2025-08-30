# PredicMem25 - `testbench` directory

This folder contains the testbench and validation utilities for the **PredicMem25** project. It provides the main entry point for running experiments, reading traces, and validating the described hardware.

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

## Trace types and structure

The traces that are used for validation are composed of lines with the following format:
`<input field 1>, <...>, <input field N>; <target field 1>, ..., <target field M>`
, where the input fields are the parameter values given as input to the module, and the target fields are those values that are expected to be returned during simulation in Vitis HLS.

The trace types correspond with each type of validation, and are parsed following the code in `src/testbench/reading.hpp`. They are listed as follows:

### Input buffer validation


 
- **Input fields:**
  - `<input buffer address>`
  - `<valid flag>` (always set to true)
  - `<tag>`
  - `<last memory block address>`
  - `<class sequence[0]>, <class sequence[1]>, ..., <class sequence[SEQUENCE_LENGTH-1]>`
  - `<LRU value>`
  - `<perform-read flag>`

- **Target fields:**
  - `<valid flag>`
  - `<tag>`
  - `<last memory block address>`
  - `<entry.sequence[0]>, <entry.sequence[1]>, ..., <entry.sequence[SEQUENCE_LENGTH-1]>`
  - `<LRU value>`
  - `<is-hit flag>`


### Dictionary validation



- **Input fields:**
  - `<index>`
  - `<delta>`
  - `<perform-read flag>`

- **Target fields:**
  - `<valid flag>`
  - `<delta>`
  - `<LFU value>`
  - `<resulting index>`
  - `<is-hit flag>`


### SVM validation



- **Input fields:**
  - `<input class[0]>, ..., <input class[SEQUENCE_LENGTH - 1]>` (input vector)
  - `target class` (target class to learn)

- **Target fields:**
  - `output class`(predicted, output class)


### GASP/SGASP prefetcher validation



- **Input fields:**
  - `<instruction pointer>` (set to zero for SGASP; non-zero for GASP)
  - `<memory block address>`
  - `clock cycle`

- **Target fields:**
  - `<prefetched address[0]>, ..., <prefetched address[MAX_PREFETCHING_DEGREE-1]>`


---

### BSGASP prefetcher validation



- **Input fields:**
  - `<instruction pointer>` (set to zero for SGASP; non-zero for GASP)
  - `<memory block address>`
  - `clock cycle`
  - `burst length`

- **Target fields:**
  - `<prefetched address[0]>, ..., <prefetched address[MAX_PREFETCHING_DEGREE-1]>`
  - `predicted burst length`


---

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

   Additionally, the directory where the traces and the header are located has to be indicated with:
   - `--tracePath <path>` / `-t <path>`: Indicates the path of the directory containing the traces and the header.

2. **Experiment setup**  
   For each selected validation, loads the corresponding trace header and sets up a series of experiments using the `Experimentation` class.

3. **Experiment execution**  
   Iterates through each experiment and its operations, feeding inputs to the prediction modules (input buffer, dictionary, SVM, or GASP-based prefetchers) and collecting outputs.

4. **Validation**  
   Compares the outputs against expected results or computes metrics (hit rate, matching rate, precision) to determine if the experiment passes.

5. **Result reporting**  
   Prints whether all selected validations passed.

---

This structure allows users to easily run different types of validation and experiments on the developed hardware, ensuring reproducibility and flexibility.