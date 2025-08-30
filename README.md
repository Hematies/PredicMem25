# PredicMem25

## What is this repository?

This repository contains all the C++ code of the PredicMem25 project: a hardware description of the Spatial Greedily Accurate SVM-based Prefetcher (SGASP) cache memory prefetcher in High Level Synthesis (HLS), oriented to specific-purpose. A full description of the project, its proposal and its implementation details is included in this report file, which is written in a scientific paper style.

The SGASP prefetcher (along with the GASP family of prefetchers) was originally proposed and thoroughly described in a paper that is under peer-review right now. Nevertheless, its description is also detailed in the provided report file.

## Why HLS?

HLS is a hardware description method that exploits a high-level abstraction language (such as C++) to provide behavioral definition of the target hardware. Among the advantages that this provides, in this project we considerably leverage: (1) the ability to integrate testbenches based on complex, traced-based tests; and (2) the potential of C++ for the description of the hardware by using data structures, functions, classes, loops, etc, which greatly simplifies the design and development.

## How is the repository structured?

All the code is stored under src/, which contains the following sub-directories:

### include/

It contains all the hardware descriptions of each of the modules that are described in the report. It also contains the basic definition of the data types that are used, and the configuration of the hardware and the data types (their bitfields).

### top/

It contains the descriptions of all possible top functions. A top function defines the interface of a target hardware plus its implementation. In this project, each top function is a synthesisable hardware module, either for testing (such as the input buffer, SVM and dictionary modules) or for packaging a prefetcher implementation as an IP core.

### test/

It contains the implementation of the trace-based testbench used for validating the HLS designs via either C-simulation (software) or co-simulation (software + hardware). It features a command-line interface for all requested operations.

## How can I synthesise a hardware implementation?

Any of the top functions that are described in src/top/src.cpp can be synthesisable as a hardware module. In Vitis HLS, this can be done by adding a "#pragma HLS TOP name=<function name>" under the function header, and indicating the function name as top in "Project" > "Project Settings" > "Synthesis" > "Top funcion". The module is synthesised by clicking in "Synthesis". Finally, it can exported as IP core in "Tool" > "Export as RTL".

## How can I validate a hardware implementation?

Six different modules can be validated in the current version of PredicMem25: the input buffer, the dictionary, the SVM, the GASP prefetcher, the SGASP prefetcher and the BSGASP prefetcher. The validation of GASP and BSGASP prefetchers are in a work-in-progress-state, and thus they are not included in the report.

The file src/testbench/main.cpp file can be modified to call to a custom implementation (given as top function) of a hardware module. For example, it is possible modify the code to call a top function of SGASP based on a single pipeline design, like prefetchWithSGASPWithNopWithDataflow(). The file src/testbench/experimentation.hpp includes validation models for monitoring the functioning of the target module.

A trace file has to be indicated through the command-line interface for the validation. In our work, we employ a trace format for each hardware module to validate. They can be checked in the documentation in src/testbench/README.md.

Finally, in order to run the validation, either C-simulation or co-simulation are the available options in Vitis HLS. The C-simulation can be started simply by clicking "C-simulation". In the case of a co-simulation, the target hardware module of src/top/top.cpp has to be synthesised first, and then the process can start by clicking "Co-simulation".

## Authorship and contact.

This repository was made thanks to Pablo Sánchez (owner of this repository), Antonio Pérez, Plácido Fernández, Dagnier Curra, Santiago Díaz and Antonio Ríos. The detailed authorship and contact information is presented in the report file.

## Licensing
