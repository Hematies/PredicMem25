# PredicMem25 - `src` directory

All the code is stored under `src/`, which contains the following sub-directories:

### src/include/

It contains all the hardware descriptions of each of the modules that are described in [the report](report.pdf). It also contains the basic definition of the data types that are used, and the configuration of the hardware and the data types (including their bitfields).

### src/top/

It contains the descriptions of all possible top functions. A top function defines the interface of a target hardware plus its implementation. In this project, each top function is a synthesisable hardware module, either for testing (such as the input buffer, SVM and dictionary modules) or for packaging a prefetcher implementation as an IP core.

### src/testbench/

It contains the implementation of the trace-based testbench used for validating the HLS designs via either C-simulation (software) or co-simulation (software + hardware). It features a command-line interface for all requested operations.