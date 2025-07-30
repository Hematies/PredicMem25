import argparse

from burst_distribution import BurstDistributionHandler
from config import Config
from trace import TraceHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='BGASP burst values generator')
    parser.add_argument('-i', '--input', help='Path to the input trace file',
                        type=str, required=True)
    parser.add_argument('-o', '--output', help='Path to the output trace file',
                        type=str, required=True)
    parser.add_argument('-a', '--address_index', help='Memory address index in trace line',
                        type=int, required=False, default=1)
    parser.add_argument('-b', '--burst_index', help='Burst value index in trace line',
                        type=int, required=False, default=-1)
    args = parser.parse_args()

    input_file_path, output_file_path = args.input, args.output
    address_index, burst_index = args.address_index, args.burst_index

    trace_handler = TraceHandler(input_file_path, output_file_path, address_index, burst_index)
    addresses = trace_handler.read_memory_addresses()

    config = Config()
    handler = BurstDistributionHandler(config)
    bursts, next_bursts = handler.fit_hmm_models_and_generate_bursts_sequences(addresses)

    trace_handler.insert_bursts_and_export(bursts, next_bursts)
    print("The end!")
