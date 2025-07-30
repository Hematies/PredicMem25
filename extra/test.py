from burst_distribution import BurstDistributionHandler
from config import Config
from trace import TraceHandler


def basic_test():
    config = Config(region_block_size_log2=32, block_size_log2=0)
    handler = BurstDistributionHandler(config)
    addresses = [
        0,1,0,1,0,1,0,1,2,2,2,2,4,2,4,2,5,1,5,1,5,1,5,1
    ]
    region_model_table = handler.fit_hmm_models(addresses)

    for region, model_and_cathegories_map in region_model_table.items():
        model, cathegories_map = model_and_cathegories_map
        sequence = handler.generate_burst_sequences(model, cathegories_map, 100)

    bursts = handler.fit_hmm_models_and_generate_bursts_sequences(addresses=addresses)
    print("")


def trace_test():
    input_file_path = "prefetcherTrace.txt"
    output_file_path = "prefetcherTrace_out.txt"
    address_index = 1
    burst_index = -1

    trace_handler = TraceHandler(input_file_path, output_file_path, address_index, burst_index)
    addresses = trace_handler.read_memory_addresses()

    config = Config()
    handler = BurstDistributionHandler(config)
    bursts = handler.fit_hmm_models_and_generate_bursts_sequences(addresses)

    trace_handler.insert_bursts_and_export(bursts)
    print("The end!")


trace_test()