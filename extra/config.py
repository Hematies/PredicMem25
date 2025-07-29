import math


class Config:
    def __init__(self,
                 num_address_bits=32,
                 block_size_log2=6,
                 region_block_size_log2=6,
                 max_block_burst_length_log2=2):
        self.num_address_bits = num_address_bits
        self.block_size_log2 = block_size_log2
        self.region_block_size_log2 = region_block_size_log2
        self.max_block_burst_length_log2 = max_block_burst_length_log2

        self.num_region_address_bits = num_address_bits - region_block_size_log2 - block_size_log2
        self.num_block_address_bits = num_address_bits - block_size_log2

        self.block_size = 1 << block_size_log2
        self.region_block_size = 1 << region_block_size_log2
        self.region_size = 1 << (region_block_size_log2 + block_size_log2)
        self.max_block_burst_length = 1 << max_block_burst_length_log2
