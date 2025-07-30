
class TraceHandler:

    def __init__(self, input_file_path, output_file_path, memory_address_index = 1, burst_index = -1):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.memory_address_index = memory_address_index
        self.burst_index = burst_index

    def parse_memory_address(self, line):
        input, output = line.split(";")
        memory_address_str = input.split(",")[self.memory_address_index]
        memory_address = int(memory_address_str, 16)
        return memory_address

    def read_memory_addresses(self):
        f = open(self.input_file_path, "r")
        lines = f.readlines()
        f.close()
        memory_addresses = [self.parse_memory_address(line) for line in lines]
        return memory_addresses

    def insert_bursts_and_export(self, bursts):
        input_file = open(self.input_file_path, "r")
        input_lines = input_file.readlines()
        input_file.close()

        output_lines = []

        for i, line in zip(list(range(0, len(input_lines))), input_lines):
            first_part, second_part = line.split(";")
            elements = first_part.split(",")
            burst_index = self.burst_index
            if burst_index == -1:
                burst_index = len(elements)
            left_elements = elements[:burst_index]
            right_elements = elements[burst_index:]
            output_line = ""
            for element in left_elements:
                output_line = output_line + element + ","
            output_line = output_line + str(bursts[i])
            for element in right_elements:
                output_line = output_line + "," + element
            output_line = output_line + ";" + second_part
            output_lines.append(output_line)

        output_file = open(self.output_file_path, "a")
        output_file.writelines(output_lines)
        output_file.close()






