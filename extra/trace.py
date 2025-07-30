
class TraceHandler:

    def __init__(self, input_file_path, output_file_path, memory_address_index = 1,
                 burst_index = -1, next_burst_index = -1):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.memory_address_index = memory_address_index
        self.burst_index = burst_index
        self.next_burst_index = next_burst_index

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

    def __insert_element_in_line(self, line, new_element, index, separator=","):
        elements = line.split(separator)
        if index == -1:
            index = len(elements)
        left_elements = elements[:index]
        right_elements = elements[index:]
        output_line = ""
        for element in left_elements:
            output_line = output_line + element + separator
        output_line = output_line + new_element
        for element in right_elements:
            output_line = output_line + separator + element
        return output_line

    def insert_bursts_and_export(self, bursts, next_bursts):
        input_file = open(self.input_file_path, "r")
        input_lines = input_file.readlines()
        input_file.close()

        output_lines = []

        for i, line in zip(list(range(0, len(input_lines))), input_lines):
            first_part, second_part = line.split(";")
            first_part, second_part = first_part.strip(), second_part.strip()
            burst_index, next_burst_index = self.burst_index, self.next_burst_index
            output_line = self.__insert_element_in_line(first_part, str(bursts[i]), burst_index)
            output_line = output_line + ";"
            output_line = output_line + \
                          self.__insert_element_in_line(second_part, str(next_bursts[i]), next_burst_index) +  "\n"
            output_lines.append(output_line)

        output_file = open(self.output_file_path, "w")
        output_file.writelines(output_lines)
        output_file.close()






