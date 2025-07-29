from burst_distribution import BurstDistributionHandler
from config import Config

config = Config()
handler = BurstDistributionHandler(config)
region_model_table = handler.fit_hmm_models([
    0,1,0,1,0,1,0,1,2,2,2,2,4,2,4,2,5,1,5,1,5,1,5,1
])

for region, model_and_cathegories_map in region_model_table.items():
    model, cathegories_map = model_and_cathegories_map
    sequence = handler.generate_burst_sequences(model, cathegories_map, 100)
    print()
