import os
import pandas as pd

from typing import Dict, Tuple, List

def print_results(res_dict: Dict[Tuple[int, int], List[float]]):
    df = pd.DataFrame(columns=[
        "Batch Size",
        "Seq Len",
        "Throughput (token/s)",
        "Min Throughput (token/s)",
        "Max Throughput (token/s)"
    ])

    for (batch_size, seq_len), res in res_dict.items():
        df.loc[len(df)] = [batch_size, seq_len, res[0], res[1], res[2]]
    
    print(df.to_string())