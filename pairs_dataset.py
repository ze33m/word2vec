from datasets import load_from_disk
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import yaml
import os
from array import array

"""
intdataset/   --->   pairs/

Строит parquet датасет из пар [target,context]
"""

if __name__ == '__main__':

    load_dataset = load_from_disk("intdataset")

    os.makedirs("pairs", exist_ok=True)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    window_size = config["dataset"].get("window_size", 5)

    schema = pa.schema([
        ("target", pa.int32()),
        ("context", pa.int32()),
    ])

    SHARD_ROWS = 50_000_000     
    shard_id = 0
    rows = 0
    buf_t, buf_c = array('i'), array('i') # буферы для таргета и контекста

    def new_writer(i):
        return pq.ParquetWriter(
            f"pairs/shard-{i:05d}.parquet",
            schema,
            compression="zstd"
        )

    writer = new_writer(shard_id)
    try:
        for row in tqdm(load_dataset, desc="проход по строкам"):
            toks = row["tokens"]
            for i in range(window_size, len(toks) - window_size):
                t = toks[i]
                for offset in range(-window_size, window_size+1):
                    if offset != 0:  
                        buf_t.append(t)
                        buf_c.append(toks[i+offset])
                        rows += 1

                        # когда достигли SHARD_ROWS строчек, записываем все в шард и заново собираем буферы
                        if rows >= SHARD_ROWS:
                            arr_t = pa.array(buf_t, type=pa.int32())
                            arr_c = pa.array(buf_c, type=pa.int32())
                            table = pa.Table.from_arrays([arr_t, arr_c], schema=schema)
                            writer.write_table(table)
                            writer.close()

                            shard_id += 1
                            writer = new_writer(shard_id)
                            rows = 0
                            buf_t, buf_c = array('i'), array('i')
        if len(buf_t) > 0:
            arr_t = pa.array(buf_t, type=pa.int32())
            arr_c = pa.array(buf_c, type=pa.int32())
            writer.write_table(pa.Table.from_arrays([arr_t, arr_c], schema=schema))
    
    finally:
        writer.close()

