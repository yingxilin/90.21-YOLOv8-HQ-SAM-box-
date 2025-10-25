import pyarrow.parquet as pq
path = r"D:\search\fungi\26\data\masks\FungiTastic-Mini-ValidationMasks.parquet"
table = pq.read_table(path)
print("Columns in Parquet:", table.column_names[:20])
print("Example row:")
print(table.to_pandas().head(3))
