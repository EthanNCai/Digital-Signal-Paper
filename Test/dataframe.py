import pandas as pd

# 定义行索引和列索引
row_index = ['a', 'b', 'c']
col_index = list(range(11))

# 创建一个全部为 '-' 的 DataFrame
df = pd.DataFrame('-', index=row_index, columns=col_index)

# 将 DataFrame 输出为 Excel 文件
df.to_excel('output.xlsx', index=True)