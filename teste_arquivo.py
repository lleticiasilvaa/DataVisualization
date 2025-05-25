import pandas as pd

df = pd.read_csv('/home/user/Documentos/UFV/Mestrado/VisualizacaoDados/DataVisualization/estrategias.csv')

df['name'] = 'Qwen2.5-'+df['parametros'].astype(str) + ' - ' + df['estrategia'].astype(str) + ' - ' + df['esquema'].astype(str)
df['memory'] = df['Memoria']
df['input_tokens'] = df['Input']
df['output_tokens'] = df['Output']
df['inference_time'] = df['Tempo']
df['ex_all_spider-dev'] = df['Spider-Dev ']
df['ex_all_cnpj'] = df['CNPJ']


df = df.drop(['parametros', 'estrategia', 'esquema', 'Input', 'Output', 'Total',
       'Tempo', 'Memoria', 'GPT-4o-mini', 'max_modelos', 'Req/Hora 1 modelo',
       'Req/Hora max_modelo', 'Spider-Dev ', 'CNPJ'], axis=1)

print(df)

df.to_csv('saida.csv', index=False)
