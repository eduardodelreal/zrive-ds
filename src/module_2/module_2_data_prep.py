import boto3
import pandas as pd
import os

import datetime

'''
# Inicializar el cliente S3
s3 = boto3.client('s3')

# Definir el bucket y la carpeta de origen
bucket_name = 'zrive-ds-data'
prefix = 'groceries/sampled-datasets/'

# Listar los archivos en el directorio especificado
objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Crear un directorio local para guardar los archivos
local_directory = 'local_data'
os.makedirs(local_directory, exist_ok=True)

# Descargar cada archivo
for obj in objects.get('Contents', []):
    key = obj['Key']
    if not key.endswith('/'):  # Ignorar carpetas (keys que terminan en '/')
        destination_path = os.path.join(local_directory, key.split('/')[-1])
        s3.download_file(bucket_name, key, destination_path)
'''

#first of all we do a little of analysis, as we open all the different datasets

df_inventory = pd.read_parquet('local_data/inventory.parquet', engine='pyarrow')
df_abandoned_carts = pd.read_parquet('local_data/abandoned_carts.parquet', engine='pyarrow')
df_orders = pd.read_parquet('local_data/orders.parquet', engine='pyarrow')
df_regulars = pd.read_parquet('local_data/regulars.parquet', engine='pyarrow')
df_users = pd.read_parquet('local_data/users.parquet', engine='pyarrow')

############################################3
#funciones de prueba
def mean_price():
    price_column=df_inventory["price"]
    mean_price = price_column.mean()
    print(mean_price)

def order_dates(n:int):
    
    dates = pd.to_datetime(df_orders['order_date'])
    new_dates= dates.to_list()
    if n==0: #si es cero que nos printee todo
        print(new_dates)
    else:
        for i in range(n): #sino que nos de un rango de fechas especificas
            print(new_dates[i])

######################################################
            
def crear_df_from_orders_as_productid_n_orders_purch_prob(): #creamos un df con id del producto, 
    #el num de veces que se ha pedido y la probabilidad de compra
    
    # convertir columna de pedidos en una serie de valores individuales.
    exploded_items = df_orders['ordered_items'].explode()
    #recuento de la frecuencia de cada 'item_id' con value_counts():   
    item_counts = exploded_items.value_counts()

# 'item_counts' es ahora una Serie donde el índice es el 'item_id' y el valor es el recuento de frecuencias.
    #print(item_counts)

# Convertimos la serie de recuentos en un DataFrame con reset_index
    item_counts_df_oders = item_counts.reset_index()
    item_counts_df_oders.columns = ['product_id', 'number_of_orders']

# Calculamos el número total de orders
    total_orders = len(df_orders)

# Agregamos la columna de probabilidad de compra dividiendo por el número total de órdenes
    item_counts_df_oders['purchase_probability'] = item_counts_df_oders['number_of_orders'] / total_orders

# Mostramos los primeros elementos para verificar
    #print(item_counts_df_oders.head())
    print(len(item_counts_df_oders))
    return item_counts_df_oders
    
def combinar_datasets_prod_pedidos_inventario():
    item_counts_df_oders=crear_df_from_orders_as_productid_n_orders_purch_prob()
    df_ordered_products_existing=item_counts_df_oders[item_counts_df_oders["product_id"].isin(df_inventory["variant_id"])]
    print(df_ordered_products_existing.head())
    print("length of dataset: ",len(df_ordered_products_existing), " and length of inventory: ", len(df_inventory))
    return df_ordered_products_existing

def combinar_datasets_abandoned_cart_inventario():
    item_abandoned=crear_df_from_orders_as_productid_n_orders_purch_prob() #cambiar por funcion equivalente 
    df_ordered_products_existing=item_counts_df_oders[item_counts_df_oders["product_id"].isin(df_inventory["variant_id"])]
    print(df_ordered_products_existing.head())
    print("length of dataset: ",len(df_ordered_products_existing), " and length of inventory: ", len(df_inventory))
    return df_ordered_products_existing
    

            



def main():
   # mean_price()
   #order_dates(5)
   #crear_df_from_orders_as_productid_n_orders_purch_prob()
   #print(len(df_inventory))
   #print(len(df_orders))
   combinar_datasets_prod_pedidos_inventario()
   


if __name__ == "__main__":
    main()
