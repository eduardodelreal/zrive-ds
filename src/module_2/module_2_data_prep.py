import boto3
import pandas as pd
import os
import datetime



#first of all we do a little of analysis, as we open all the different datasets

df_inventory = pd.read_parquet('local_data/inventory.parquet', engine='pyarrow')
df_abandoned_carts = pd.read_parquet('local_data/abandoned_carts.parquet', engine='pyarrow')
df_orders = pd.read_parquet('local_data/orders.parquet', engine='pyarrow')
df_regulars = pd.read_parquet('local_data/regulars.parquet', engine='pyarrow')
df_users = pd.read_parquet('local_data/users.parquet', engine='pyarrow')

def combinar_datasets_totales():
    # Paso 1: Crear DataFrame de pedidos
    exploded_items = df_orders['ordered_items'].explode()  
    item_counts = exploded_items.value_counts()
    item_counts_df_orders = item_counts.reset_index()
    item_counts_df_orders.columns = ['product_id', 'number_of_orders']
    total_orders = len(df_orders)
    item_counts_df_orders['purchase_probability'] = item_counts_df_orders['number_of_orders'] / total_orders
    print("primeros cinco df orders: ", item_counts_df_orders.head())

    # Paso 2: Crear DataFrame de carritos abandonados
    exploded_items_abandoned = df_abandoned_carts['variant_id'].explode()
    item_counts_abandoned = exploded_items_abandoned.value_counts()
    item_counts_df_abandoned = item_counts_abandoned.reset_index()
    item_counts_df_abandoned.columns = ['product_id', 'number_of_abandoned']
    total_abandoned_carts = len(df_abandoned_carts)
    item_counts_df_abandoned['abandon_probability'] = item_counts_df_abandoned['number_of_abandoned'] / total_abandoned_carts
    print("primeros cinco df abandoned: ", item_counts_df_abandoned.head())

    # Paso 3: Filtrar para incluir solo productos en inventario
    df_orders_inventory = item_counts_df_orders[item_counts_df_orders["product_id"].isin(df_inventory["variant_id"])]
    df_abandoned_inventory = item_counts_df_abandoned[item_counts_df_abandoned["product_id"].isin(df_inventory["variant_id"])]

    # Paso 4: Combinar los DataFrames
    df_combined = df_orders_inventory.merge(df_abandoned_inventory, on="product_id", how="outer")

    # Rellenar valores NaN con 0, ya que algunos productos pueden no estar en ambos DataFrames
    df_combined.fillna(0, inplace=True)

    print(df_combined.head())
    print("Longitud del dataset combinado: ", len(df_combined), " y longitud del inventario: ", len(df_inventory))
    return df_combined

# Uso de la función:
# df_combinado_total = combinar_datasets_totales(df_orders, df_abandoned_carts, df_inventory)
# df_combinado_total.head()



    

            



def main():
   # mean_price()
   #order_dates(5)
   #crear_df_from_orders_as_productid_n_orders_purch_prob()
   #print(len(df_inventory))
   #print(len(df_orders))
   #combinar_datasets_prod_pedidos_inventario()
   combinar_datasets_totales()
   


if __name__ == "__main__":
    main()
