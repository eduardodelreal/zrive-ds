import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np





#first of all we do a little of analysis, as we open all the different datasets

df_inventory = pd.read_parquet('local_data/inventory.parquet', engine='pyarrow')
df_abandoned_carts = pd.read_parquet('local_data/abandoned_carts.parquet', engine='pyarrow')
df_orders = pd.read_parquet('local_data/orders.parquet', engine='pyarrow')
df_regulars = pd.read_parquet('local_data/regulars.parquet', engine='pyarrow')
df_users = pd.read_parquet('local_data/users.parquet', engine='pyarrow')

def combinar_datasets_totales(): #to see what products are the most purchased and to see the most abandoned
    # Paso 1: Crear DataFrame de pedidos
    exploded_items = df_orders['ordered_items'].explode()  
    item_counts = exploded_items.value_counts()
    item_counts_df_orders = item_counts.reset_index()
    item_counts_df_orders.columns = ['product_id', 'number_of_orders']
    total_orders = len(df_orders)
    item_counts_df_orders['purchase_probability'] = item_counts_df_orders['number_of_orders'] / total_orders
    #print("primeros cinco df orders: ", item_counts_df_orders.head())

    # Paso 2: Crear DataFrame de carritos abandonados
    exploded_items_abandoned = df_abandoned_carts['variant_id'].explode()
    item_counts_abandoned = exploded_items_abandoned.value_counts()
    item_counts_df_abandoned = item_counts_abandoned.reset_index()
    item_counts_df_abandoned.columns = ['product_id', 'number_of_abandoned']
    total_abandoned_carts = len(df_abandoned_carts)
    item_counts_df_abandoned['abandon_probability'] = item_counts_df_abandoned['number_of_abandoned'] / total_abandoned_carts
    #print("primeros cinco df abandoned: ", item_counts_df_abandoned.head())

    # Paso 3: Filtrar para incluir solo productos en inventario
    df_orders_inventory = item_counts_df_orders[item_counts_df_orders["product_id"].isin(df_inventory["variant_id"])]
    df_abandoned_inventory = item_counts_df_abandoned[item_counts_df_abandoned["product_id"].isin(df_inventory["variant_id"])]

    # Paso 4: Combinar los DataFrames
    df_combined = df_orders_inventory.merge(df_abandoned_inventory, on="product_id", how="outer")

    # Rellenar valores NaN con 0, ya que algunos productos pueden no estar en ambos DataFrames
    df_combined.fillna(0, inplace=True)

    #print(df_combined.head())
    #print("Longitud del dataset combinado: ", len(df_combined), " y longitud del inventario: ", len(df_inventory))
    
    
    # Configuración de la visualización
    
    plt.figure(figsize=(20, 10))

    # Top 10 productos por frecuencia de compra
    plt.subplot(2, 2, 1)
    sns.barplot(x='product_id', y='number_of_orders', data=df_combined.sort_values('number_of_orders', ascending=False).head(10))
    plt.title('Top 10 Productos por Frecuencia de Compra')
    plt.xticks(rotation=45)

    # Top 10 productos por probabilidad de compra
    plt.subplot(2, 2, 2)
    sns.barplot(x='product_id', y='purchase_probability', data=df_combined.sort_values('purchase_probability', ascending=False).head(10))
    plt.title('Top 10 Productos por Probabilidad de Compra')
    plt.xticks(rotation=45)

    # Top 10 productos por frecuencia de abandono
    plt.subplot(2, 2, 3)
    sns.barplot(x='product_id', y='number_of_abandoned', data=df_combined.sort_values('number_of_abandoned', ascending=False).head(10))
    plt.title('Top 10 Productos por Frecuencia de Abandono')
    plt.xticks(rotation=45)

    # Top 10 productos por probabilidad de abandono
    plt.subplot(2, 2, 4)
    sns.barplot(x='product_id', y='abandon_probability', data=df_combined.sort_values('abandon_probability', ascending=False).head(10))
    plt.title('Top 10 Productos por Probabilidad de Abandono')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    print("Guardando el gráfico...")
    plt.savefig('/home/edu/zrive-ds-1/zrive-ds/src/module_2/graphs/graph_1.png')
    print("Gráfico guardado.")

    plt.show()
    

    # Cierra la figura para liberar memoria
    plt.close()
    return df_combined



'''
one of the things we expect at first in our bussiness is that if price is reduced, customers
will be more attracted to that product than before and therefor will have a bigger purchased probability.
Let's consider a significat discount those bigger than the 20%. If there's a "significant discount", then 
we believe there'll be more purchase probability than if there was less %
'''

def probability_per_discount_check(percentage):
    data=df_inventory
    menos=[]
    mas=[]
    for index, row in data.iterrows():
        if row['compare_at_price'] == 0:
            # Opción 1: Continuar con la siguiente iteración
            continue

            # Opción 2: Establecer el porcentaje de descuento en cero o algún valor predeterminado
            # porcentaje_descuento = 0
        else:
            descuento = row['compare_at_price'] - row['price']
            porcentaje_descuento = (descuento / row['compare_at_price']) * 100

            # Comparar el porcentaje de descuento
            if porcentaje_descuento > percentage:
                mas.append(row['variant_id'])
            else:
                menos.append(row['variant_id'])
    
    # Crear subdataframes
    df_mas = data[data['variant_id'].isin(mas)]
    df_menos = data[data['variant_id'].isin(menos)]
    
    # Calcular la probabilidad de ser comprado
    total_orders = df_orders['ordered_items'].explode().value_counts()
    df_mas['purchase_probability'] = df_mas['variant_id'].apply(lambda x: total_orders.get(x, 0) / len(df_orders))
    df_menos['purchase_probability'] = df_menos['variant_id'].apply(lambda x: total_orders.get(x, 0) / len(df_orders))
    print(df_mas.head(),df_menos.head())
    return df_mas, df_menos
    
    '''
    But we can see how this has nothing to do with the disccount we make on the products unless we 
    have a huge disccount that catches the attention of our client. So maybe we should ask ourselfs that for our
    online shop we should focus more on some specific target products. Which ones? let's see
    '''
    
    
def contar_pedidos_por_tipo(): #nos da un graph e info sobre los product_type mas demandados
    variant_info = df_inventory.set_index('variant_id')[['product_type', 'price']]

    # Filtrar variant_id en df_orders y df_abandoned_carts que están en df_inventory
    valid_variants = set(df_inventory['variant_id'])
    ordered_variants = df_orders['ordered_items'].explode().map(lambda x: x if x in valid_variants else None).dropna()
    abandoned_variants = df_abandoned_carts['variant_id'].explode().map(lambda x: x if x in valid_variants else None).dropna()
    
    # Asignar product_type y price a cada variant_id en los pedidos y en los abandonos
    ordered_variants_info = variant_info.loc[ordered_variants]
    abandoned_variants_info = variant_info.loc[abandoned_variants]

    # Calcular ingresos por variant_id
    ordered_variants_info['revenue'] = ordered_variants_info['price']
    abandoned_variants_info['lost_revenue'] = abandoned_variants_info['price']

    # Agregar y agrupar por product_type
    revenue_per_type = ordered_variants_info.groupby('product_type')['revenue'].sum()
    lost_revenue_per_type = abandoned_variants_info.groupby('product_type')['lost_revenue'].sum()

    # Calcular ingresos netos por product_type
    net_revenue_per_type = revenue_per_type - lost_revenue_per_type

    # Contar pedidos y abandonos por product_type
    order_counts = ordered_variants_info['product_type'].value_counts()
    abandoned_counts = abandoned_variants_info['product_type'].value_counts()

    # Crear un nuevo DataFrame
    top_types = pd.DataFrame({
        'order_count': order_counts,
        'abandoned_count': abandoned_counts.reindex(order_counts.index, fill_value=0),
        'net_revenue': net_revenue_per_type.reindex(order_counts.index, fill_value=0)
    }).reset_index().rename(columns={'index': 'product_type'})

    # Ordenar por ingresos netos (net_revenue) de mayor a menor
    top_types = top_types.sort_values(by='net_revenue', ascending=False)
    
    top_types = top_types.sort_values(by='net_revenue', ascending=False)
    top_10_types = top_types.head(10)
    
    #print(top_10_types)

    # Crear un gráfico de barras para el ingreso neto de las 10 principales categorías
    plt.figure(figsize=(10, 6))
    plt.bar(top_10_types['product_type'], top_10_types['net_revenue'], color='skyblue')
    plt.xlabel('Product Type')
    plt.ylabel('Net Revenue')
    plt.title('Top 10 Product Types by Net Revenue')
    plt.xticks(rotation=45)
    plt.show()

    return top_types


'''
We can see that clearly long-life-milk-substitudes is the product_type with a highest revenue. 
This will be very useful since we now can focus more specificaly on this and others really high reveneu product
types. I believe that regular users do buy a really high percentage on products belonging to the first 
five highest revenues ones. Let's check. 
'''

def best_users_product_type():
    # Mapear variant_id a product_type
    variant_info = df_inventory.set_index('variant_id')[['product_type', 'price']].to_dict('index')

    # Expandir los items ordenados y abandonados en df_orders y df_abandoned_carts
    expanded_orders = df_orders.explode('ordered_items')
    expanded_abandoned = df_abandoned_carts.explode('variant_id')

    # Filtrar variant_id en df_orders y df_abandoned_carts que están en df_inventory
    valid_variants = set(df_inventory['variant_id'])
    expanded_orders = expanded_orders[expanded_orders['ordered_items'].isin(valid_variants)]
    expanded_abandoned = expanded_abandoned[expanded_abandoned['variant_id'].isin(valid_variants)]

    # Asignar product_type y price a cada variant_id en los pedidos y abandonos
    expanded_orders['product_type'] = expanded_orders['ordered_items'].map(lambda x: variant_info[x]['product_type'])
    expanded_orders['price'] = expanded_orders['ordered_items'].map(lambda x: variant_info[x]['price'])
    expanded_abandoned['product_type'] = expanded_abandoned['variant_id'].map(lambda x: variant_info[x]['product_type'])

    # Función para manejar grupos vacíos al encontrar el product_type más común
    def most_common_product_type(x):
        if len(x) > 0:
            return x.value_counts().idxmax()
        else:
            return None

    # Contar pedidos, abandonos y encontrar product_type más común por user_id
    user_order_counts = expanded_orders['user_id'].value_counts()
    user_abandoned_counts = expanded_abandoned['user_id'].value_counts()
    most_bought_product_type = expanded_orders.groupby('user_id')['product_type'].agg(most_common_product_type)
    most_abandoned_product_type = expanded_abandoned.groupby('user_id')['product_type'].agg(most_common_product_type)

    # Calcular el revenue por usuario
    revenue_per_user = expanded_orders.groupby('user_id')['price'].sum()

   # Crear un nuevo DataFrame
    best_users = pd.DataFrame({
        'user_id': user_order_counts.index,
        'order_count': user_order_counts,
        'most_bought_product_type': most_bought_product_type,
        'abandoned_count': user_abandoned_counts.reindex(user_order_counts.index, fill_value=0),
        'most_abandoned_product_type': most_abandoned_product_type.reindex(user_order_counts.index),
        'revenue': revenue_per_user
    }).reset_index(drop=True)

    # Ordenar el DataFrame por 'revenue' en orden descendente
    best_users = best_users.sort_values(by='revenue', ascending=False)

    print(best_users.head())
    return best_users


'''
okay, now we have a dataset with every user with information according their revenue, their total orders, their total
abandons, their fav product type...
'''
'''
#now, let's check what would be the bestest vendors:

def vendors_with_biggest_revenue():
    variant_info = df_inventory.set_index('variant_id')[['vendor', 'price']].to_dict('index')

    # Expandir los items ordenados y abandonados
    expanded_orders = df_orders.explode('ordered_items')
    expanded_abandoned = df_abandoned_carts.explode('variant_id')

    # Filtrar variant_id en df_orders y df_abandoned_carts que están en df_inventory
    valid_variants = set(df_inventory['variant_id'])
    expanded_orders = expanded_orders[expanded_orders['ordered_items'].isin(valid_variants)]
    expanded_abandoned = expanded_abandoned[expanded_abandoned['variant_id'].isin(valid_variants)]

    # Asignar vendor y price a cada variant_id en los pedidos y en los abandonos
    expanded_orders['vendor'] = expanded_orders['ordered_items'].apply(lambda x: variant_info.get(x, {}).get('vendor', 'Unknown'))
    expanded_orders['revenue'] = expanded_orders['ordered_items'].apply(lambda x: variant_info.get(x, {}).get('price', 0))
    abandoned_variants_info = variant_info.loc[expanded_abandoned['variant_id']]

    # Calcular ingresos y perdidas por variant_id
    abandoned_variants_info['lost_revenue'] = abandoned_variants_info['price']

    # Agregar y agrupar por vendor
    revenue_per_vendor = expanded_orders.groupby('vendor')['revenue'].sum()
    lost_revenue_per_vendor = abandoned_variants_info.groupby('vendor')['lost_revenue'].sum()

    # Calcular ingresos netos por vendor
    net_revenue_per_vendor = revenue_per_vendor - lost_revenue_per_vendor

    # Crear un nuevo DataFrame y ordenarlo por ingresos netos en orden ascendente
    top_vendors = pd.DataFrame({
        'vendor': net_revenue_per_vendor.index,
        'net_revenue': net_revenue_per_vendor.values
    })
    top_vendors = top_vendors.sort_values(by='net_revenue', ascending=True)  # Orden ascendente

    # Crear un gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.barh(top_vendors['vendor'], top_vendors['net_revenue'], color='skyblue')
    plt.xlabel('Net Revenue')
    plt.ylabel('Vendor')
    plt.title('Vendors with Biggest Revenue (Ascending)')
    plt.xticks(rotation=45)
    plt.show()

    return top_vendors


'''


def test_combinar_datasets_totales():
    # Aquí, suponemos que los DataFrames ya están cargados o los cargamos dentro de esta función.
    # También, asegúrate de que la función 'combinar_datasets_totales' devuelva 'df_combined'.

    df_combined = combinar_datasets_totales()

    # Verificar que el resultado es un DataFrame.
    assert isinstance(df_combined, pd.DataFrame), "El resultado debe ser un DataFrame."

    # Comprobar que las columnas esperadas están presentes.
    expected_columns = ['product_id', 'number_of_orders', 'purchase_probability', 'number_of_abandoned', 'abandon_probability']
    for column in expected_columns:
        assert column in df_combined.columns, f"Falta la columna esperada: {column}"

    # Verificar que no hay valores NaN inesperados.
    assert df_combined.notna().all().all(), "No deben existir valores NaN inesperados."

    # Opcional: Comprobar el tamaño del DataFrame resultante.
    # assert len(df_combined) > 0, "El DataFrame combinado no debe estar vacío."

    print("Todas las pruebas pasaron correctamente.")
    
    
def test_best_users_product_type():
    # Crear datos de prueba
    mock_inventory_data = {
        'variant_id': [1, 2],
        'product_type': ['type1', 'type2'],
        'price': [10.0, 20.0]
    }
    mock_orders_data = {
        'user_id': ['user1', 'user2', 'user1'],
        'ordered_items': [[1], [2], [1]]
    }
    mock_abandoned_carts_data = {
        'user_id': ['user1', 'user2'],
        'variant_id': [1, 2]
    }

    df_inventory = pd.DataFrame(mock_inventory_data)
    df_orders = pd.DataFrame(mock_orders_data)
    df_abandoned_carts = pd.DataFrame(mock_abandoned_carts_data)

    # Ejecutar la función best_users_product_type
    best_users = best_users_product_type()
    

    # Comprobaciones de la prueba
    assert isinstance(best_users, pd.DataFrame), "El resultado debe ser un DataFrame."
    expected_columns = ['user_id', 'order_count', 'most_bought_product_type', 'abandoned_count', 'most_abandoned_product_type', 'revenue']
    for column in expected_columns:
        assert column in best_users.columns, f"Falta la columna esperada: {column}"
 
    
    print("Todas las pruebas pasaron correctamente.")



def test_contar_pedidos_por_tipo():
    # Crear datos de prueba
    mock_inventory_data = {
        'variant_id': [1, 2, 3],
        'product_type': ['type1', 'type2', 'type3'],
        'price': [10.0, 20.0, 15.0]
    }
    mock_orders_data = {
        'user_id': ['user1', 'user2', 'user1'],
        'ordered_items': [[1, 2], [2], [3]]
    }
    mock_abandoned_carts_data = {
        'user_id': ['user1', 'user2'],
        'variant_id': [3, 1]
    }

    df_inventory = pd.DataFrame(mock_inventory_data)
    df_orders = pd.DataFrame(mock_orders_data)
    df_abandoned_carts = pd.DataFrame(mock_abandoned_carts_data)

    # Ejecutar la función contar_pedidos_por_tipo
    top_types = contar_pedidos_por_tipo()

    # Comprobaciones de la prueba
    assert isinstance(top_types, pd.DataFrame), "El resultado debe ser un DataFrame."
    expected_columns = ['product_type', 'order_count', 'abandoned_count', 'net_revenue']
    for column in expected_columns:
        assert column in top_types.columns, f"Falta la columna esperada: {column}"
    assert all(column in top_types.columns for column in expected_columns), "El DataFrame no tiene las columnas esperadas."
    
    print("Todas las pruebas pasaron correctamente.")

def main():
   # mean_price()
   #order_dates(5)
   #crear_df_from_orders_as_productid_n_orders_purch_prob()
   #print(len(df_inventory))
   #print(len(df_orders))
   #combinar_datasets_prod_pedidos_inventario()
   combinar_datasets_totales()
   #test_combinar_datasets_totales()
   #test_plot_combined_dataset()
   #plot_combined_dataset()
   #hours_vs_orders_plot(df_abandoned_carts,'created_at')
   #probability_per_discount_check(20)
   #contar_pedidos_por_tipo()
   #best_users_product_type()
   #test_best_users_product_type()
   #test_contar_pedidos_por_tipo()
   #vendors_with_biggest_revenue()
   
   #print(site.getsitepackages())

   


if __name__ == "__main__":
    main()
