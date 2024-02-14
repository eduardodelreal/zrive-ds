#import pandas as pd
#import datetime
#import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np

import matplotlib.pyplot as plt

def graph(x, y):
    # Crear gráfico de línea
    plt.plot(x, y)

    # Añadir títulos y etiquetas
    plt.title("Gráfico de Línea Simple")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")

    # Mostrar el gráfico
    plt.show()

# Ejemplo de uso
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]






    


#para el plot; focus on :
#Top 10 productos por frecuencia de compra.
#Top 10 productos por probabilidad de compra.
#Top 10 productos por frecuencia de abandono.
#Top 10 productos por probabilidad de abandono.

'''
def plot_combined_dataset():
    
    df_combined=combinar_datasets_totales()
    
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
    plt.show()

'''

    

            



def main():
   # mean_price()
   #order_dates(5)
   #crear_df_from_orders_as_productid_n_orders_purch_prob()
   #print(len(df_inventory))
   #print(len(df_orders))
   #combinar_datasets_prod_pedidos_inventario()
   #combinar_datasets_totales()
   #plot_combined_dataset()
   #hours_vs_orders_plot(df_abandoned_carts,'created_at')
   graph(x, y)
   


if __name__ == "__main__":
    main()
