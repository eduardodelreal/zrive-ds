import boto3
import os

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



def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
