# Emotional Neural Interface

Este es un repositorio se encuentra un proyecto el cual con señales EEG en formato CSV y por medio de Deep Learning se intenta codificar las señales para poder clasificarlas por sus el estímulo que genera.

## Instalación

Se tienen dos tipos de instalación, la instalación mediante [anaconda](https://www.anaconda.com/) y estando en el prompt se ejecuta el siguiente comando

```bash
conda env create --name envname --file=neuralinterface.yml
```

## Uso
Para el uso del programa en general se tiene que editar algunos parámetros del archivo config.yaml
Lo principal es configurar el path de los datos con los que se quiere trabajar y el directorio donde se quiere guardar los resultados del entrenamiento del modelo.

```bash
python workbench.py
```

## Contributing


## License
[MIT](https://choosealicense.com/licenses/mit/)