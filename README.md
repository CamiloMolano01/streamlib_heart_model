## Pre requirements

- Have [Git](https://git-scm.com/) installed.
- Have [Miniconda](https://docs.anaconda.com/miniconda/) installed (Miniconda is a minimal installer for conda).

1. (Opc1) Instala los paquetes necesarios usando miniconda:
    ```bash
    cd streamlit_heart_model
    conda env create -f environment.yml
    conda activate streamlit-heart-env
    ```

2. (Opc2) Instala los paquetes usando pip en tu ambiente:
    ```bash
    cd streamlit_heart_model
    pip install requirements.txt
    ```