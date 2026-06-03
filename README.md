## Pre requirements

- Have [Git](https://git-scm.com/) installed.
- Have [Miniconda](https://docs.anaconda.com/miniconda/) installed (Miniconda is a minimal installer for conda).

1. (Opc1) Instala los paquetes necesarios usando miniconda:
    ```bash
    cd streamlit_heart_model
    conda env create -f environment.yml
    conda activate streamlit-heart-env
    ```

2. (Opc2) Instala los paquetes usando pip en tu ambiente *(Python >= 3.11)*:
    ```bash
    cd streamlit_heart_model
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    ```

3. Ejecutar servidor local:
    ```bash
   streamlit run heart_pred_stream.py
   ```