SANAL ORTAM KURULUMU

MAC:

cd Feature_Ranking_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

WINDOWS:

winget install --id Python.Python.3.13 --exact --version 3.13.5
cd Feature_Ranking_Project
py -3.13 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

KONTROL:

python --version
python -c "import numpy, pandas, sklearn, scipy, tensorflow, keras; print(numpy.__version__, pandas.__version__, sklearn.__version__, scipy.__version__, tensorflow.__version__, keras.__version__)"

Beklenen ana sürümler:

Python 3.13.5
numpy 2.4.4
pandas 3.0.2
scikit-learn 1.8.0
scipy 1.17.1
tensorflow 2.21.0
keras 3.13.2
