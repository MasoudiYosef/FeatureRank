CNN original
python scripts/run_train_cnn.py --mode original --dataset-name breast_cancer_data.csv --target-column diagnosis --id-column none --feature-percent 20

CNN filtered
python scripts/run_train_cnn.py --mode filtered --dataset-name breast_cancer_data.csv --target-column target --id-column none --feature-percent 20 --ranking-type avg


Autoencoder Filtered :

    Encoder max filtered için:

    python scripts/run_train_autoencoder.py --mode filtered --dataset-name breast_cancer_data.csv --target-column diagnosis --id-column none --feature-percent 20 --ranking-source encoder --ranking-type max

    Decoder max filtered için:

    python scripts/run_train_autoencoder.py --mode filtered --dataset-name breast_cancer_data.csv --target-column diagnosis --id-column none --feature-percent 20 --ranking-source decoder --ranking-type max