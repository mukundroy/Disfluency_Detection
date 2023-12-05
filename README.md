# CRF based Disfluency_Detection

## External Libraries
a. NLTK toolkit
b. sklearn_crfsuite

## How to run:
1. POS Tag the Testdata using
   ```bash
   python postag_testdata.py <lang e.g hindi, bangla, marathi> <input-file>
   ```
2. Execute
   ```bash
    python test_disfluency.py <model-path> <test-file-path>
   ```

## How to train:
1. POS Tag the Training Data using
   ```bash
   python postag_data.py <lang e.g hindi, bangla, marathi> <input-file>
   ```
2. For Training, use:
   ```bash
   python train_disfluency.py <ltrain-file-path> <modelname-to-save>
   ```
3. Use the Dev file to print the evaluate the model performance using:
   ```bash
   python dev_test_disfluency.py <dev-file-path> <model-file-path>
   ```
   
