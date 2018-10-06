# Multitask disfluency detection
Code for the paper "Multi-Task Learning for Domain-General Spoken Disfluency Detection in Dialogue Systems" ([Igor Shalyminov](https://github.com/ishalyminov), [Arash Eshghi](https://github.com/araesh), and [Oliver Lemon](https://github.com/olemon1))

bAbI+ disfluency study data generation
==
1. Get https://github.com/ishalyminov/babi_tools and install requirements
2. Download [bAbI dialog tasks](https://research.fb.com/downloads/babi/) into the `babi_tools` folder
2. Run `sh make_generalization_study_datasets.sh <RESULT_FOLDER>`
3. Run `sh tag_dataset.sh <RESULT_FOLDER> <config_file_name>` for every config in `2018_generalization_study_configs`
4. The resulting datasets are `<RESULT_FOLDER>/<BABI_DATASET_NAME>/*.tagged.json`