# LLM-drug-discovery
Automating Biomedical Literature Review for Rapid Drug Discovery: Leveraging GPT-4 to Expedite Pandemic Response.

## Introduction
This repository contains the code and resources for our research project aimed at curtailing the time and resources traditionally associated with manual reviews in drug discovery and repurposing. Our automated framework identifies papers with potential drug targets, aiding researchers in maintaining an up-to-date map of current readiness for priority pathogens.

## Getting Started

### Setting Up the Virtual Environment
To set up the virtual environment and install the required libraries, execute the following commands:

```bash
conda env create -f environment.yml

conda activate drug
```
### OpenAI API Key
Before using the scripts in this repository, you need to obtain an API key from OpenAI and replace your API key in the scripts.

**Note**: Keep your API key secure and do not share it publicly.


## Usage

This section provides detailed instructions on how to run the scripts in this repository.

### Data Preparation

#### Step 1: Data Collection

```bash
python data_preparation/PubMed_abstract_extraction.py
```

#### Step 2: Explanation Generation

```bash
python data_preparation/concatenate_sections.py

python data_preparation/generate_explanation.py
```

#### Step 3: Embedding Fetching

```bash
python data_preparation/generate_embedding.py
```

#### Step 4: Data Split

```bash
python data_preparation/run_cross_validation.py

python data_preparation/run_cross_validation_embeddings.py
```

### Query Response

#### Zero-Shot

```bash
python model/run_zero.py --cot 0
```

#### Zero-Shot-CoT

```bash
python model/run_zero.py --cot 1
```

#### Few-Shot

```bash
python model/run_few_random.py --cot 0
```

#### Few-Shot-CoT

```bash
python model/run_few_random.py --cot 1
```

#### Similar-Shot

```bash
python model/run_few_similar.py --cot 0
```

#### Similar-Shot-CoT

```bash
python model/run_few_similar.py --cot 1
```

### Authors or Acknowledgments

**Title**: Automating Biomedical Literature Review for Rapid Drug Discovery: Leveraging GPT-4 to Expedite Pandemic Response

**Authors**:
- Jingmei Yang
- Kenji C. Walker
- Ayse A. Bekar-Cesaretli
- Boran Hao
- Nahid Bhadelia, M.D.
- Diane Joseph-McCarthy, Ph.D.
- Ioannis Ch. Paschalidis, Ph.D

## Cite US
```@article{yang2024automating,
  title={Automating biomedical literature review for rapid drug discovery: Leveraging GPT-4 to expedite pandemic response},
  author={Yang, Jingmei and Walker, Kenji C and Bekar-Cesaretli, Ayse A and Hao, Boran and Bhadelia, Nahid and Joseph-McCarthy, Diane and Paschalidis, Ioannis Ch},
  journal={International Journal of Medical Informatics},
  pages={105500},
  year={2024},
  publisher={Elsevier}
}```


## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).


## Contact Information

- **Email**: jmyang@bu.edu.





