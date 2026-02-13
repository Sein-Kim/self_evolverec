## Self-EvolveRec: Self-Evolving Recommender Systems with LLM-based Directional Feedback

This repository is designed for implementing Self-EvolveRec.


## Run Evolve
To evolve a recommender, place your model code into the `examples` folder. We have  included initial examples for the SASRec on Amazon CDs dataset.

Before start the evolution, you must set your OpenAI API key as an environment variable.
```
export OPENAI_API_KEY="your_openai_api_key"
```

To start the evolution for SASRec on the CDs dataset, run
```
python self_evolverec.py problem="sasrec_cds"
```


## Testing Evolved Models
We provide evolved models (SASRec on CDs dataset) that have already undergone the directional feedback loop in the `get_recommendation` folder.

To run the evaluation and obtain the test scores for an evolved SASRec model (For this evaluation, the User Simulator and Model Diagnosis Tool have been omitted from the script to provide performance metrics without requiring OpenAI API usage.):
```
cd get_recommendation/evolved_sasrec_cd
python main_code.py
```

## Environment Settings
Our experiments were conducted using Python 3.9.21 .

```
pip install -r requirements.txt
```

Our environment configuration is closely aligned with established LLM-driven evolution baselines. For more detailed context or troubleshooting regarding the dependency structure, you may also refer to the environment setups of [DeepEvolve](https://github.com/liugangcode/deepevolve) and [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) (the open-source implementation of AlphaEvolve).

## Acknowledgemets
Self-EvolveRec is built upon various open-source projects, and we are deeply grateful for their invaluable contributions.
- [DeepEvolve](https://github.com/liugangcode/deepevolve)
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)