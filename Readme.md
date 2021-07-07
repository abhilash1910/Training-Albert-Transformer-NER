# Albert-NER-Conll-Notebook



<img src="https://huggingface.co/front/assets/huggingface_logo.svg">



This Notebook contains set of instructions how to train [Albert](https://huggingface.co/transformers/model_doc/albert.html) from Huggingface in Google Colab.
Training is done on the [Conllu](https://drive.google.com/uc?export=download&id=1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P) dataset. The model can be accessed via [HuggingFace](https://huggingface.co/abhilash1910/albert-german-ner):


```python

from transformers import AutoTokenizer,TFAutoModelForTokenClassification
from transformers import pipeline

model=TFAutoModelForTokenClassification.from_pretrained('abhilash1910/albert-german-ner')
tokenizer=AutoTokenizer.from_pretrained('abhilash1910/albert-german-ner')
ner_model = pipeline('ner', model=model, tokenizer=tokenizer)
seq='Berlin ist die Hauptstadt von Deutschland'
ner_model(seq)
```

The result is:

```bash
[{'entity': 'B-PERderiv',
  'index': 1,
  'score': 0.09580112248659134,
  'word': '▁berlin'},
 {'entity': 'B-ORGpart',
  'index': 2,
  'score': 0.08364498615264893,
  'word': '▁is'},
 {'entity': 'B-LOCderiv',
  'index': 3,
  'score': 0.07593920826911926,
  'word': 't'},
 {'entity': 'B-PERderiv',
  'index': 4,
  'score': 0.09574996680021286,
  'word': '▁die'},
 {'entity': 'B-LOCderiv',
  'index': 5,
  'score': 0.07097965478897095,
  'word': '▁'},
 {'entity': 'B-PERderiv',
  'index': 6,
  'score': 0.07122448086738586,
  'word': 'haupt'},
 {'entity': 'B-PERderiv',
  'index': 7,
  'score': 0.12397754937410355,
  'word': 'stadt'},
 {'entity': 'I-OTHderiv',
  'index': 8,
  'score': 0.0818650871515274,
  'word': '▁von'},
 {'entity': 'I-LOCderiv',
  'index': 9,
  'score': 0.08271490037441254,
  'word': '▁'},
 {'entity': 'B-LOCderiv',
  'index': 10,
  'score': 0.08616268634796143,
  'word': 'deutschland'}]
 ```


## ALBERT

The ALBERT model was proposed in [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. It presents two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:

- Splitting the embedding matrix into two smaller matrices.

- Using repeating layers split among groups.

The abstract from the paper is the following:

Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations, longer training times, and unexpected model degradation. To address these problems, we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.

Tips:

ALBERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.

ALBERT uses repeating layers which results in a small memory footprint, however the computational cost remains similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same number of (repeating) layers.

