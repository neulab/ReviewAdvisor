# Aspect Tagger
We define 8 aspects which are **Summary**, **Motivation/Impact**, **Originality**, **Soundness/Correctness**, **Substance**, **Replicability**, **Meaningful Comparison**, **Clarity**. Our tagger can tag appropriate spans that indicate those aspects with sentiment polarity (e.g. positive originality). Requirements for all the libraries are in the **requirements.txt**, please use the right version for each library in order to use our trained tagger.

<br>

## Batch Annotation
For batch annotation, please follow the format shown in **sample.txt** to prepare your data. Specifically, one line should be one review. Batch annotation support both CPU and GPU, but we highly suggest using GPU for efficiency reasons.

As an example, to prepare the proper input data for tagger, run
```bash
sh prepare.sh sample.txt
```

This will result in a **test.txt** file and a **id.txt** file, which will be used to feed into our tagger as well as for later alignment use.

To tag the prepared file **test.txt**, run
```bash
python run_tagger.py config.json
```

This will write the results in **seqlab_final/test_predictions.txt**. To further clean the results and apply our heuristic rules, run 
```bash
sh post_process.sh
```

The results will be written into **result.jsonl**, one line for each review.

<br>

## Direct Annotation
Except batch annotation, we also provide an interface to conveniently annotate a single review. See the usage below.

```python
from annotator import Annotator
annotator = Annotator('labels.txt', 'seqlab_final', 'cpu')  # The last argument can be 'cpu' or 'gpu'.
annotator.annotate('The paper is well written and easy to follow.') # the input is plain text.
```

You should be able to get the following output
```python
>>> annotator.annotate('The paper is well written and easy to follow.')
[('The', 'clarity_positive'), ('paper', 'clarity_positive'), ('is', 'clarity_positive'), ('well', 'clarity_positive'), ('written', 'clarity_positive'), ('and', 'clarity_positive'), ('easy', 'clarity_positive'), ('to', 'clarity_positive'), ('follow', 'clarity_positive'), ('.', 'clarity_positive')]

```

