## Mixed Tracker Manual Evaluation Dataset
negative.tsv and positive.tsv contain the negative and positive predictions by Duumviri, along with our manual labels. Each file contains five columns:
- Site: The URL of the site containing the tracker.
- Request: The analyzed request.
- Request Field: The analyzed request field.
- Request Field Value: The value of the request field.
- Manual Label: Our assigned label. Since the values in this column were written by humans, they may not always be clear. We explain the content below.

An example loading script is provided in load.py.

## Manual Label Explanation
In this section, we explain the labels in positive.tsv and negative.tsv separately. The labels roughly correpond to Table XII of the submission. 

### Labels in positive.tsv
We used the following command to obtain all possible values of the manual label with their counts from positive.tsv. 

```bash
>>> df_p['Manual Label'].value_counts()
Manual Label
tracker               26
stale                  6
breakage               4
undecided              3
potential breakage     1
```

#### How to Verify the Results
The results from positive.tsv correspond to Table XIII of the original paper submission.

### Labels in negative.tsv
```bash
>>> df_n['Manual Label'].value_counts()
Manual Label
breakage     18
stale        14
undecided     6
tracker       2
```

#### How to Verify the Results
These results from negative.tsv correspond to Table XIV of the original submission.

### Accuracy Calculation
Please refer to the second paragraph of the Negative Case Analysis in Section V.B.3, which is located just above Table XIV on page 11 of the original submission.
