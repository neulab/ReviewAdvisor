# sh jsonl.sh gold_sent.txt gold_id.txt gold.jsonl
rm cached_test_BertTokenizer_512 cached_test_BertTokenizer_512.lock
mv seqlab_final/test_predictions.txt ./test.txt
python helper/heuristics.py id.txt test.txt result.jsonl
rm test.txt id.txt