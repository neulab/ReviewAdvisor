# sh prepare.sh results/review.test gold_sent.txt gold_id.txt
python helper/jsonlize.py "$1" out.jsonl
python helper/split.py out.jsonl 1 test.txt id.txt
rm out.jsonl