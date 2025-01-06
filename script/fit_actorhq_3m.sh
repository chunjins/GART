dataset="actorhq"
profile="actorhq_3m"
logbase=${profile}

for seq in 'actor0101' 'actor0301' 'actor0601' 'actor0701'
do
    python solver.py --profile ./profiles/custom/custom_3m.yaml --dataset $dataset --seq $seq --logbase $logbase --fast --no_eval
done