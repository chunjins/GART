dataset="actorhq"
profile="actorhq_3m"
logbase=${profile}

#for seq in 'actor0101' 'actor0301' 'actor0601' 'actor0701'
for seq in 'actor0301' 'actor0601' 'actor0701'
do
    python solver.py --profile ./profiles/custom/custom_3m.yaml --dataset $dataset --seq $seq --eval_only --log_dir logs/${logbase}/seq=${seq}_prof=custom_3m_data=${dataset}
done