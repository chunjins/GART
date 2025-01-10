dataset="mpi"
profile="mpi_3m"
logbase=${profile}

for seq in 'Antonia' 'Magdalena' '0056' 'FranziRed'
do
    python solver.py --profile ./profiles/custom/custom_3m.yaml --dataset $dataset --seq $seq --logbase $logbase --fast --no_eval
done

# data_type = 'novel_pose' in test_func
for seq in 'Antonia' 'Magdalena' '0056' 'FranziRed'
do
    python solver.py --profile ./profiles/custom/custom_3m.yaml --dataset $dataset --seq $seq --eval_only --log_dir logs/${logbase}/seq=${seq}_prof=custom_3m_data=${dataset}
done

