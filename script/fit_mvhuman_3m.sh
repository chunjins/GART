dataset="mvhuman"
profile="mvhuman_3m"
logbase=${profile}

#for seq in '100846' '100990' '102107' '102145' '103708' '200173' '204112' '204129'
for seq in '102107'
do
    python solver.py --profile ./profiles/custom/custom_3m.yaml --dataset $dataset --seq $seq --logbase $logbase --fast --no_eval
done