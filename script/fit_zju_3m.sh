dataset="zju"
profile="zju_3m"
logbase=${profile}

#for seq in  "my_377" "my_386" "my_387" "my_392" "my_393"  "my_394"
for seq in "394" "393" "392" "390" "387" "386" "377" "315" "313"
do
    python solver.py --profile ./profiles/zju/${profile}.yaml --dataset $dataset --seq $seq --logbase $logbase --fast --no_eval
done