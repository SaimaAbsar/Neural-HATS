export PYTHONPATH="${PYTHONPATH}:../"

python run_grid_search_modified.py  \
  --experiment new-dataset          \
  --model gvar                      \
  --T 500                          \
  --num-sim 1                     \
  --K 10                             \
  --num-hidden-layers 1             \
  --hidden-layer-size 50            \
  --batch-size 64                   \
  --num-epochs 1000                 \
  --initial-lr 0.0001               \
  --seed 42 
  --graph './../../Data/A_6_ER_new.npy' \  
  --ci './../../CI_tests/outputs/CI_table_ER1.txt' \