
# python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
#         --mode run-relaxations --config-yml configs/is2rs/dimenetpp_1M_20M_lbfgs.yml \
#         --run-dir dimenetpp --checkpoint dimenetpp.pt --distributed

python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
        --mode run-relaxations --config-yml configs/is2rs/dimenetpp_1M_20M_gd.yml \
        --run-dir dimenetpp --checkpoint dimenetpp.pt --distributed
