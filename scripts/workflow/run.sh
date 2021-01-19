# good for local debugging

python -u -m torch.distributed.launch main.py --mode run-relaxations --config-yml configs/s2ef/200k/schnet/schnet.yml \
    --checkpoint /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-10-11-04-23-10-schnet_200k/checkpoint.pt \
    --distributed --debug

# python main.py --mode run-relaxations --config-yml configs/s2ef/200k/schnet/schnet.yml \
#     --sweep-yml configs/s2ef/sweeps/relax_opt.yml \
#     --checkpoint /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-10-11-04-23-10-schnet_200k/checkpoint.pt \
#     --num-gpus 8 --num-nodes 1 --distributed --slurm-partition learnfair --submit

# python main.py --mode run-relaxations --config-yml configs/s2ef/20M/dimenet/dimenetpp_1M.yml \
#     --sweep-yml configs/s2ef/sweeps/relax_opt.yml \
#     --checkpoint /private/home/anuroops/sync/baselines/dimenetpp/20M/checkpoints/2020-10-28-23-09-04-dimenetpp_1M_20M_run0/checkpoint.pt \
#     --amp --num-gpus 8 --num-nodes 1 --distributed --slurm-partition learnfair --submit

# python main.py --mode run-relaxations --config-yml configs/s2ef/20M/schnet/schnet.yml \
#     --sweep-yml configs/s2ef/sweeps/relax_opt.yml \
#     --checkpoint /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-10-05-03-02-48-schnet_20M_run0/checkpoint.pt \
#     --num-gpus 8 --num-nodes 1 --distributed --slurm-partition learnfair --submit

# benchmark best dimenet++ model
python main.py --mode run-relaxations --config-yml configs/s2ef/all-forceonly/dimenet-plus-plus/dpp_10.8M_ec0/dpp_10.8M_ec0.yml \
    --sweep-yml configs/s2ef/sweeps/relax_opt.yml \
    --checkpoint /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-11-16-23-34-24-dpp10.8M_forceonly_all_restart_ep2/checkpoint.pt \
    --num-gpus 8 --num-nodes 1 --distributed --slurm-partition learnfair --submit

# 34772878 (1st order), 34773530 (lbfgs)

# benchmark second best dimenet++ model
python main.py --mode run-relaxations --config-yml configs/s2ef/all-forceonly/dimenet-plus-plus/dpp_1.8M_ec0/dpp_ec0.yml \
    --sweep-yml configs/s2ef/sweeps/relax_opt.yml \
    --checkpoint /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-11-08-18-31-31-dpp1.8M_forceonly_all_restart_ep4.5/checkpoint.pt \
    --num-gpus 8 --num-nodes 1 --distributed --slurm-partition learnfair --submit

# 34773606 (1st order), 34774005 (lbfgs), 34775195 (adam_leftover)

# benchmark best schnet model
python main.py --mode run-relaxations --config-yml configs/s2ef/all-forceonly/schnet/schnet_ec0.yml \
    --sweep-yml configs/s2ef/sweeps/relax_opt.yml \
    --checkpoint /checkpoint/abhshkdz/ocp_baselines_run/checkpoints/2020-10-14-02-55-54-schnet_all_ec0/checkpoint.pt \
    --num-gpus 8 --num-nodes 1 --distributed --slurm-partition learnfair --submit
# 34774668 (1st order), 34775138 (lbfgs)
