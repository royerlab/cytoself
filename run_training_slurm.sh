set -e
cur_fname="$(basename $0 .sh)"
script_name=$(basename $0)
partition=gpu 

cmd="/hpc/mydata/james.burgess/.conda/envs/cytoself/bin/python scripts/train_cytoself_on_opencell_no_nucdist_balanced_classes.py"
cmd="/hpc/mydata/james.burgess/.conda/envs/cytoself/bin/python scripts/train_cytoself_on_opencell_balanced_classes.py"
mem="512gb"

sbatch <<< \
    "#!/bin/bash
#SBATCH --job-name=${cur_fname}-${partition}
#SBATCH --output=slurm_logs/${cur_fname}-${partition}-%j-out.txt
#SBATCH --error=slurm_logs/${cur_fname}-${partition}-%j-err.txt
#SBATCH --gpus=1
#SBATCH --mem=$mem
#SBATCH -c 2
#SBATCH -p $partition 
#SBATCH --time=144:00:00  
#SBATCH --mail-user=jmhb@stanford.edu
#SBATCH --mail-type=FAIL

echo \"$cmd\"
eval \"$cmd\"
"

