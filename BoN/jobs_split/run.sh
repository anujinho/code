# loop over all sbatch files in the directory, print the filename and submit the job to SLURM
#

for FILE in *.sbatch; do
    if [[ "${FILE}" == *"pytorch"* ]]; then
        echo "Not submitting template."
    else
        echo ${FILE}
        sbatch ${FILE}
        sleep 1
    fi
done