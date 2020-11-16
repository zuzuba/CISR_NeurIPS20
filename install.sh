echo "Creating new conda environment, choose name"
read ENV_NAME
echo "Name $ENV_NAME was chosen";

conda create -n $ENV_NAME --yes python=3.7
source activate $ENV_NAME

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pip install -e $PROJECT_DIR
conda install --yes -c anaconda mpi4py

echo "# To activate this environment and use the library, use
#
#     $ conda activate $ENV_NAME
#
# To deactivate an active environment, use
#
#     $ conda deactivate
"
