script_name="$0"
DATASET_PATH=Datasets
PROGRAMS_PATH=Programs
DATASET_DEFAULT=1
PROGRAMS_DEFAULT=1
CONDA_ENV_NAME=""

while [[ $# -gt 0 ]]; do
    key="$1"

    case "$key" in
	-p)
	    PROGRAMS_PATH="$2"
	    PROGRAMS_DEFAULT=0
	    shift
	    shift
	    ;;
	--program_path=*)
	    PROGRAMS_PATH="${key#*=}"
	    PROGRAMS_DEFAULT=0
	    shift
	    ;;
	-d)
	    DATASET_PATH="$2"
	    DATASET_DEFAULT=0
	    shift
	    shift
	    ;;
	--dataset_path=*)
	    DATASET_PATH="${key#*=}"
	    DATASET_DEFAULT=0
	    shift
	    ;;
	-e)
	    CONDA_ENV_NAME="$2"
	    shift
	    shift
	    ;;
	--env_name=*)
	    CONDA_ENV_NAME="${key#*=}"
	    shift
	    ;;
	*)
	    cat <<EOF
Unknown argumment $key

Usage:
    
    $script_name [ -d <path> | --dataset_path=<path> ]
        [ -p <path> | --programs_path=<path> ]
        -e <name> | --env_name=<name>
EOF
	    exit 1
	    ;;
    esac
done

if [ -z "$CONDA_ENV_NAME" ]; then
    cat <<EOF
Missing environement name

Usage:
    
    $script_name [ -d <path> | --dataset_path=<path> ]
        [ -p <path> | --programs_path=<path> ]
        -e <name> | --env_name=<name>
EOF
    exit 1
fi
