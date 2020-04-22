script_name="$0"
DATASET_PATH=Datasets
PROGRAMS_PATH=Programs
DATASET_DEFAULT=1
PROGRAMS_DEFAULT=1
CONDA_ENV_NAME=""

function help() {
cat <<EOF
Usage:

    $script_name [ -d <path> | --dataset_path=<path> ]
        [ -p <path> | --programs_path=<path> ]
        -e <name> | --env_name=<name>

    -d NO will skip installing the datasets
    -p NO will skip installing the programs

    If you don't use the default values for dataset and program paths
    you will have to adjust the paths in the code to match.

EOF

}

while [[ $# -gt 0 ]]; do
    key="$1"

    case "$key" in
	-p)
            if [ -z "${2+x}" ]; then
                echo "-p requires an argument"
                echo ""
                help
                exit 1
            fi
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
            if [ -z "${2+x}" ]; then
                echo "-d requires an argument"
                echo ""
                help
                exit 1
            fi
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
            if [ -z "${2+x}" ]; then
                echo "-e requires an argument"
                echo ""
                help
                exit 1
            fi
	    CONDA_ENV_NAME="$2"
	    shift
	    shift
	    ;;
	--env_name=*)
	    CONDA_ENV_NAME="${key#*=}"
	    shift
	    ;;
        -h|--help)
            help
            exit 0
            ;;
	*)
            echo "Unknown argumment $key"
            echo ""
            help
	    exit 1
	    ;;
    esac
done

if [ -z "$CONDA_ENV_NAME" ]; then
    echo "Missing environement name"
    echo ""
    help
    exit 1
fi
