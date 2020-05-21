script_name="$0"
DATASET_PATH=Datasets
PROGRAMS_PATH=Programs
DATASET_DEFAULT=1
PROGRAMS_DEFAULT=1

function help() {
cat <<EOF
Usage:

    $script_name [ -d <path> | --dataset_path=<path> ]
        [ -p <path> | --programs_path=<path> ]

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

if [ $DATASET_DEFAULT -eq 1 ]; then
    read -p "Install directory for datasets (NO to skip) [$DATASET_PATH]:" dpath
    if [ ! -z "$dpath" ]; then
        DATASET_PATH="$dpath"
    fi
fi

if [ "$DATASET_PATH" != "NO" ]; then
    mkdir -p "$DATASET_PATH"
    cd $DATASET_PATH
    git clone --depth 1 https://github.com/MKorablyov/fragdb
    git clone --depth 1 https://github.com/MKorablyov/brutal_dock
    cd ..
fi


if [ $PROGRAMS_DEFAULT -eq 1 ]; then
    read -p "Install directory for programs (NO to skip) [$PROGRAMS_PATH]:" ppath
    if [ ! -z "$ppath" ]; then
        PROGRAMS_PATH="$ppath"
    fi
fi

if [ "$PROGRAMS_PATH" != "NO" ]; then
    mkdir -p "$PROGRAMS_PATH"
    cd $PROGRAMS_PATH
    git clone --depth 1 https://github.com/MKorablyov/dock6
    git clone --depth 1 https://github.com/MKorablyov/chimera tmp
    cd tmp
    cat xaa xab > chimera.bin
    chmod 755 chimera.bin
    echo '../chimera' | ./chimera.bin
    cd ..
    rm -rf tmp
    cd ..
fi
