script_name="$0"
DATASETS_DEFAULT=1
PROGRAMS_DEFAULT=1
SUMMARIES_DEFAULT=1

function help() {
cat <<EOF
Usage:

    $script_name [ -d <path> | --datasets_dir=<path> ]
        [ -p <path> | --programs_dir=<path> ]
        [ -s <path> | --summaries_dir=<path> ]

    -d where to install datasets
    -p where to install programs
    -s where to install summaries

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
	    PROGRAMS_DIR="$2"
	    PROGRAMS_DEFAULT=0
	    shift
	    shift
	    ;;
	--programs_dir=*)
	    PROGRAMS_DIR="${key#*=}"
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
	    DATASETS_DIR="$2"
	    DATASETS_DEFAULT=0
	    shift
	    shift
	    ;;
	--datasets_dir=*)
	    DATASETS_DIR="${key#*=}"
	    DATASETS_DEFAULT=0
	    shift
	    ;;
	-s)
            if [ -z "${2+x}" ]; then
                echo "-s requires an argument"
                echo ""
                help
                exit 1
            fi
	    SUMMARIES_DIR="$2"
	    SUMMARIES_DEFAULT=0
	    shift
	    shift
	    ;;
	--programs_dir=*)
	    SUMMARIES_DIR="${key#*=}"
	    SUMMARIES_DEFAULT=0
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

if [ $DATASETS_DEFAULT -eq 1 ]; then
    read -p "Install directory for datasets [$DATASETS_DIR]:" dpath
    if [ ! -z "$dpath" ]; then
        DATASETS_DIR="$dpath"
    fi
fi

if [ $PROGRAMS_DEFAULT -eq 1 ]; then
    read -p "Install directory for programs [$PROGRAMS_DIR]:" ppath
    if [ ! -z "$ppath" ]; then
        PROGRAMS_DIR="$ppath"
    fi
fi

if [ $SUMMARIES_DEFAULT -eq 1 ]; then
    read -p "Install directory for summaries [$SUMMARIES_DIR]:" ppath
    if [ ! -z "$ppath" ]; then
        SUMMARIES_DIR="$ppath"
    fi
fi



# make external_dirs.cfg
#[dir]
#datasets = /home/maksym/Datasets
#programs = /home/maksym/Programs
#summaries = /home/maksym/Summaries
echo -en "[dir]\ndatasets_dir=$DATASETS_DIR\nprograms_dir=$PROGRAMS_DIR\nsummaries_dir=$SUMMARIES_DIR" > external_dirs.cfg


mkdir -p "$DATASETS_DIR"
cd $DATASETS_DIR
echo $DATASETS_DIR
git clone --depth 1 https://github.com/MKorablyov/fragdb
git clone --depth 1 https://github.com/MKorablyov/brutal_dock
cd ..

mkdir -p "$PROGRAMS_DIR"
cd $PROGRAMS_DIR
git clone --depth 1 https://github.com/MKorablyov/dock6
git clone --depth 1 https://github.com/MKorablyov/chimera tmp
cd tmp
cat xaa xab > chimera.bin
chmod 755 chimera.bin
echo '../chimera' | ./chimera.bin
cd ..
rm -rf tmp
cd ..

mkdir -p "$SUMMARIES_DIR"
