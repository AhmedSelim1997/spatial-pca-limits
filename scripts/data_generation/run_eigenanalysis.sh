#!/bin/bash
## This code creates a parameters file inside the specified "params_dir" directory, and runs the snakemake protocol using this parameters file as an input



# Parse command line arguments
while [[ $# -gt 0 ]] ## while the number of arguments greater than zero
do
    key="$1" ## set the first read argument as the key

    case $key in ## set the second read argument as the parameter 
        -t|--tau)
        TAU="$2"
        shift ## This removes the command line argument
        shift ## applied twice to remove the argument and its parameter
        ;;
        -m|--migration) ## migration will be the variance of normal dispersal kernel in cont simulations
        MIG="$2"
        shift 
        shift 
        ;;
        -k|--num_demes)
        K="$2"
        shift
        shift 
        ;;
        -N|--pop_size)
        N="$2"
        shift
        shift 
        ;;
        -d|--demography)
        DEM="$2"
        shift 
        shift 
        ;;
        -o|--output_directory)
        output_directory="$2"
        shift 
        shift 
        ;;
        *)    # unknown option
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Set default values if not provided
TAU=${TAU:-0.0005}
MIG=${MIG:-0.01}
K=${K:-3}
DEM=${DEM:-"split"}
N=${N:-1000}
output_directory=${output_directory:-"./"}


## Create a unique name for the new internal output directory
if [[ "$DEM" = "SteppingStones_1d" || "$DEM" = "SteppingStones_2d" ]]; then
    output_directory="${output_directory}K=${K}/m=${MIG}/"
elif [[ "$DEM" = "split" ]]; then
    TAU_scientific=$(printf "%.2e" "$TAU")
    output_directory="${output_directory}tau=${TAU_scientific}/"
elif [[ "$DEM" = "cont" ]]; then
    output_directory="${output_directory}K=${K},N=${N}/m=${MIG}/"
fi

## Create a new output directory if it's not there
pwd 
if [ ! -d "$output_directory" ]; then
    # Directory does not exist, create it
    mkdir -p "$output_directory"
    echo "Directory created: $directory"
else
    echo "Directory already exists: $directory"
fi


CONFIG=$output_directory"config.yml"
echo $CONFIG
# Create configuration file
# This file stores the parameters needed for the snakefile that runs a simulation a generates eigenvalues and eigenvectors matrix

if [[ "$DEM" = "split" ]]; then

cat > "${CONFIG}" << EOF

    demography: ${DEM}
    output_path: ${output_directory}

    # Step 1 parameters
    simulation:
        N: ${N} 
        tau: !!float ${TAU}
        n: 1000
        L: 15000

    # Step 2 parameters
    eigen:
        n_min: 2
        n_max: 990
        n_num: 50
        
        L_min: 50
        L_max: 8000
        L_num: 20

        n_space: "lin"
        L_space: "geom"
        p: 1
EOF

elif [[ "$DEM" = "SteppingStones_1d" ]]; then
    
cat > "${CONFIG}" << EOF
    demography: ${DEM}
    output_path: ${output_directory}

    # Step 1 parameters
    simulation:
        N: ${N}
        m: !!float ${MIG}
        K: !!int ${K}
        n: 200 
        L: 10000

    # Step 2 parameters
    eigen:
        n_min: 2 
        n_max: 100 
        n_num: 50 

        L_min: 50
        L_max: 4000
        L_num: 50

        n_space: "lin"
        L_space: "geom"
        p: 2

EOF

elif [[ "$DEM" = "SteppingStones_2d" ]]; then
    
cat > "${CONFIG}" << EOF
    demography: ${DEM}
    output_path: ${output_directory}

    # Step 1 parameters
    simulation:
        N: ${N}
        m: !!float ${MIG}
        K: !!int ${K}
        n: 120
        L: 10000

    # Step 2 parameters
    eigen:
        n_min: 2
        n_max: 100
        n_num: 50

        L_min: 500
        L_max: 8000
        L_num: 20

        n_space: "lin"
        L_space: "geom"
        p: 4

EOF
    
elif [[ "$DEM" = "cont" ]]; then

cat > "${CONFIG}" << EOF
    demography: ${DEM}
    output_path: ${output_directory}

    # Step 1 parameters
    simulation:
        N: ${N}
        m: !!float ${MIG}
        K: !!int ${K}
        L: 15000

    # Step 2 parameters
    eigen:        
        n_min: 2
        n_max: 100
        n_num: 50
    
        # L_min: 500
        # L_max: 8000
        # L_num: 20

        n_space: "lin"
        L_space: "geom"
        p: 4

EOF

fi

# Run snakemake
snakemake --configfile "${CONFIG}" --cores all --notemp --printshellcmds
