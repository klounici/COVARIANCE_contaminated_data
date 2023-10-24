N_EXP=10
MAX_EPSILON=0.4
N_EPSILON=15

CONTAMINATION="bernoulli"
CONTOPTION="gauss"

dimensions=(
    50
    100
)

sample_sizes=(
    110
    200
)

effective_ranks=(
    2
    5
    10
)

intensities=(
    0.1
    0.5
    1
    2
)

# Bernoulli contamination
for dim in ${dimensions[@]};
do
    for sample_size in ${sample_sizes[@]};
    do
        for e_rank in ${effective_ranks[@]};
        do
            for intensity in ${intensities[@]};
            do
                python main.py --dim_size $dim --sample_size $sample_size --n_exp $N_EXP --effective_rank $e_rank --contamination $CONTAMINATION --method "all" --max_epsilon $MAX_EPSILON --n_epsilon $N_EPSILON --intensity $intensity --contoption $CONTOPTION
            done    
        done
    done
done



python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=20 --contamination=adversarial --method=tailMV --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=20 --contamination=adversarial --method=DDCMV --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=20 --contamination=adversarial --method=randomMV --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=20 --contamination=adversarial --method=classical --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=tailMV --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=DDCMV --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=randomMV --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=classical --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=tailMV --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=DDCMV --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=randomMV --max_epsilon=0.4 --n_epsilon=15
python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=classical --max_epsilon=0.4 --n_epsilon=15