# this file should be used without TSGS and DI, for very high dimensional data

N_EXP=10
MAX_EPSILON=0.93
N_EPSILON=20

CONTAMINATION="bernoulli"
CONTOPTION="gauss"

dimensions=(
    400
)

sample_sizes=(
    500
)

effective_ranks=(
    5
)

intensities=(
    4
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
                # python main.py --dim_size $dim --sample_size $sample_size --n_exp $N_EXP --effective_rank $e_rank --contamination $CONTAMINATION --method "all" --max_epsilon $MAX_EPSILON --n_epsilon $N_EPSILON --intensity $intensity --contoption $CONTOPTION
                # python main.py --dim_size $dim --sample_size $sample_size --n_exp $N_EXP --effective_rank $e_rank --contamination $CONTAMINATION --method "all" --max_epsilon $MAX_EPSILON --n_epsilon $N_EPSILON --intensity $intensity --contoption uniform
                python main.py --dim_size $dim --sample_size $sample_size --n_exp $N_EXP --effective_rank $e_rank --contamination $CONTAMINATION --method "classical" --max_epsilon $MAX_EPSILON --n_epsilon $N_EPSILON --intensity $intensity --contoption dirac
                python main.py --dim_size $dim --sample_size $sample_size --n_exp $N_EXP --effective_rank $e_rank --contamination $CONTAMINATION --method "oracleMV" --max_epsilon $MAX_EPSILON --n_epsilon $N_EPSILON --intensity $intensity --contoption dirac
                python main.py --dim_size $dim --sample_size $sample_size --n_exp $N_EXP --effective_rank $e_rank --contamination $CONTAMINATION --method "DDCMV95" --max_epsilon $MAX_EPSILON --n_epsilon $N_EPSILON --intensity $intensity --contoption dirac
                
            done    
        done
    done
done



# python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=20 --contamination=adversarial --method=tailMV --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=20 --contamination=adversarial --method=DDCMV --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=20 --contamination=adversarial --method=randomMV --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=20 --contamination=adversarial --method=classical --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=tailMV --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=DDCMV --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=200 --sample_size=500 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=randomMV --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=classical --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=tailMV --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=DDCMV --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=randomMV --max_epsilon=0.4 --n_epsilon=15
# python main.py --dim_size=500 --sample_size=800 --n_exp=10 --effective_rank=5 --contamination=adversarial --method=classical --max_epsilon=0.4 --n_epsilon=15

#python main.py --dim_size 50 --sample_size 100 --n_exp 10 --effective_rank 2 --contamination "bernoulli" --method "all+" --max_epsilon 0.3 --n_epsilon 15 --intensity 0.5 --contoption dirac
#python main.py --dim_size 50 --sample_size 100 --n_exp 10 --effective_rank 2 --contamination "bernoulli" --method "all+" --max_epsilon 0.4 --n_epsilon 15 --intensity 0.5 --contoption gauss
#python main.py --dim_size 50 --sample_size 100 --n_exp 10 --effective_rank 2 --contamination "bernoulli" --method "all+" --max_epsilon 0.3 --n_epsilon 15 --intensity 1.0 --contoption dirac
#python main.py --dim_size 50 --sample_size 100 --n_exp 10 --effective_rank 2 --contamination "bernoulli" --method "all+" --max_epsilon 0.4 --n_epsilon 15 --intensity 1.0 --contoption gauss
#python main.py --dim_size 50 --sample_size 100 --n_exp 10 --effective_rank 2 --contamination "bernoulli" --method "all+" --max_epsilon 0.3 --n_epsilon 15 --intensity 4.0 --contoption dirac
#python main.py --dim_size 50 --sample_size 100 --n_exp 10 --effective_rank 2 --contamination "bernoulli" --method "all+" --max_epsilon 0.4 --n_epsilon 15 --intensity 4.0 --contoption gauss
              