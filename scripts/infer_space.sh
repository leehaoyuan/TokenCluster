python3 src/inference.py \
	--summary_data /afs/cs.unc.edu/home/haoyuanl/SemAE/data/space/space_summ.json \
	--gold_data /afs/cs.unc.edu/home/haoyuanl/SemAE/data/space/gold \
	--sentencepiece /afs/cs.unc.edu/home/haoyuanl/SemAE/data/sentencepiece/spm_unigram_32k.model \
        --model /afs/cs.unc.edu/home/haoyuanl/SemAE/models/spacev2_7_model.pt \
	--gpu 0 \
	--run_id spacev27_mean \
	--outdir outputs \
	--max_tokens 75 \
	--min_tokens 4 \
        --no_cut_sents \
        --cos_thres 0.5 ;
