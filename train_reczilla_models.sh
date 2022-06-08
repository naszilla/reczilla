mkdir ReczillaModels
cd RecSys2019_DeepLearning_Evaluation
python -m ReczillaClassifier.run_reczilla \
    --train_meta \
	--metamodel_filepath="../ReczillaModels/prec_10.pickle" \
    --target_metric="PRECISION_cut_10" \
    --num_algorithms=10 \
    --num_metafeatures=10
	
python -m ReczillaClassifier.run_reczilla \
    --train_meta \
	--metamodel_filepath="../ReczillaModels/time_on_train.pickle" \
    --target_metric="time_on_train" \
    --num_algorithms=10 \
    --num_metafeatures=10
	
python -m ReczillaClassifier.run_reczilla \
    --train_meta \
	--metamodel_filepath="../ReczillaModels/ndcg_10.pickle" \
    --target_metric="NDCG_cut_10" \
    --num_algorithms=10 \
    --num_metafeatures=10
	
python -m ReczillaClassifier.run_reczilla \
	--train_meta \
	--metamodel_filepath="../ReczillaModels/mrr_10.pickle" \
	--target_metric="MRR_cut_10" \
	--num_algorithms=10 \
	--num_metafeatures=10

python -m ReczillaClassifier.run_reczilla \
    --train_meta \
	--metamodel_filepath="../ReczillaModels/item_hit_cov.pickle" \
    --target_metric="COVERAGE_ITEM_HIT_cut_10" \
    --num_algorithms=10 \
    --num_metafeatures=10
	
cd ..