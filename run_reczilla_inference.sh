cd RecSys2019_DeepLearning_Evaluation
python -m ReczillaClassifier.run_reczilla \
	--dataset_split_path="all_data/splits-v5/CiaoDVD/DataSplitter_leave_k_out_last" \
	--metamodel_filepath="../ReczillaModels/time_on_train.pickle"
	
python -m ReczillaClassifier.run_reczilla \
	--dataset_split_path="all_data/splits-v5/CiaoDVD/DataSplitter_leave_k_out_last" \
	--metamodel_filepath="../ReczillaModels/prec_10.pickle"
cd ..