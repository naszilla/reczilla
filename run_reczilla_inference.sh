cd RecSys2019_DeepLearning_Evaluation

python -m ReczillaClassifier.run_reczilla \
	--dataset_split_path="all_data/splits-v5/AmazonGiftCards/DataSplitter_leave_k_out_last" \
	--metamodel_filepath="../ReczillaModels/prec_10.pickle" \
	--rec_model_save_path="../prec_10_"
	
read -p "Press enter to continue"
	
python -m ReczillaClassifier.run_reczilla ^
	--dataset_split_path="all_data/splits-v5/AmazonGiftCards/DataSplitter_leave_k_out_last" \
	--metamodel_filepath="../ReczillaModels/time_on_train.pickle" \
	--rec_model_save_path="../train_time_"
	
cd ..