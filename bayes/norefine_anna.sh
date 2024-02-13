for high_threshold in $(seq 0.2 0.025 1.0)
do
	for low_threshold in 0
	do
		echo "$low_threshold < $high_threshold"
		if (( $(echo "$low_threshold < $high_threshold" |bc -l) ))
		then 
			echo $high_threshold
			python3 -W ignore main_norefine.py \
						--anna \
			            --low_threshold=$low_threshold \
	                    --high_threshold=$high_threshold \
	                    --threshold=0 \
	                    --RESULT_PATH=/Users/hc/Documents/JHU/PJ/Mathias/AnnaProject/Pelvic/Journal_Extention/pelvic/3d_results/IntensityShift0_ContrastScale1.0.json \
	                    --result_csv_name=all_results_no_refine_anna.csv
		fi
	done
done
