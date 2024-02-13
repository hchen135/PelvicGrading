for high_threshold in 0.8
do
	for low_threshold in 0.8
	do
		for threshold in 0.5
		do 
			echo "$low_threshold < $high_threshold"
			if (( $(echo "$low_threshold <= $high_threshold" |bc -l) ))
			then 
				echo $high_threshold
				python3 -W ignore main.py \
	                    --low_threshold=$low_threshold \
	                    --high_threshold=$high_threshold \
	                    --threshold=$threshold \
	                    --RESULT_PATH=/Users/hc/Documents/JHU/PJ/Mathias/AnnaProject/Pelvic/Journal_Extention/pelvic/final_results/out.json \
	                    --result_csv_name=all_results_graph1_visualization_zlow_change.csv
			fi
		done
	done
done

