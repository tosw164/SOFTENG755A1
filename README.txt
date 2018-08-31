Project contains 4 folders, each containing a single .py script to run machine learning on that specific data set.
-------------------------------------------------------------------------------------------------------------------
World Cup: Classification and Regression
	- Run by calling either
		python3 worldcup.py --input FILE_PATH --type classification --model MODEL_NAME
		python3 worldcup.py --input FILE_PATH --type regression --model ASDF

Traffic: Regression
	- Run by calling
		python3 traffic.py --input FILE_PATH


Landsat & Occupancy: Classification
	- Run by calling 
		python3 landsat.py --input FILE_PATH --model MODEL_NAME
-------------------------------------------------------------------------------------------------------------------
NOTE: there is no error logic for arguments passed in CLI. Please ensure all information entered is correct
-------------------------------------------------------------------------------------------------------------------
--type values:
	regression = regression, enter anything for --model as it will run all regression models
	classification = classification. Please enter --model value

--input values: relative path of file to pwd

--model values: 
	SVM = SVM
	DT = Decision Tree
	NAIVE = Naive Bayes
	PERC = Perceptron
	KNN = K Nearest Neighbours
-------------------------------------------------------------------------------------------------------------------
