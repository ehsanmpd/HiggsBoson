# Project 1 : Detecting Higgs Boson

This project was created as the first project of Machine Learning course [ **CS-433** ] at EPFL.

## Project Structure

The project is organized as follows:

    .
    ├── data                     # Train and test datasets.
        ├──sub_datasets          # Splitting datasets into 8 sub-datasets.

    ├── scripts                  # files to pre-process, train and test datasets.
    │   ├── helper_functions.py   
    │   ├── proj1_helpers.py     
    │   ├── implementations.py   

    ├── README.md                # README file.

    ├── report                   # The requested report.

    └── run.py                   # Script to train the regularized logestic regression model. This is the one that we submitted.
    
    └── run_ridge_regression.py  # Script to train the ridge regression model.

    └── run_test.py              # compare 6 different methods.

    └── Final_Prediction-Submitted.csv     # Final predictions (Categorical Accuracy = 0.837, F1-Score = 0.754)
    
## Running

Before running the code, please make sure that the train.csv and test.csv files are in the `data` folder. You just need to unzip the files : `data\train.zip` and `data\test.zip` in the data folder.

To reproduce our results, the one we submitted, you just need to run the following command:

``` 
python run.py
```

After running the script, you can find the generated predictions in the same folder that you are. Our final predictions are in the `Final_Prediction-Submitted.csv` file for comparison.

Note: After running the code, we make a sub-folder in data folder. So, if you want to run the code another time, please make sure that you have deleted this sub-folder before running. 

#------------------

Beside the regularized logestic regression metheod, we implemented the ridge regression method as well. To see the result of this method, you need to run the following command: 

``` 
python run_ridge_regression.py
```
#------------------

Finally, in order to make a comparison between 6 different methods, you need to run the following command: 

``` 
python run_test.py
```

At the end of this section, we just understand in our case the best way is using the regularized logestic regression method.

## Authors

* Ehsan Mohammadpour        ehsan.mohammadpour@epfl.ch
* Fereshte Mozafari         fereshte.mozafari@epfl.ch
* Mohammad Tohidivahdat     mohammad.vahdat@epfl.ch
