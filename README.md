# Decision-Trees-SGD
This project uses various decision tree algorithms to generate features for learning a linear separator.
The  data  includes a list of  names - both the first and last names. The labels (+ or -) are generated according to a linear function. 

In this project I extracted features from data in DataExtraction.py, then used multiple learning algorithms and reported their training accuracy, k-fold accuracy and testing accuracy. 

The algorithms I implemented were: 

(a) simple SGG: in SGD.py 

(b) full decisiontree (c) decision stump of depth four, (d) decision stump of depth eight: in DecisionTree.py

(e) SGD over features derived from 100 decision stump (by training  hundred  different  decisionstumps  of  maximum  depth  eight  on  the  entire  training  set): in DecisionStump.py

