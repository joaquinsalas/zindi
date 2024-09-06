
# Original band information

I am using the test partition.

Band statistics

|Band |	Mean   |	SD     |	Max  |	Min|
|-----|--------|---------|-------|-----|
|Blue |	1509.37|	1011.62|	20000|-9999|
|Green|	1509.19|	1011.38|	20000|-9999|
|Red  |	1506.84|	1010.13|	20000|-9999|
|NIR  |	1506.67|	1009.91|	20000|-9999|
|SWIR1|	1507.48|	1009.74|	20000|-9999|
|SWIR2|	1507.33|	1009.57|	20000|-9999|



![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/Blue_orig.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/Green_orig.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/Red_orig.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/NIR_orig.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/SWIR1_orig.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/SWIR2_orig.png?raw=true)

# Preprocessing

I chop the values of each band between 0 and 6,000. These are the new results

|Band	|Mean	  |SD     |	Max	|Min|
|-----|-------|-------|-----|---|
|Blue	|1513.63|	980.20|	6000|	0 |
|Green|1513.42|	980.10|	6000|	0 |
|Red	|1511.09|	978.71|	6000|	0 |
|NIR	|1510.91|	978.60|	6000|	0 |
|SWIR1|1511.73|	978.29|	6000|	0 |
|SWIR2|1511.56|	978.23|	6000|	0 |

 ![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/Blue.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/Green.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/Red.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/NIR.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/SWIR1.png?raw=true)
![alt text](https://github.com/joaquinsalas/zindi/blob/main/code/data_exploration/SWIR2.png?raw=true)

The linear correlation coefficient between the bands is


|	    |Blue |Green|	Red	|NIR	 |SWIR1|SWIR2|
|-----|-----|-----|-----|-----|-----|-----|     
|Blue	|     |0.985|0.954|0.941|0.926|0.920|
|Green|     |	    |0.972|0.954|0.933|0.926|
|Red	 |     |     |     |0.985|0.954|0.941|
|NIR	 |     |     |     |	    |0.972|0.954|
|SWIR1|     |     |     |     |     |0.985|
|SWIR2|     |     |     |     |     | 	   |


La matriz de informacion mutua normalizada es 

|	    |Blue |Green|	Red	|NIR	 |SWIR1|SWIR2|
|-----|-----|-----|-----|-----|-----|-----|     
|Blue	|    |0.255|0.192|0.175|0.161|0.156|
|Green|    |     |0.234|0.192|0.169|0.161|
|Red	 |    |     | 	   |0.255|0.192|0.175|
|NIR	 |    |     |     | 	   |0.234|0.192|
|SWIR1|    |     |     |     | 	   |0.255|
|SWIR2|    |     |     |     |     |	    |





