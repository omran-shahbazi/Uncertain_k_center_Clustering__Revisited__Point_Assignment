# Uncertain 𝑘-center Clustering, Revisited: Point Assignment

We put the data of `pokemon` and `crime` in the `data` folder. Since the volume of `taxi` data was large, we did not put it in folder `data`. You can get the `taxi` data from <a>https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew</a> and put it in the `data` folder after cleaning it.

To run the project, first run the bellow command to install dependencies:
```
pip install -r requirements.txt
```
Then run this command to run algorithms on `pokemon` and `crime` data (The results with the same name will be saved in the `results` folder. If it does not exist, create it before running the algorithms):

```
python src/main.py pokemon crime
```

In case you want to run the algorithm on your own data, first clean it with `pokemon` or `crime` data format and put it in the `data` folder with the desired name, for example `new_data.csv`. Then put the list of parameters which you want to run the algorithm on them into the `config.py` file:

```
new_data_parameters = [
    [n1, z_sqrt1, k1, b1],
    [n2, z_sqrt2, k2, b2],
    ...
]
```
Then add the corresponding field to the `parameters` dictionary:
```
parameters = {
    'pokemon': pokemon_parameters,
    'crime': crime_parameters,
    'taxi': taxi_parameters,
    'new_data': new_data_parameters
}

```
Finally run the bellow command to run algorithms on `new_data`:
```
python src/main.py new_data
```

Please note that we have considered a time limit of 2 hours for running the algorithms. If you want to change it, change function `is_time_limited` at the end of file `assignment.py`.