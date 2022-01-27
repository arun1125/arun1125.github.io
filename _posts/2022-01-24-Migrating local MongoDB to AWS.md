# Migrating Local Mongo DB to AWS



This is actually probably a part 4 in a longer series around how I went from inception to deployment of my in game win probability app for NBA games, the problem is I spent an entire weekend learning about how to deploy things to AWS and thought I had to write about it to others in a similar predicament and before I forgot

### MongoDB to AWS


AWS has two options for noSQL db's, dynamoDB and DocumentDB
Currently I am exploring how to set up dynamoDB but on my way to this point I figured out how to set up DocumentDB and it was kind of long to be honest

### Difference between DocumentDB and DynamoDB

DynamoDB is Serverless where as DocumentDB is not.
In terms of dollar value (which is what we all care about at the end of the day). DyanmoDB is pay per use/as you use resources and DocumentDB is pay hourly.

Also querying the databases are different - docDB can be intereacted with the mongodb driver where as dynamoDB has it's own api.

## Setting up Document DB

#### Steps

- 1) okay look yeah you're going to need an AWS account, this shit ain't free (well the free stuff is free but documentDB doesn't have a free tier) - (writing this after my first day of docDB charges and damn re-reading this hurt my soul) 
- 2) Access, AWS handles access through the IAM - Identity and Access Manager. You will need to create a new user and add the appropriate policies to it. AmazonDocDBFullAccess.
- 3) Create a VPC for your documentDB 
- 4) Create an EC2 instance 
    - why? okay so heres the annoying part about using documentDB it can only interact with other aws services within the same vpc. 
    - what does that mean for you? (or me) - well it means uploading the data from my local mongoDB is going to be a pain and we're going to use the EC2 instance as a sort of intermediary between us and documentDB
    
- 5) Create a security group so that your EC2 instance can actually connect to your documentDB
- 6) Create your DocumentDB cluster
- 7) Connect your EC2 instance to your Document DB cluster
- 8) Install the Mongo Shell on your EC2 instance
- 9) Manage TLS 
- 10) Test Connection with mongo shell 
- 11) Install Jupyter notebooks and pymongo on EC2 instance 
- 12) SCP your local CSV file to your EC2 instance
- 13) insert your data into documentDB w chunksize probably since you'll be choosing a cheap instance

For Steps 1 - 10 you can follow this document [https://docs.aws.amazon.com/documentdb/latest/developerguide/connect-ec2.html](https://docs.aws.amazon.com/documentdb/latest/developerguide/connect-ec2.html)

It will get you up and running with the services 

Step 11

- You can follow parts of this tutorial - https://chrisalbon.com/code/aws/basics/run_project_jupyter_on_amazon_ec2/
- _right click_ and *COPY LINK ADDRESS* https://www.anaconda.com/products/individual
- You won't need to set up a whole new virtual env as the reason for this ec2 instance is just to communicate with your docDB

Step 12

Now that you have your docDB cluster set up, your ec2 instance set up and theyre in the same vpc we can now start sending our data over.

- SCP/SFTP: You will need to ssh into your ec2 instance using the special .epm/.cer key that you downloaded following steps 1-10.

- Download save your mongoDB database as a csv
    - can be done through pd.DataFrame.from_records(db.collection.find()).to_csv('{name}.csv', index=False)

- 'Put' your csv into your ec2 instance. The three datasets I was putting were ~7k rows, ~3million rows, ~3million rows. I also used the free tier instance but if its just for the file transfer and you don't want to do anything with chunksize i'd say just provision a larger instance for the file transfer then shut it down

- Write your data to docDB, [how to programatically connect to docDB](https://docs.aws.amazon.com/documentdb/latest/developerguide/connect_programmatically.html)
    - if you provisioned the smallest instance then in your script to be memory efficient you will need to create an iterator for your dataset that reads and writes chunks at a time.
    

## Why I'm trying DynamoDB?

Using docDB as our primary database for our entire NBA ecosystem will slightly complicate any apps I deploy to EB regarding VPCs/Security Groups etc. Currently I haven't figured out how to connect to docDB from my docker container but as I was figuring this out guess what happened?  

They charged me 9$ (Freedom not Maple) for hosting my DB for a day ... infact it was overnight! once I saw this bill I said screw it i'm moving to dynamoDB for serverless hosting as they will charge depending on traffic! 

## Setting up DynamoDB

Setting up DynamoDB is A LOT easier than before. In fact it can be done through your local machine. 

#### Prerequisites 

- Have the Python SDK for AWS installed on your pc (it's called boto3, I don't know why though)
- Obviously have an AWS account lol 

_note_: the api is a little weird if you haven't seen stuff like this before, and i'm not too entirely comfortable with it either however, working through examples and being able to perform CRUD and some batch operations should get you up and running enough to go out and debug examples on your own

For our example below we will be uploading the same .csv files as in our docDB walk through. Specifically: game_log and historical_pbp_modelled

```python
#define your imports
import pandas as pd
import boto3
```

```python
#you make your connection to dynamoDB through the boto3 resources method
dynamoDB = boto3.resource('dynamodb', region_name = 'us-east-2')
```

```python
#you might have to include your accessID and secret ID from your IAM role if it isnt automatically detected
#in that case you just set up a session 
access_id = 'wouldnt you like to know'
secret_id = 'I aint telling you'

session = boto3.Session(access_id, secret_id)
dynamoDB = session.resource('dynamoDB', region_name = 'us-east-2')
```

Now that we have the dynamoDB object to interact with we can start to create tables but before we do that you need to know how you are going to use the table. Knowing this will allow us to design the appropriate index structure to for efficent lookups.

An Example below:

```python
game_log = pd.read_csv('../NBA/Data/game_log.csv').drop('_id', axis = 1)
game_log.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEASON_ID</th>
      <th>TEAM_ID</th>
      <th>TEAM_ABBREVIATION</th>
      <th>TEAM_NAME</th>
      <th>GAME_ID</th>
      <th>GAME_DATE</th>
      <th>MATCHUP</th>
      <th>WL</th>
      <th>MIN</th>
      <th>FGM</th>
      <th>...</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
      <th>PLUS_MINUS</th>
      <th>VIDEO_AVAILABLE</th>
      <th>Home</th>
      <th>Away</th>
      <th>home_team_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22015</td>
      <td>1610612744</td>
      <td>GSW</td>
      <td>Golden State Warriors</td>
      <td>21500003</td>
      <td>2015-10-27</td>
      <td>GSW vs. NOP</td>
      <td>W</td>
      <td>240</td>
      <td>41</td>
      <td>...</td>
      <td>8</td>
      <td>7</td>
      <td>20</td>
      <td>29</td>
      <td>111</td>
      <td>16</td>
      <td>1</td>
      <td>GSW</td>
      <td>NOP</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22015</td>
      <td>1610612741</td>
      <td>CHI</td>
      <td>Chicago Bulls</td>
      <td>21500002</td>
      <td>2015-10-27</td>
      <td>CHI vs. CLE</td>
      <td>W</td>
      <td>240</td>
      <td>37</td>
      <td>...</td>
      <td>6</td>
      <td>10</td>
      <td>13</td>
      <td>22</td>
      <td>97</td>
      <td>2</td>
      <td>1</td>
      <td>CHI</td>
      <td>CLE</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22015</td>
      <td>1610612737</td>
      <td>ATL</td>
      <td>Atlanta Hawks</td>
      <td>21500001</td>
      <td>2015-10-27</td>
      <td>ATL vs. DET</td>
      <td>L</td>
      <td>240</td>
      <td>37</td>
      <td>...</td>
      <td>9</td>
      <td>4</td>
      <td>15</td>
      <td>25</td>
      <td>94</td>
      <td>-12</td>
      <td>1</td>
      <td>ATL</td>
      <td>DET</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22015</td>
      <td>1610612747</td>
      <td>LAL</td>
      <td>Los Angeles Lakers</td>
      <td>21500017</td>
      <td>2015-10-28</td>
      <td>LAL vs. MIN</td>
      <td>L</td>
      <td>240</td>
      <td>35</td>
      <td>...</td>
      <td>2</td>
      <td>4</td>
      <td>14</td>
      <td>29</td>
      <td>111</td>
      <td>-1</td>
      <td>1</td>
      <td>LAL</td>
      <td>MIN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22015</td>
      <td>1610612756</td>
      <td>PHX</td>
      <td>Phoenix Suns</td>
      <td>21500014</td>
      <td>2015-10-28</td>
      <td>PHX vs. DAL</td>
      <td>L</td>
      <td>240</td>
      <td>34</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>18</td>
      <td>30</td>
      <td>95</td>
      <td>-16</td>
      <td>1</td>
      <td>PHX</td>
      <td>DAL</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



For our in game win probability use case we want the user:
- 1) To be able to select a date
- 2) Pick the games from this date they want to see the win probability for.

To achieve this with a regular SQL db, amazon RDS, mongo/noSQL db, docDB is pretty simple but with dynamoDB it can get a little tricky. The tricky part is in the inital learning but after that it becomes a lot clearer.


## DynamoDB key structure

- Hash Key(aka Partition Key): 
    - This is a required key
    - _Single_ Tables are defined as tables that ONLY have a Hash Key. 
        - A single table is something you generally want to avoid with dynamoDB because it limits you to one read at a time as it **only** enables the get_item method. I'm gyuessing this isn't what you want and that you'll probably want to query your data
    - This is the main key and _must_ be unique


unless you have a ...

- Sort Key(aka Range key): 
    - This is an optional second key but can be used in conjunction with the Hash key to create a _Composite_ index where your Hash/Sort Key pair HAS to be unique. 
    - Having a composite key will allow us to perform more complex methods to retrieve data from our dynamoDB including Query and Scan operations.
    
For our use case we will create a composite key consisting of our GAME_DATE as our HASH key and GAME_ID as our SORT key. 

The Reason I have mentioned these BEFORE we create the table is because once created we can't change it. Annoying, but is it really a bug or feature lol.


[Further Reading on this topic](https://dynobase.dev/dynamodb-keys/#:~:text=Is%20it%20possible%20to%20change,then%20remove%20the%20first%20table.) - There are other key/index structures like GSI's but they are currently out of scope. If they ever become in scope i'll write about them.

```python
# Creating the table

game_log_ddb = dynamodb.create_table(
        TableName='game_log',
        KeySchema=[
            {
                'AttributeName': 'GAME_DATE',
                'KeyType': 'HASH'  # Partition key
            },
            {
                'AttributeName': 'GAME_ID',
                'KeyType': 'RANGE' # Sort key
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'GAME_DATE',
                'AttributeType': 'S' #string
            },
            {
                'AttributeName': 'GAME_ID',
                'AttributeType': 'N' #number
            },

        ],
    BillingMode= 'PAY_PER_REQUEST', 
    ) 

# A quick note on Billing Mode: PAY_PER_REQUEST provisions resources depending on traffic
# and is useful if you don't know the frequency at which your db will get called
# otherwise you can pre provision resources which will throttle read and write speeds
# but for me since this app is a personal project I am going with PAY_PER_REQUEST for fast write speeds 
# during intial upload and I'll only be querying this db a handful of times a week. 
```

# Time to write our local data. 

We cant just dump our csv in like mongoDB with the insert_many method instead we will have to create a _batch_writer_ object and load each row indvidually but if you selected BillingMode=PAY_PER_REQUEST then the data even if its in a for loop will be written in parallely. 

Additionally, floats arent compatible so you'll have to convert all your floats to Decimal (which I didn't even know was a type until getting to this step)

```python
game_log_dict = game_log[~game_log['home_team_win'].isna()].to_dict(orient = 'records')
#how we convert float to decimal
game_log_json = [json.loads(json.dumps(item), parse_float=Decimal) for item in game_log_dict]
```

```python
from decimal import Decimal
import json

#if a float column has a nan you will have problems - again this is annoying, alternatively you can convert your
#data entirely to strings and then write it but Im going to get rid of the few rows that have nan in any float columns
#I am working with


with game_log_ddb.batch_writer() as batch:
    for i in range(len(game_log_json)):
        batch.put_item(Item = game_log_json[i])
        
        
#should take under a minute for the ~7500 rows I have in this dataset.
```

Doing this for the data that has the win probabilities ~3million rows took between 2-3 hours.

## How do you query the data? 

```python
from boto3.dynamodb.conditions import Key

#Example query. GAME_DATE is our HASH KEY
data = game_log_ddb.query(
    KeyConditionExpression=Key('GAME_DATE').eq('2022-01-01')
)
```

```python
#if you want to make a query with both GAME_DATE and GAME_ID then  

data = game_log_ddb.query(
    KeyConditionExpression=Key('GAME_DATE').eq(game_date) & Key('GAME_ID').eq(game_id)
)
```
