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

For our example below we will be uploading the same .csv files as in our docDB walk through. Specifically: game_log, historical_pbp, historical_pbp_modelled

```python
#define your imports
import pandas as pd
import boto3
```
