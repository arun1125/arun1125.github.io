# Migrating Local Mongo DB to AWS



This is actually probably a part 4 in a longer series around how I went from inception to deployment of my in game win probability app for NBA games, the problem is I spent an entire weekend learning about how to deploy things to AWS and thought I had to write about it to others in a similar predicament and before I forgot

### MongoDB to AWS


AWS has two options for noSQL db's, dynamoDB and DocumentDB
Currently I am exploring how to set up dynamoDB but on my way to this point I figured out how to set up DocumentDB and it was kind of long to be honest

### Difference between DocumentDB and DynamoDB

DynamoDB is Serverless where as DocumentDB is not.
In terms of dollar value (which is what we all care about at the end of the day). DyanmoDB is pay per use/as you use resources and DocumentDB is pay hourly

As I was writing this blog post one day of hosting on documentDB cost me $9 usd! I definitely tore down the instance and am currently in the process of moving to dynamoDB but knowing how to set up both noSQL databases will probably help you get more familiar with the entire ecosystem.

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
    

## FIN! 

### The problem though, as we move on to other parts of our application, ie deploying the frontend and backend, we will need them to be apart of the same VPC's 