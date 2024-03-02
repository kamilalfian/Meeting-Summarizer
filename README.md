# Meeting-Summarizer
This is a meeting summarizer. As the name suggests, it intends to make meeting summarization easier.

## Workflow
In general, this summarizer will go through 5 main stages:
1. Data Ingestion: Retrieving the data
2. Data Validation: Making sure that the data received is in the correct Huggingface format
3. Data Transformation: Cap entries to 4096 tokens, dropping entries that are too long
4. Model Trainer: Train existing model, either from Huggingface hub or local disk
5. Model Evaluation: Measure rouge metric, store metric and summarization in PostgreSQL database

## Paradigm
Each of 5 stages in workflow above is coded by repeating these general procedures
1. Update config.yaml
2. Update params.yaml
3. Update entity 
4. Update configuration manager 
5. Update components 
6. Update pipeline 
7. Update main.py 
8. Update app.py

## Quick start: How to run
1. Clone repo  
```git clone https://github.com/kamilalfian/Meeting-Summarizer.git```
2. Run app.py  
```python app.py```, this will take you to FastAPI localhost. You can play around with the model trainer or inferencer. Please be aware that you need very strong GPU to play with the model trainer.

## What's next
This project is built so it can be accessed from outside. For this reason DigitalOcean droplet is used. For those who are interested in building their own server using DO:
1. Sign up to DigitalOcean at digitalocean.com, create your droplet and registry there
2. Assign new secret variables in GitHUb secrets. Those variables are:  
a. secrets.HOST  (your droplet IPV4)
b. secrets.USERNAME  (root as default, needed to ssh login (ssh root@secrets.HOST))
c. secrets.SSHKEY  (your private droplet key, you can generate it along with the public key by SSH logging in and then type ```ssh-keygen -t ed25519 -a 200 -C "your_email@example.com"```)  
d. secrets.PASSPHRASE  (basically just a password)  
e. secrets.DIGITALOCEAN_ACCESS_TOKEN  (your DigitalOcean API Key, get it here https://cloud.digitalocean.com/account/api/tokens)
3. Don't forget to also edit your authorized keys inside your VM by SSH logging in (ssh root@secrets.HOST), and then type ```nano authorized_keys``` and then replace the content with the public key you generated before
4. Download PGAdmin and PostgreSQL package. Sign up.
5. Use the credentials you used for signing up to PGAdmin to set up another github secrets:  
secrets.DB_HOSTNAME  (your hostname)
secrets.DB_DATABASE  (your database name)
secrets.DB_USERNAME  (your username)   
secrets.DB_PWD  (the password that you created when setting up new DB)  
secrets.DB_PORT  (the port number, ex: 5432)  
6. You can now finally start deploying your project using Github Action by doing git add, git commit, and git push